//--------------------------------------------------------------------------------
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met :
//
//	*Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//	
//	* Neither the name of the copyright holder nor the names of its
//	contributors may be used to endorse or promote products derived from
//	this software without specific prior written permission.
//	
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//	DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//	OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Copyright(c) 2019, Sergen Eren
// All rights reserved.
//----------------------------------------------------------------------------------
// 
//	Version 1.0: Sergen Eren, 21/10/2019
//
// File: Main entry point for render kernel. 
//
//-----------------------------------------------


#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include "helper_math.h"
#include <assert.h>
#include <vector>
#include <random>
#undef APIENTRY

#include "hdr_loader.h"
#include "kernel_params.h"
#include "bitmap_image.h"

// new classes
#include "gpu_vdb/gpu_vdb.h"
#include "gpu_vdb/camera.h"
#include "light.h"
#include "atmosphere.h"
#include "bvh/bvh_builder.h"

// Image Writers 

#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

// File parsers
#include "instancer_hda/volume_instance.h"

//#define SAVE_TGA
#define SAVE_OPENEXR

#include <Windows.h>

// Third Party

#define OIDN_STATIC_LIB
#include <OpenImageDenoise/oidn.hpp>


// Atmosphere

CUmodule cuRenderModule;
CUmodule cuTextureModule;
CUfunction cuRaycastKernel;
CUfunction cuTextureKernel;

std::vector<GPU_VDB> unique_vdb_files;
std::vector<GPU_VDB> instances;
std::vector<vdb_instance> volume_files;

static int num_volumes = 3; // TODO: read number of instances from json file 
BVH_Builder bvh_builder;

// Cam parameters 
camera	cam;
float3	lookfrom;
float3	lookat;
float3	vup;
float	fov;
float	aspect;
float	aperture;

// Atmosphere

atmosphere earth_atmosphere;


#define check_success(expr) \
    do { \
        if(!(expr)) { \
            fprintf(stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            exit(EXIT_FAILURE); \
        } \
    } while(false)

// Env sampling functions
static bool solveQuadratic(float a, float b, float c, float& x1, float& x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return false;
		x1 = 0; x2 = sqrtf(-c / a);
		return true;
	}
	float discr = b * b - 4 * a * c;

	if (discr < 0) return false;

	float q = (b < 0.f) ? -0.5f * (b - sqrtf(discr)) : -0.5f * (b + sqrtf(discr));
	x1 = q / a;
	x2 = c / q;

	return true;
}

// check ray against earth and atmosphere upper bound
static bool raySphereIntersect(const float3& orig, const float3& dir, const float& radius, float& t0, float& t1)
{

	float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
	float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

	if (!solveQuadratic(A, B, C, t0, t1)) return false;

	if (t0 > t1) {
		float tempt = t1;
		t1 = t0;
		t0 = tempt;
	}
	return true;
}

// Degree to radians conversion
static float degree_to_radians(float degree)
{

	return degree * float(M_PI) / 180.0f;

}

// Polar coordinates to direction
static float3 degree_to_cartesian(float azimuth, float elevation)
{

	float az = clamp(azimuth, .0f, 360.0f);
	float el = clamp(elevation, .0f, 90.0f);

	az = degree_to_radians(az);
	el = degree_to_radians(90.0f - el);

	float x = sinf(el) * cosf(az);
	float y = cosf(el);
	float z = sinf(el) * sinf(az);

	return normalize(make_float3(x, y, z));
}

// Draw a sample from sky
static float3 sample_atmosphere(const Kernel_params &kernel_params, const float3 orig, const float3 dir, const float3 intensity)
{

	// initial parameters
	float	atmosphereRadius = 6420e3f;
	float3	sunDirection = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);
	float	earthRadius = 6360e3f;
	float	Hr = 7994.0f;
	float	Hm = 1200.0f;
	float3	betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
	float3	betaM = make_float3(21e-6f);
	//


	float t0, t1;
	float tmin, tmax = FLT_MAX;
	float3 pos = orig;
	pos.y += 1000 + 6360e3f;

	if (raySphereIntersect(pos, dir, earthRadius, t0, t1) && t1 > .0f) tmax = fmaxf(.0f, t0);
	tmin = .0f;
	if (!raySphereIntersect(pos, dir, atmosphereRadius, t0, t1) || t1 < 0) return make_float3(1.0f, .0f, .0f);
	if (t0 > tmin && t0 > 0) tmin = t0;
	if (t1 < tmax) tmax = t1;

	uint numSamples = 16;
	uint numSamplesLight = 8;

	float segmentLength = (tmax - tmin) / numSamples;
	float tCurrent = tmin;
	float3 sumR = make_float3(0.0f, .0f, .0f); // Rayleigh contribution
	float3 sumM = make_float3(0.0f, .0f, .0f); // Mie contribution

	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
	float phaseR = 3.f / (16.f * float(M_PI)) * (1 + mu * mu);
	float g = 0.76f;

	float phaseM = 3.f / (8.f * float(M_PI)) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

	for (uint i = 0; i < numSamples; ++i) {
		float3 samplePosition = pos + (tCurrent + segmentLength * 0.5f) * dir;
		float height = length(samplePosition) - earthRadius;
		// compute optical depth for light
		float hr = exp(-height / Hr) * segmentLength;
		float hm = exp(-height / Hm) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float t0Light, t1Light;
		raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
		float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
		float opticalDepthLightR = 0, opticalDepthLightM = 0;
		uint j;
		for (j = 0; j < numSamplesLight; ++j) {
			float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
			float heightLight = length(samplePositionLight) - earthRadius;
			if (heightLight < 0) break;
			opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
			tCurrentLight += segmentLengthLight;
		}
		if (j == numSamplesLight) {
			float3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
			float3 attenuation = make_float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
		tCurrent += segmentLength;
	}


	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * intensity;
}

// Save Exr Images
static bool save_exr(float4 *rgba, int width, int height, std::string filename) {

	filename.append(".exr");

	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = 4;

	std::vector<float> images[4];
	for (int i = 0; i < image.num_channels; i++) images[i].resize(width*height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int idx = i * width + j;

			// Flip image vertically
			i = height - i - 1;
			int idx2 = i * width + j;

			images[0][idx] = rgba[idx2].x;
			images[1][idx] = rgba[idx2].y;
			images[2][idx] = rgba[idx2].z;
			images[3][idx] = rgba[idx2].w;

		}
	}

	float* image_ptr[4];

	image_ptr[0] = &(images[3].at(0)); // A
	image_ptr[1] = &(images[2].at(0)); // B
	image_ptr[2] = &(images[1].at(0)); // G
	image_ptr[3] = &(images[0].at(0)); // R

	image.images = (unsigned char**)image_ptr;
	image.width = width;
	image.height = height;

	header.num_channels = 4;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
	strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
	strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
	strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err = NULL; // or nullptr in C++11 or later.
	int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
	if (ret != TINYEXR_SUCCESS) {
		fprintf(stderr, "Save EXR err: %s\n", err);
		return false;
	}
	printf("Saved exr file. [ %s ] \n", filename.c_str());

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);

	return true;
}

static bool save_exr(float3 *rgba, int width, int height, std::string filename) {

	filename.append(".exr");

	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = 3;

	std::vector<float> images[3];
	for (int i = 0; i < image.num_channels; i++) images[i].resize(width*height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int idx = i * width + j;

			// Flip image vertically
			i = height - i - 1;
			int idx2 = i * width + j;

			images[0][idx] = rgba[idx2].x;
			images[1][idx] = rgba[idx2].y;
			images[2][idx] = rgba[idx2].z;
		}
	}

	float* image_ptr[3];

	image_ptr[0] = &(images[2].at(0)); // B
	image_ptr[1] = &(images[1].at(0)); // G
	image_ptr[2] = &(images[0].at(0)); // R

	image.images = (unsigned char**)image_ptr;
	image.width = width;
	image.height = height;

	header.num_channels = 3;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
	strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
	strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err = NULL; // or nullptr in C++11 or later.
	int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
	if (ret != TINYEXR_SUCCESS) {
		fprintf(stderr, "Save EXR err: %s\n", err);
		return false;
	}
	printf("Saved exr file. [ %s ] \n", filename.c_str());

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);

	return true;
}

static bool save_tga(float4 *rgba, int width, int height, std::string filename) {

	filename.append(".tga");

	unsigned char* image_data;
	image_data = new unsigned char[(width * height * 3)];

	int idx = 0;
	for (int i = 0; i < width * height; i++) {

		unsigned char ir = (unsigned int)(255.0f * fminf(fmaxf(rgba[i].x, 0.0f), 1.0f));
		unsigned char ig = (unsigned int)(255.0f * fminf(fmaxf(rgba[i].y, 0.0f), 1.0f));
		unsigned char ib = (unsigned int)(255.0f * fminf(fmaxf(rgba[i].z, 0.0f), 1.0f));

		image_data[idx++] = ir;
		image_data[idx++] = ig;
		image_data[idx++] = ib;

	}

	stbi_flip_vertically_on_write(1);
	int success = stbi_write_tga(filename.c_str(), width, height, 3, image_data);

	if (!success) return false;

	return true;
}

// Initialize GLFW and GLEW.
static GLFWwindow *init_opengl()
{

	check_success(glfwInit());
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);


	GLFWwindow *window = glfwCreateWindow(1280, 640, "volume path tracer", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Error creating OpenGL window.\n");;
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);

	const GLenum res = glewInit();
	if (res != GLEW_OK) {
		fprintf(stderr, "GLEW error: %s.\n", glewGetErrorString(res));
		glfwTerminate();
	}

	glfwSwapInterval(0);

	check_success(glGetError() == GL_NO_ERROR);
	return window;
}

// Initialize CUDA with OpenGL interop.
static void init_cuda()
{
	int cuda_devices[1];
	unsigned int num_cuda_devices;
	check_success(cudaGLGetDevices(&num_cuda_devices, cuda_devices, 1, cudaGLDeviceListAll) == cudaSuccess);
	if (num_cuda_devices == 0) {
		fprintf(stderr, "Could not determine CUDA device for current OpenGL context\n.");
		exit(EXIT_FAILURE);
	}

	check_success(cudaSetDevice(cuda_devices[0]) == cudaSuccess);
	cuInit(0);
}

// Utility: add a GLSL shader.
static void add_shader(GLenum shader_type, const char *source_code, GLuint program)
{
	GLuint shader = glCreateShader(shader_type);
	check_success(shader);
	glShaderSource(shader, 1, &source_code, NULL);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	check_success(success);

	glAttachShader(program, shader);
	check_success(glGetError() == GL_NO_ERROR);
}

// Create a simple GL program with vertex and fragement shader for texture lookup.
static GLuint create_shader_program()
{
	GLint success;
	GLuint program = glCreateProgram();

	const char *vert =
		"#version 330\n"
		"in vec3 Position;\n"
		"out vec2 TexCoord;\n"
		"void main() {\n"
		"    gl_Position = vec4(Position, 1.0);\n"
		"    TexCoord = 0.5 * Position.xy + vec2(0.5);\n"
		"}\n";
	add_shader(GL_VERTEX_SHADER, vert, program);

	const char *frag =
		"#version 330\n"
		"in vec2 TexCoord;\n"
		"out vec4 FragColor;\n"
		"uniform sampler2D TexSampler;\n"
		"void main() {\n"
		"    FragColor = texture(TexSampler, TexCoord);\n"
		"}\n";
	add_shader(GL_FRAGMENT_SHADER, frag, program);

	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		fprintf(stderr, "Error linking shadering program\n");
		glfwTerminate();
	}

	glUseProgram(program);
	check_success(glGetError() == GL_NO_ERROR);

	return program;
}

// Create a quad filling the whole screen.
static GLuint create_quad(GLuint program, GLuint* vertex_buffer)
{
	static const float3 vertices[6] = {
		{ -1.f, -1.f, 0.0f },
		{ -1.f,  1.f, 0.0f },
		{  1.f, -1.f, 0.0f },
		{  1.f, -1.f, 0.0f },
		{  1.f,  1.f, 0.0f },
		{ -1.f,  1.f, 0.0f }
	};

	glGenBuffers(1, vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	GLuint vertex_array;
	glGenVertexArrays(1, &vertex_array);
	glBindVertexArray(vertex_array);

	const GLint pos_index = glGetAttribLocation(program, "Position");
	glEnableVertexAttribArray(pos_index);
	glVertexAttribPointer(
		pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);

	check_success(glGetError() == GL_NO_ERROR);

	return vertex_array;
}

// Context structure for window callback functions.
struct Window_context
{
	int zoom_delta;

	bool moving;
	bool panning;
	bool save_image;
	double move_start_x, move_start_y;
	double move_dx, move_dy, move_mx, move_my;

	float exposure;

	bool change;
	unsigned int config_type;
};

// GLFW scroll callback.
static void handle_scroll(GLFWwindow *window, double xoffset, double yoffset)
{
	Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));
	if (yoffset > 0.0)
		ctx->zoom_delta = 1;
	else if (yoffset < 0.0)
		ctx->zoom_delta = -1;
}

// GLFW keyboard callback.
static void handle_key(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	float dist = 0;

	if (action == GLFW_PRESS) {
		Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		case GLFW_KEY_KP_SUBTRACT:
		case GLFW_KEY_LEFT_BRACKET:
			fov -= 10.0f;
			cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);
			ctx->change = true;
			break;
		case GLFW_KEY_KP_ADD:
		case GLFW_KEY_RIGHT_BRACKET:
			fov += 10.0f;
			cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);
			ctx->change = true;
			break;
		case GLFW_KEY_S:
			ctx->save_image = true;
			break;
		case GLFW_KEY_F: // Frame camera to include objects

			float3 bbox_min = make_float3(.0f);
			float3 bbox_max = make_float3(.0f);;

			for(GPU_VDB vdb: instances) { 
			
				bbox_min = fminf(bbox_min ,vdb.get_xform().transpose().transform_point(vdb.vdb_info.bmin));
				bbox_max = fmaxf(bbox_max ,vdb.get_xform().transpose().transform_point(vdb.vdb_info.bmax));
			
			}

			float3 center = (bbox_max + bbox_min) / 2;
			dist = length(bbox_max - bbox_min); // diagonal length of gpu_vdb object

			lookat = center;
			lookfrom = make_float3(center.x + (dist), center.y + (dist), center.z + (dist));
			cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);

			ctx->change = true;
			break;
		default:
			break;
		}
	}
}

// GLFW mouse button callback.
static void handle_mouse_button(GLFWwindow *window, int button, int action, int mods)
{
	Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));
	bool imgui_hover = ImGui::IsAnyWindowHovered();

	if (button == GLFW_MOUSE_BUTTON_LEFT && !imgui_hover) {
		if (action == GLFW_PRESS) {
			ctx->moving = true;
			glfwGetCursorPos(window, &ctx->move_start_x, &ctx->move_start_y);
		}
		else
			ctx->moving = false;
	}

	if (button == GLFW_MOUSE_BUTTON_MIDDLE && !imgui_hover) {
		if (action == GLFW_PRESS) {
			ctx->panning = true;
			glfwGetCursorPos(window, &ctx->move_start_x, &ctx->move_start_y);
		}
		else
			ctx->panning = false;
	}
}

// GLFW mouse position callback.
static void handle_mouse_pos(GLFWwindow *window, double xpos, double ypos)
{
	Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));
	if (ctx->moving)
	{
		ctx->move_dx += xpos - ctx->move_start_x;
		ctx->move_dy += ypos - ctx->move_start_y;
		ctx->move_start_x = xpos;
		ctx->move_start_y = ypos;
	}
	if (ctx->panning)
	{
		ctx->move_mx += xpos - ctx->move_start_x;
		ctx->move_my += ypos - ctx->move_start_y;
		ctx->move_start_x = xpos;
		ctx->move_start_y = ypos;
	}
}

// Resize OpenGL and CUDA buffers for a given resolution.
static void resize_buffers(
	float3 **accum_buffer_cuda,
	float4 **raw_buffer_cuda,
	cudaGraphicsResource_t *display_buffer_cuda,
	int width, int height,
	GLuint display_buffer)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	check_success(glGetError() == GL_NO_ERROR);

	if (*display_buffer_cuda)
		check_success(cudaGraphicsUnregisterResource(*display_buffer_cuda) == cudaSuccess);
	check_success(
		cudaGraphicsGLRegisterBuffer(
			display_buffer_cuda, display_buffer, cudaGraphicsRegisterFlagsWriteDiscard) == cudaSuccess);

	if (*accum_buffer_cuda)
		check_success(cudaFree(*accum_buffer_cuda) == cudaSuccess);
	check_success(cudaMalloc(accum_buffer_cuda, width * height * sizeof(float3)) == cudaSuccess);

	if (*raw_buffer_cuda)
		check_success(cudaFree(*raw_buffer_cuda) == cudaSuccess);
	check_success(cudaMalloc(raw_buffer_cuda, width * height * sizeof(float4)) == cudaSuccess);

}

// TO-DO: make gpu do this.  
static bool create_cdf(
	Kernel_params &kernel_params,
	cudaArray_t *env_val_data,
	cudaArray_t *env_func_data,
	cudaArray_t *env_cdf_data,
	cudaArray_t *env_marginal_func_data,
	cudaArray_t *env_marginal_cdf_data)
{
	if (kernel_params.debug) {
		printf("\ncreating cdf and function textures for environment...");
	}

	// Fill the value, function, marginal and cdf values
	//----------------------------------------------------------------------

	float3 pos = make_float3(0.0f, 0.0f, 0.0f);
	const unsigned res = 180;
	kernel_params.env_sample_tex_res = res;

	float az = 0;
	float el = 0;

	float3 *val = new float3[res*res], *val_p = val;							//RGB values of env sky
	float *func = new float[res*res], *func_p = func;							// Luminous power of sky
	float *cdf = new float[res*res], *cdf_p = cdf;							// constructed CDF of directions 
	float *marginal_func = new float[res], *marginal_func_p = marginal_func;		// values for marginal distribution
	float *marginal_cdf = new float[res], *marginal_cdf_p = marginal_cdf;			// cdf for marginal distribution

	memset(val, 0x0, sizeof(float3) * res * res);
	memset(func, 0x0, sizeof(float) * res * res);
	memset(cdf, 0x0, sizeof(float) * res * res);
	memset(marginal_func, 0x0, sizeof(float) * res);
	memset(marginal_cdf, 0x0, sizeof(float) * res);

	*val_p = make_float3(0.0f, 0.0f, 0.0f);
	*func_p = .0f;
	*cdf_p = .0f;

	for (int y = 0; y < res; ++y, ++marginal_func_p) {
		el = float(y) / float(res - 1) * float(M_PI);			// elevation goes from 0 to 180 degrees
		*(cdf_p - 1) = .0f;
		for (int x = 0; x < res; ++x, ++val_p, ++func_p, ++cdf_p) {

			az = float(x) / float(res - 1) * float(M_PI) * 2.0f;		// azimuth goes from 0 to 360 degrees 

			float3 dir = make_float3(sinf(el) * cosf(az), cosf(el), sinf(el) * sinf(az)); // polar to cartesian 			
			*val_p = sample_atmosphere(kernel_params, pos, dir, kernel_params.sky_color);
			*func_p = length((*val_p));
			*cdf_p = *(cdf_p - 1) + *(func_p - 1) / (res);
		}

		*marginal_func_p = *(cdf_p - 1);
	}

	//reset pointers
	val_p = val;
	func_p = func;
	cdf_p = cdf;
	marginal_func_p = marginal_func;

	float total_int = 0.0f;
	for (int j = 0; j < res; j++)
	{
		total_int += *marginal_func_p;
	}
	marginal_func_p = marginal_func;

	if (total_int == .0f) {
		for (int y = 0; y < res; ++y) {
			for (int x = 0; x < res; ++x, ++cdf_p) {
				*cdf_p = (float(x) / float(res)) * (float(y) / float(res));
			}
		}
	}

	else {
		for (int y = 0; y < res; y++, ++marginal_func_p) {
			for (int x = 0; x < res; ++x, ++cdf_p) {
				*cdf_p /= *marginal_func_p;
				if (x == res - 1) *cdf_p = 1.0f;//Last element of cdf must be 1
			}
		}
	}


	// Construct marginal distribution cdf array
	marginal_func_p = marginal_func;
	*marginal_cdf_p = 0.0f;

	for (int y = 0; y < res; ++y, ++marginal_func_p, ++marginal_cdf_p) {

		*marginal_cdf_p = *(marginal_cdf_p - 1) + *marginal_func_p / res;
		//printf("\n%d	%f",y ,*marginal_func_p);
	}
	float marginal_int = *(marginal_cdf_p - 1);
	kernel_params.env_marginal_int = marginal_int;
	//printf("\nmarginal distribution integral is %f", marginal_int);

	//divide cdf values with total marginal func integral
	marginal_cdf_p = marginal_cdf;

	if (marginal_int > .0f) {
		for (int y = 0; y < res; ++y, ++marginal_func_p, ++marginal_cdf_p) {
			*marginal_cdf_p /= fmaxf(.000001f, marginal_int);
			//printf("\n%d	%f", y, *marginal_cdf_p);
		}
	}
	*marginal_cdf_p = 1.0f;

	// End array filling
	//------------------------------------------------------------------------------------



	// Send data to GPU 
	//-------------------------------------------------------------------------------------

	// Send val data

	float4 *texture = new float4[res*res];
	for (int i = 0; i < res*res; i++) {
		texture[i] = make_float4(val[i].x, val[i].y, val[i].z, 1.0f);
	}

	const cudaChannelFormatDesc channel_desc_val = cudaCreateChannelDesc<float4>();
	check_success(cudaMallocArray(env_val_data, &channel_desc_val, res, res) == cudaSuccess);
	check_success(cudaMemcpy2DToArray(*env_val_data, 0, 0, texture, res * sizeof(float4), res * sizeof(float4), res, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc_val;
	memset(&res_desc_val, 0, sizeof(res_desc_val));
	res_desc_val.resType = cudaResourceTypeArray;
	res_desc_val.res.array.array = *env_val_data;

	cudaTextureDesc tex_desc_val;
	memset(&tex_desc_val, 0, sizeof(tex_desc_val));
	tex_desc_val.addressMode[0] = cudaAddressModeWrap;
	tex_desc_val.addressMode[1] = cudaAddressModeClamp;
	tex_desc_val.addressMode[2] = cudaAddressModeWrap;
	tex_desc_val.filterMode = cudaFilterModeLinear;
	tex_desc_val.readMode = cudaReadModeElementType;
	tex_desc_val.normalizedCoords = 1;

	check_success(cudaCreateTextureObject(&kernel_params.sky_tex, &res_desc_val, &tex_desc_val, NULL) == cudaSuccess);


	// Send func data
	const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
	check_success(cudaMallocArray(env_func_data, &channel_desc, res, res) == cudaSuccess);
	check_success(cudaMemcpy2DToArray(*env_func_data, 0, 0, func, res * sizeof(float), res * sizeof(float), res, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc_func;
	memset(&res_desc_func, 0, sizeof(res_desc_func));
	res_desc_func.resType = cudaResourceTypeArray;
	res_desc_func.res.array.array = *env_func_data;

	cudaTextureDesc tex_desc_func;
	memset(&tex_desc_func, 0, sizeof(tex_desc_func));
	tex_desc_func.addressMode[0] = cudaAddressModeWrap;
	tex_desc_func.addressMode[1] = cudaAddressModeClamp;
	tex_desc_func.addressMode[2] = cudaAddressModeWrap;
	tex_desc_func.filterMode = cudaFilterModePoint;
	tex_desc_func.readMode = cudaReadModeElementType;
	tex_desc_func.normalizedCoords = 0;

	check_success(cudaCreateTextureObject(&kernel_params.env_func_tex, &res_desc_func, &tex_desc_func, NULL) == cudaSuccess);

	// Send cdf data 

	check_success(cudaMallocArray(env_cdf_data, &channel_desc, res, res) == cudaSuccess);
	check_success(cudaMemcpy2DToArray(*env_cdf_data, 0, 0, cdf, res * sizeof(float), res * sizeof(float), res, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc_cdf;
	memset(&res_desc_cdf, 0, sizeof(res_desc_cdf));
	res_desc_cdf.resType = cudaResourceTypeArray;
	res_desc_cdf.res.array.array = *env_cdf_data;

	cudaTextureDesc tex_desc_cdf;
	memset(&tex_desc_cdf, 0, sizeof(tex_desc_cdf));
	tex_desc_cdf.addressMode[0] = cudaAddressModeWrap;
	tex_desc_cdf.addressMode[1] = cudaAddressModeClamp;
	tex_desc_cdf.addressMode[2] = cudaAddressModeWrap;
	tex_desc_cdf.filterMode = cudaFilterModePoint;
	tex_desc_cdf.readMode = cudaReadModeElementType;
	tex_desc_cdf.normalizedCoords = 0;

	check_success(cudaCreateTextureObject(&kernel_params.env_cdf_tex, &res_desc_cdf, &tex_desc_cdf, NULL) == cudaSuccess);

	// Send Marginal 1D distribution func data

	check_success(cudaMallocArray(env_marginal_func_data, &channel_desc, res, 0) == cudaSuccess);
	check_success(cudaMemcpy2DToArray(*env_marginal_func_data, 0, 0, marginal_func, res * sizeof(float), res * sizeof(float), 1, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc_marginal_func;
	memset(&res_desc_marginal_func, 0, sizeof(res_desc_marginal_func));
	res_desc_marginal_func.resType = cudaResourceTypeArray;
	res_desc_marginal_func.res.array.array = *env_marginal_func_data;

	cudaTextureDesc tex_desc_marginal_func;
	memset(&tex_desc_marginal_func, 0, sizeof(tex_desc_marginal_func));
	tex_desc_marginal_func.addressMode[0] = cudaAddressModeWrap;
	tex_desc_marginal_func.addressMode[1] = cudaAddressModeClamp;
	tex_desc_marginal_func.addressMode[2] = cudaAddressModeWrap;
	tex_desc_marginal_func.filterMode = cudaFilterModePoint;
	tex_desc_marginal_func.readMode = cudaReadModeElementType;
	tex_desc_marginal_func.normalizedCoords = 0;

	check_success(cudaCreateTextureObject(&kernel_params.env_marginal_func_tex, &res_desc_marginal_func, &tex_desc_marginal_func, NULL) == cudaSuccess);


	// Send Marginal 1D distribution cdf data

	check_success(cudaMallocArray(env_marginal_cdf_data, &channel_desc, res, 0) == cudaSuccess);
	check_success(cudaMemcpy2DToArray(*env_marginal_cdf_data, 0, 0, marginal_cdf, res * sizeof(float), res * sizeof(float), 1, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc_marginal_cdf;
	memset(&res_desc_marginal_cdf, 0, sizeof(res_desc_marginal_cdf));
	res_desc_marginal_cdf.resType = cudaResourceTypeArray;
	res_desc_marginal_cdf.res.array.array = *env_marginal_cdf_data;

	cudaTextureDesc tex_desc_marginal_cdf;
	memset(&tex_desc_marginal_cdf, 0, sizeof(tex_desc_marginal_cdf));
	tex_desc_marginal_cdf.addressMode[0] = cudaAddressModeWrap;
	tex_desc_marginal_cdf.addressMode[1] = cudaAddressModeWrap;
	tex_desc_marginal_cdf.filterMode = cudaFilterModePoint;
	tex_desc_marginal_cdf.readMode = cudaReadModeElementType;
	tex_desc_marginal_cdf.normalizedCoords = 0;

	check_success(cudaCreateTextureObject(&kernel_params.env_marginal_cdf_tex, &res_desc_marginal_cdf, &tex_desc_marginal_cdf, NULL) == cudaSuccess);



	//End host to device data migration
	//------------------------------------------------------------------------------------------

	// render texture images if requested
	//------------------------------------------------------------------------------------------
#if RENDER_ENV_SAMPLE_TEXTURES

	if (CreateDirectory("./env_sample", NULL) || ERROR_ALREADY_EXISTS == GetLastError());
	else {

		printf("\nError: unable to create directory for environment sample textures\n");
		exit(-1);

	};

	std::ofstream ofs_val("./env_sample/val.ppm", std::ios::out | std::ios::binary);
	ofs_val << "P6\n" << res << " " << res << "\n255\n";

	std::ofstream ofs_func("./env_sample/func.ppm", std::ios::out | std::ios::binary);
	ofs_func << "P6\n" << res << " " << res << "\n255\n";

	std::ofstream ofs_cdf("./env_sample/cdf.ppm", std::ios::out | std::ios::binary);
	ofs_cdf << "P6\n" << res << " " << res << "\n255\n";

	std::ofstream ofs_marginal_func("./env_sample/marginal_func.ppm", std::ios::out | std::ios::binary);
	ofs_marginal_func << "P6\n" << 1 << " " << res << "\n255\n";

	std::ofstream ofs_marginal_cdf("./env_sample/marginal_cdf.ppm", std::ios::out | std::ios::binary);
	ofs_marginal_cdf << "P6\n" << 1 << " " << res << "\n255\n";

	val_p = val;
	func_p = func;
	cdf_p = cdf;
	marginal_func_p = marginal_func;
	marginal_cdf_p = marginal_cdf;

	for (unsigned j = 0; j < res; ++j, ++marginal_func_p, ++marginal_cdf_p)
	{
		ofs_marginal_func << (unsigned char)(min(1.0f, (*marginal_func_p)) * 255)
			<< (unsigned char)(min(1.0f, (*marginal_func_p)) * 255)
			<< (unsigned char)(min(1.0f, (*marginal_func_p)) * 255);

		ofs_marginal_cdf << (unsigned char)(min(1.0f, (*marginal_cdf_p)) * 255)
			<< (unsigned char)(min(1.0f, (*marginal_cdf_p)) * 255)
			<< (unsigned char)(min(1.0f, (*marginal_cdf_p)) * 255);

		for (unsigned i = 0; i < res; ++i, ++val_p, ++func_p, ++cdf_p)
		{
			(*val_p).x = (*val_p).x < 1.413f ? pow((*val_p).x * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*val_p).x);
			(*val_p).y = (*val_p).y < 1.413f ? pow((*val_p).y * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*val_p).y);
			(*val_p).z = (*val_p).z < 1.413f ? pow((*val_p).z * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*val_p).z);

			ofs_val << (unsigned char)(min(1.0f, (*val_p).x) * 255)
				<< (unsigned char)(min(1.0f, (*val_p).y) * 255)
				<< (unsigned char)(min(1.0f, (*val_p).z) * 255);

			ofs_func << (unsigned char)(min(1.0f, (*func_p)) * 255)
				<< (unsigned char)(min(1.0f, (*func_p)) * 255)
				<< (unsigned char)(min(1.0f, (*func_p)) * 255);

			ofs_cdf << (unsigned char)(min(1.0f, (*cdf_p)) * 255)
				<< (unsigned char)(min(1.0f, (*cdf_p)) * 255)
				<< (unsigned char)(min(1.0f, (*cdf_p)) * 255);

		}
	}
	ofs_val.close();
	ofs_func.close();
	ofs_cdf.close();
	ofs_marginal_func.close();
	ofs_marginal_cdf.close();

#endif


	delete[] val, func, cdf;
	return true;

}

// Create enviroment texture.
static bool create_environment(
	cudaTextureObject_t *env_tex,
	cudaArray_t *env_tex_data,
	const char *envmap_name)
{
	unsigned int rx, ry;
	float *pixels;
	if (!load_hdr_float4(&pixels, &rx, &ry, envmap_name)) {
		fprintf(stderr, "error loading environment map file %s\n", envmap_name);
		return false;
	}

	const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
	check_success(cudaMallocArray(env_tex_data, &channel_desc, rx, ry) == cudaSuccess);

	check_success(cudaMemcpy2DToArray(*env_tex_data, 0, 0, pixels, rx * sizeof(float4), rx * sizeof(float4), ry, cudaMemcpyHostToDevice) == cudaSuccess);

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = *env_tex_data;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeWrap;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	check_success(cudaCreateTextureObject(env_tex, &res_desc, &tex_desc, NULL) == cudaSuccess);
	return true;
}

static bool load_blue_noise(float3 **buffer, std::string filename, int &width, int &height) {

	// Load blue noise texture from assets directory and send to gpu 
	bitmap_image image(filename.c_str());

	if (!image)
		return false;

	width = image.width();
	height = image.height();

	float3 *values = new float3[height * width];

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {

			rgb_t color;

			image.get_pixel(x, y, color);
			int idx = y * width + x;
			values[idx].x = float(color.red) / 255.0f;
			values[idx].y = float(color.blue) / 255.0f;
			values[idx].z = float(color.green) / 255.0f;

		}
	}

	check_success(cudaMalloc(buffer, width * height * sizeof(float3)) == cudaSuccess);
	check_success(cudaMemcpy(*buffer, values, width * height * sizeof(float3), cudaMemcpyHostToDevice) == cudaSuccess);

	return true;
}

static void read_instance_file(std::string file_name) {
	
	assert(!file_name.empty());

	std::ifstream stream;
	stream.open(file_name.c_str());

	std::string num_vdbs;
	std::getline(stream, num_vdbs);
	std::istringstream iss(num_vdbs);

	int num_volumes;
	iss >> num_volumes;
	std::cout << "number of vdbs: " << num_volumes << "\n";
	volume_files.resize(num_volumes);

	for (int i = 0; i < num_volumes; ++i) {
		std::string vdb_file_name;
		std::getline(stream, vdb_file_name);
		volume_files.at(i).vdb_file = vdb_file_name.c_str();

		std::string num_instances;
		std::getline(stream, num_instances);
		std::istringstream nis(num_instances);
		nis >> volume_files.at(i).num_instances;

		volume_files.at(i).instances.resize(volume_files.at(i).num_instances);

		for (int x = 0; x < volume_files.at(i).num_instances; ++x) {
			
			std::string instance_parameters;
			std::getline(stream, instance_parameters);
			std::istringstream params(instance_parameters);
			params >> volume_files.at(i).instances.at(x).position[0]
				>> volume_files.at(i).instances.at(x).position[1]
				>> volume_files.at(i).instances.at(x).position[2]
				>> volume_files.at(i).instances.at(x).rotation[0]
				>> volume_files.at(i).instances.at(x).rotation[1]
				>> volume_files.at(i).instances.at(x).rotation[2]
				>> volume_files.at(i).instances.at(x).rotation[3]
				>> volume_files.at(i).instances.at(x).scale;
			
			std::cout << " position x: " << volume_files.at(i).instances.at(x).position[0];
			std::cout << " position y: " << volume_files.at(i).instances.at(x).position[1];
			std::cout << " position z: " << volume_files.at(i).instances.at(x).position[2];

			std::cout << " rotation x: " << volume_files.at(i).instances.at(x).rotation[0];
			std::cout << " rotation y: " << volume_files.at(i).instances.at(x).rotation[1];
			std::cout << " rotation z: " << volume_files.at(i).instances.at(x).rotation[2];
			std::cout << " rotation w: " << volume_files.at(i).instances.at(x).rotation[3];

			std::cout << " scale: " << volume_files.at(i).instances.at(x).scale << "\n";
		}
	}

	for (int i = 0; i < volume_files.size(); ++i) {

		unique_vdb_files.push_back(GPU_VDB());
		unique_vdb_files.at(i).loadVDB(volume_files.at(i).vdb_file, "density");

		for (int x = 0; x < volume_files.at(i).num_instances; ++x) {
			
			GPU_VDB new_instance(GPU_VDB(unique_vdb_files.at(i)));
			
			mat4 xform = unique_vdb_files.at(i).get_xform();
			
			// Translate with instance position
			xform.translate(make_float3(
				volume_files.at(i).instances.at(x).position[0],
				volume_files.at(i).instances.at(x).position[1],
				volume_files.at(i).instances.at(x).position[2]	));
			
			// Apply instance rotation
			mat4 rotation_matrix = quaternion_to_mat4(
				volume_files.at(i).instances.at(x).rotation[0],
				volume_files.at(i).instances.at(x).rotation[1],
				volume_files.at(i).instances.at(x).rotation[2],
				volume_files.at(i).instances.at(x).rotation[3]);

			//xform = xform * rotation_matrix;

			// Set scale
			xform.scale(make_float3(volume_files.at(i).instances.at(x).scale));

			new_instance.set_xform(xform);
			instances.push_back(new_instance);
		}
	}

}

// Process camera movement.
static void update_camera(double dx, double dy, double mx, double my, int zoom_delta, const atmosphere &atm)
{
	float rot_speed = 1;
	float zoom_speed = 500;
	float dist = length(lookfrom - lookat);
	// Rotation

	lookfrom -= cam.u * float(dx) * rot_speed  * dist * 0.01f;
	lookfrom += cam.v * float(dy) * rot_speed  * dist * 0.01f;

	//Pan
	lookfrom -= cam.u * float(mx) * rot_speed;
	lookfrom += cam.v * float(my) * rot_speed;
	lookat -= cam.u * float(mx) * rot_speed;
	lookat += cam.v * float(my) * rot_speed;

	// Zoom 
	lookfrom -= cam.w * float(zoom_delta) * zoom_speed;

	float3 earth_center = make_float3(.0f, -atm.atmosphere_parameters.bottom_radius, .0f);

	if (length(lookfrom - earth_center) < atm.atmosphere_parameters.bottom_radius) lookfrom += normalize(lookfrom - earth_center) * (atm.atmosphere_parameters.bottom_radius - length(lookfrom - earth_center));


	cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);

}


static void update_debug_buffer(
	float3 **debug_buffer_cuda,
	Kernel_params kernel_params)
{
	if (*debug_buffer_cuda)	check_success(cudaFree(*debug_buffer_cuda) == cudaSuccess);
	check_success(cudaMalloc(debug_buffer_cuda, 1000 * sizeof(float3)) == cudaSuccess);
}


int main(const int argc, const char* argv[])
{


	if (argc < 2) {
		printf("Please specify a vdb file!");
		return 0;
	}

	Window_context window_context;
	memset(&window_context, 0, sizeof(Window_context));

	GLuint display_buffer = 0;
	GLuint display_tex = 0;
	GLuint program = 0;
	GLuint quad_vertex_buffer = 0;
	GLuint quad_vao = 0;
	GLFWwindow *window = NULL;
	int width = -1;
	int height = -1;
	window_context.change = false;
	window_context.save_image = false;
	// Init OpenGL window and callbacks.
	window = init_opengl();
	glfwSetWindowUserPointer(window, &window_context);
	glfwSetKeyCallback(window, handle_key);
	glfwSetScrollCallback(window, handle_scroll);
	glfwSetCursorPosCallback(window, handle_mouse_pos);
	glfwSetMouseButtonCallback(window, handle_mouse_button);

	glGenBuffers(1, &display_buffer);
	glGenTextures(1, &display_tex);
	check_success(glGetError() == GL_NO_ERROR);

	program = create_shader_program();
	quad_vao = create_quad(program, &quad_vertex_buffer);

	init_cuda();


	// SETUP IMGUI PARAMETERS
	const char* glsl_version = "#version 330";

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO(); (void)io;
	io.ConfigWindowsMoveFromTitleBarOnly = true;
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Setup gpu_vdb

	std::string fname;
	if (argc >= 2) fname = argv[1];

	read_instance_file(fname);

	// Setup modules and contexes 

	const char *render_module_name = "render_kernel.ptx";
	const char *texture_module_name = "texture_kernels.ptx";
	const char *render_kernel_name = "volume_rt_kernel";
	const char *texture_kernel_name = "calculate_textures";

	int cuda_devices[1];
	unsigned int num_cuda_devices;
	check_success(cudaGLGetDevices(&num_cuda_devices, cuda_devices, 1, cudaGLDeviceListAll) == cudaSuccess);
	if (num_cuda_devices == 0) {
		fprintf(stderr, "Could not determine CUDA device for context\n.");
		exit(EXIT_FAILURE);
	}

	CUresult error;

	error = cuModuleLoad(&cuRenderModule, render_module_name);
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleLoad, %i\n", error);
	error = cuModuleGetFunction(&cuRaycastKernel, cuRenderModule, render_kernel_name);
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleGetFunction, %i\n", error);

	error = cuModuleLoad(&cuTextureModule, texture_module_name);
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleLoad, %i\n", error);
	error = cuModuleGetFunction(&cuTextureKernel, cuTextureModule, texture_kernel_name);
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleGetFunction, %i\n", error);


	// Send volume instances to gpu

	CUdeviceptr d_volume_ptr;
	check_success(cuMemAlloc(&d_volume_ptr, sizeof(GPU_VDB) * instances.size()) == cudaSuccess);
	check_success(cuMemcpyHtoD(d_volume_ptr, instances.data(), sizeof(GPU_VDB) * instances.size()) == cudaSuccess);


	// Create BVH from vdb vector 
	AABB scene_bounds(make_float3(.0f), make_float3(.0f));
	bvh_builder.m_debug_bvh = false;
	bvh_builder.build_bvh(instances, instances.size(), scene_bounds);

	// Setup initial camera 
	lookfrom = make_float3(1300.0f, 77.0f, 0.0f);
	lookat = make_float3(-10.0f, 72.0f, -43.0f);
	vup = make_float3(.0f, 1.0f, .0f);
	fov = 30.0f;
	aspect = 1.0f;
	aperture = 0.0f;

	cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);


	// Setup point Lights
#if defined(__CAMERA_H__) || defined(__LIGHT_H__) 
#undef rand // undefine the rand coming from camera.h and light.h

	float3 min = instances[0].get_xform().transpose().transform_point(instances[0].vdb_info.bmin);
	float3 max = instances[0].get_xform().transpose().transform_point(instances[0].vdb_info.bmax);

	std::mt19937 e2(std::random_device{}());
	std::uniform_real_distribution<> dist(0, 1);

	int num_lights = 0;
	
	light_list l_list(num_lights);
	l_list.light_ptr = (point_light*)malloc(num_lights * sizeof(point_light));
	cudaMallocManaged(&l_list.light_ptr, num_lights * sizeof(point_light));

	for (int i = 0; i < num_lights; ++i) {
		float3 pos = min + make_float3(dist(e2) * (max.x - min.x), dist(e2) * (max.y - min.y), dist(e2) * (max.z - min.z));
		float3 color = make_float3(dist(e2), dist(e2), dist(e2));
		
		l_list.light_ptr[i] = point_light();
		l_list.light_ptr[i].color = color;
		l_list.light_ptr[i].pos = pos;
		l_list.light_ptr[i].power = 1000.0f;
	}

	CUdeviceptr d_lights;
	check_success(cuMemAlloc(&d_lights, sizeof(light_list)) == cudaSuccess);
	check_success(cuMemcpyHtoD(d_lights, &l_list, sizeof(light_list)) == cudaSuccess);
#endif 

	// Setup initial render kernel parameters.

	// Kernel param buffers

	float3 *accum_buffer = NULL;
	float4 *raw_buffer = NULL;
	cudaGraphicsResource_t display_buffer_cuda = NULL;
	float3 *debug_buffer = NULL;

	Kernel_params kernel_params;
	memset(&kernel_params, 0, sizeof(Kernel_params));
	kernel_params.render = true;
	kernel_params.iteration = 0;
	kernel_params.max_interactions = 100;
	kernel_params.exposure_scale = 1.0f;
	kernel_params.environment_type = 0;
	kernel_params.ray_depth = 10;
	kernel_params.phase_g1 = 0.0f;
	kernel_params.phase_g2 = 0.0f;
	kernel_params.phase_f = 1.0f;
	kernel_params.tr_depth = 1.0f;
	kernel_params.density_mult = 1.0f;
	kernel_params.albedo = make_float3(1.0f, 1.0f, 1.0f);
	kernel_params.extinction = make_float3(1.0f, 1.0f, 1.0f);
	kernel_params.azimuth = 150;
	kernel_params.elevation = 30;
	kernel_params.sun_color = make_float3(1.0f, 1.0f, 1.0f);
	kernel_params.sun_mult = 1.0f;
	kernel_params.energy_inject = 0.0f;
	kernel_params.sky_color = make_float3(1.0f, 1.0f, 1.0f);
	kernel_params.sky_mult = 1.0f;
	kernel_params.env_sample_tex_res = 360;
	kernel_params.integrator = 0;

	std::string bn_path = ASSET_PATH;
	bn_path.append("BN0.bmp");

	float3 *bn_buffer = NULL;
	int bn_width, bn_height;
	load_blue_noise(&bn_buffer, bn_path, bn_width, bn_height);
	kernel_params.blue_noise_buffer = bn_buffer;
	update_debug_buffer(&debug_buffer, kernel_params);
	kernel_params.debug_buffer = debug_buffer;

	//kernel parameters env data
	cudaArray_t env_val_data = 0;
	cudaArray_t env_tex_data = 0;
	cudaArray_t env_func_data = 0;
	cudaArray_t env_cdf_data = 0;
	cudaArray_t env_marginal_func_data = 0;
	cudaArray_t env_marginal_cdf_data = 0;
	bool env_tex = false;

	// Imgui Parameters

	int max_interaction = 100;
	float max_extinction = 1.0f;
	int ray_depth = 10;
	double energy = .0f;
	float azimuth = 120.0f;
	float elevation = 30.0f;
	int integrator = 0;
	bool render = true;
	bool debug = false;
	bool denoise = false;
	int frame = 0;
	float rot_amount = 0.0f;

	static const char *items[] = {"NONE", "APPROXIMATE", "PRECOMPUTED"};
	static int env_comp = 0;
	int temp_env_comp = 0;

	bool use_constant_solar_spectrum = true;
	bool use_ozone = true;
	bool do_white_balance = true;
	float exposure = 1.0f;

	// End ImGui parameters

	//Create env texture 
	if (argc >= 3) {

		std::string env_tex_name = ASSET_PATH;
		env_tex_name.append(argv[2]);
		env_tex = create_environment(&kernel_params.env_tex, &env_tex_data, env_tex_name.c_str());
	}
	if (env_tex) {
		kernel_params.environment_type = 1;
		window_context.config_type = 2;
	}

	// Create env map sampling textures
	create_cdf(kernel_params, &env_val_data, &env_func_data, &env_cdf_data, &env_marginal_func_data, &env_marginal_cdf_data);

	// Init atmosphere 
	
#ifdef DEBUG_TEXTURES

	if (CreateDirectory("./atmosphere_textures", NULL) || ERROR_ALREADY_EXISTS == GetLastError());
	else {

		printf("\nError: unable to create directory for atmosphere textures\n");
		exit(-1);

	};

	earth_atmosphere.init();
	AtmosphereParameters *atmos_params = &earth_atmosphere.atmosphere_parameters;

#endif // DEBUG_TEXTURES

	
	// Create OIDN devices 
	oidn::DeviceRef oidn_device = oidn::newDevice();
	oidn_device.commit();
	oidn::FilterRef filter = oidn_device.newFilter("RT");

	// Main loop

	while (!glfwWindowShouldClose(window)) {

		// Process events.
		glfwPollEvents();
		Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));

		// Update kernel params
		kernel_params.exposure_scale = powf(2.0f, ctx->exposure);
		kernel_params.max_interactions = max_interaction;
		kernel_params.ray_depth = ray_depth;
		kernel_params.render = render;
		kernel_params.azimuth = azimuth;
		kernel_params.elevation = elevation;
		kernel_params.debug = debug;
		if (energy == 0) kernel_params.energy_inject = 1.0;
		else kernel_params.energy_inject = 1.0 + (energy / 100000.0);

		const unsigned int volume_type = ctx->config_type & 1;
		const unsigned int environment_type = env_tex ? ((ctx->config_type >> 1) & 1) : 0;


		// Update atmosphere
		earth_atmosphere.m_use_constant_solar_spectrum = use_constant_solar_spectrum;
		earth_atmosphere.m_use_ozone = use_ozone;
		earth_atmosphere.m_do_white_balance = do_white_balance;
		earth_atmosphere.m_exposure = exposure;
		// Draw imgui 
		//-------------------------------------------------------------------

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Parameters window");
		ImGui::Checkbox("Render", &render);
		ImGui::Checkbox("Denoise", &denoise);
		ImGui::SliderFloat("exposure", &ctx->exposure, -10.0f, 10.0f);
		ImGui::InputInt("Max interactions", &max_interaction, 1);
		ImGui::InputInt("Ray Depth", &ray_depth, 1);
		ImGui::InputInt("Integrator", &integrator, 0);
		ImGui::Checkbox("debug", &debug);
		ImGui::SliderFloat("phase g1", &kernel_params.phase_g1, -1.0f, 1.0f);
		ImGui::SliderFloat("phase g2", &kernel_params.phase_g2, -1.0f, 1.0f);
		ImGui::SliderFloat("phase f", &kernel_params.phase_f, 0.0f, 1.0f);
		ImGui::InputFloat("Density Multiplier", &kernel_params.density_mult);
		ImGui::InputFloat("Depth Multiplier", &kernel_params.tr_depth);
		ImGui::InputFloat3("Volume Extinction", (float *)&kernel_params.extinction);
		ImGui::InputFloat3("Volume Color", (float *)&kernel_params.albedo);
		ImGui::InputDouble("Energy Injection", &energy, 0.0);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);


		ImGui::Begin("Atmosphere Parameters");
		ImGui::SliderFloat("Sky Exposure", &exposure, -10.0f, 10.0f);
		ImGui::ColorEdit3("Sun Color", (float *)&kernel_params.sun_color);
		ImGui::InputFloat("Sun Multiplier", &kernel_params.sun_mult, 0.0f, 100.0f);
		ImGui::InputFloat3("Sky Color", (float *)&kernel_params.sky_color);
		ImGui::InputFloat("Sky Multiplier", &kernel_params.sky_mult, 0.0f, 100.0f);
		ImGui::SliderFloat("Azimuth", &azimuth, 0, 360);
		ImGui::SliderFloat("Elevation", &elevation, -90, 90);
		ImGui::Combo("Env computation", &env_comp, items, IM_ARRAYSIZE(items));
		ImGui::Checkbox("Const Solar Spectrum", &use_constant_solar_spectrum);
		ImGui::Checkbox("Use Ozone Layer", &use_ozone);
		ImGui::Checkbox("Do White Balance", &do_white_balance);
		ImGui::End();
		ImGui::Render();

		//End drawing imgui
		//-----------------------------------------------------------------

		if (0) {

			// Reserved for host side debugging

#if 0
//Copy debug buffer and print
			printf("ray_depth:%d\n", ray_depth);
			float3 *c = new float3[1000];
			memset(c, 0x0, sizeof(float3) * 1000);

			check_success(cudaMemcpy(c, debug_buffer, sizeof(float3) * 1000, cudaMemcpyDeviceToHost) == cudaSuccess);


			std::ofstream ray_pos("C:/Users/Admin/Desktop/PT_Plot/Ray_tr.txt", std::ios::out);
			for (int y = 0; y < 1000; y++) {

				if (c[y].x == .0f) continue;
				ray_pos << y + 1 << "	" << c[y].x << "\n";


			}

#endif

		}

		// Restart rendering if paused and started back 
		if (!render) {

			kernel_params.iteration = 0;

		}

		// Restart rendering if there is a change 
		if (ctx->change ||
			max_interaction != kernel_params.max_interactions ||
			ray_depth != kernel_params.ray_depth ||
			integrator != kernel_params.integrator) {

			kernel_params.integrator = integrator;
			//update_debug_buffer(&debug_buffer, kernel_params);
			//kernel_params.debug_buffer = debug_buffer;
			kernel_params.iteration = 0;
			ctx->change = false;

		}

		//if (kernel_params.iteration == kernel_params.max_interactions) ctx->save_image = true;

		// Test rotation
#if 0 
		float3 rotation = make_float3(0, 0, (M_PI / 10.0f) * rot_amount);
		mat4 rot = vdbs[0].get_xform().rotate_zyx(rotation);
		vdbs[0].set_xform(rot);
		check_success(cuMemcpyHtoD(d_volume_ptr, vdbs, sizeof(GPU_VDB) * 2) == cudaSuccess);
		rot_amount += 0.1f;
		kernel_params.iteration = 0;
#endif



		// Recreate environment sampling textures if sun position changes
		if (azimuth != kernel_params.azimuth || elevation != kernel_params.elevation) {
			create_cdf(kernel_params, &env_val_data, &env_func_data, &env_cdf_data, &env_marginal_func_data, &env_marginal_cdf_data);
			kernel_params.iteration = 0;
		}

		// Recompute sky if there is a change 

		if (temp_env_comp != env_comp || 
			earth_atmosphere.m_use_constant_solar_spectrum != use_constant_solar_spectrum || 
			earth_atmosphere.m_use_ozone != use_ozone) {

			earth_atmosphere.m_use_constant_solar_spectrum = use_constant_solar_spectrum;
			earth_atmosphere.m_use_ozone = use_ozone;

			switch (env_comp)
			{
			case 0:
				earth_atmosphere.m_use_luminance = NONE;
				break;
			case 1:
				earth_atmosphere.m_use_luminance = APPROXIMATE;
				break;
			case 2:
				earth_atmosphere.m_use_luminance = PRECOMPUTED;
				break;
			default:
				break;
			}
			earth_atmosphere.recompute();
			temp_env_comp = env_comp;
			kernel_params.iteration = 0;
		}
		if (earth_atmosphere.m_do_white_balance != do_white_balance || 
			earth_atmosphere.m_exposure != exposure) {
			earth_atmosphere.m_exposure = exposure;
			earth_atmosphere.m_do_white_balance = do_white_balance;
			earth_atmosphere.update_model();
			kernel_params.iteration = 0;
		}


		// Reallocate buffers if window size changed.
		int nwidth, nheight;
		glfwGetFramebufferSize(window, &nwidth, &nheight);
		if (nwidth != width || nheight != height)
		{
			width = nwidth;
			height = nheight;

			aspect = float(width) / float(height);
			cam.update_camera(lookfrom, lookat, vup, fov, aspect, aperture);
			resize_buffers(&accum_buffer, &raw_buffer, &display_buffer_cuda, width, height, display_buffer);
			kernel_params.accum_buffer = accum_buffer;
			kernel_params.raw_buffer = raw_buffer;
			glViewport(0, 0, width, height);

			kernel_params.resolution.x = width;
			kernel_params.resolution.y = height;
			kernel_params.iteration = 0;
		}

		// Restart render if camera moves 
		if (ctx->move_dx != 0.0 || ctx->move_dy != 0.0 || ctx->move_mx != 0.0 || ctx->move_my != 0.0 || ctx->zoom_delta) {

			update_camera(ctx->move_dx, ctx->move_dy, ctx->move_mx, ctx->move_my, ctx->zoom_delta, earth_atmosphere);
			ctx->move_dx = ctx->move_dy = ctx->move_mx = ctx->move_my = 0.0;
			ctx->zoom_delta = 0;
			kernel_params.iteration = 0;

		}

		if (ctx->save_image) {

			if (CreateDirectory("./render", NULL) || ERROR_ALREADY_EXISTS == GetLastError());

			char frame_string[100];
			sprintf_s(frame_string, "%d", frame);
			char file_name[100] = "./render/pathtrace.";
			strcat_s(file_name, frame_string);

#ifdef SAVE_TGA   

			int res = width * height;
			float4 *c = (float4*)malloc(res * sizeof(float4));
			check_success(cudaMemcpy(c, raw_buffer, sizeof(float4) * res, cudaMemcpyDeviceToHost) == cudaSuccess);

			bool success = save_tga(c, width, height, file_name);
			if (!success) printf("!Unable to save exr file.\n");

#endif

#ifdef SAVE_OPENEXR
			int res = width * height;
			float4 *c = (float4*)malloc(res * sizeof(float4));
			check_success(cudaMemcpy(c, raw_buffer, sizeof(float4) * res, cudaMemcpyDeviceToHost) == cudaSuccess);

			bool success = save_exr(c, width, height, file_name);
			if (!success) printf("!Unable to save exr file.\n");

#endif

			frame++;
			ctx->save_image = false;
		}

		// Map GL buffer for access with CUDA.
		check_success(cudaGraphicsMapResources(1, &display_buffer_cuda, /*stream=*/0) == cudaSuccess);
		void *p;
		size_t size_p;

		cudaGraphicsResourceGetMappedPointer(&p, &size_p, display_buffer_cuda);
		kernel_params.display_buffer = reinterpret_cast<unsigned int *>(p);

		// Launch volume rendering kernel.
		dim3 block(8, 8, 1);
		dim3 grid(int(width / block.x) + 1, int(height / block.y) + 1, 1);
		dim3 threads_per_block(16, 16);
		dim3 num_blocks((width + 15) / 16, (height + 15) / 16);

		void *params[] = { &cam, (void *)&l_list , (void *)&d_volume_ptr, &bvh_builder.bvh.BVHNodes, &bvh_builder.root ,(void *)atmos_params, &kernel_params };
		cuLaunchKernel(cuRaycastKernel, grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, params, NULL);
		++kernel_params.iteration;

		// Unmap GL buffer.
		check_success(cudaGraphicsUnmapResources(1, &display_buffer_cuda, /*stream=*/0) == cudaSuccess);

		//Do Image denoising with OIDN library 
		if (denoise) {

			int resolution = width * height;
			float4 *in_buffer;
			float3 *temp_in_buffer, *temp_out_buffer;

			in_buffer = (float4*)malloc(resolution * sizeof(float4));

			check_success(cudaMemcpy(in_buffer, raw_buffer, sizeof(float4) * resolution, cudaMemcpyDeviceToHost) == cudaSuccess);

			temp_in_buffer = (float3*)malloc(resolution * sizeof(float3));
			temp_out_buffer = (float3*)malloc(resolution * sizeof(float3));

			for (int i = 0; i < resolution; i++) {

				temp_in_buffer[i].x = in_buffer[i].x;
				temp_in_buffer[i].y = in_buffer[i].y;
				temp_in_buffer[i].z = in_buffer[i].z;
			}

			filter.setImage("color", temp_in_buffer, oidn::Format::Float3, width, height);
			filter.setImage("output", temp_out_buffer, oidn::Format::Float3, width, height);
			filter.set("hdr", true);
			filter.set("srgb", false);

			filter.commit();
			filter.execute();

			if (1) { // save denoised image 

				/*
				float3 red = make_float3(1.0f, .0f, .0f);
				float3 blue = make_float3(.0f, .0f, 1.0f);

				for (int i = 0; i < resolution; i++) {

					float diff = length(temp_in_buffer[i] - temp_out_buffer[i]);

					temp_out_buffer[i] = lerp(blue, red, diff);

				}
				*/

				if (CreateDirectory("./render", NULL) || ERROR_ALREADY_EXISTS == GetLastError());

				char frame_string[100];
				sprintf_s(frame_string, "%d", frame);
				char file_name[100] = "./render/pathtrace.";
				strcat_s(file_name, frame_string);

				bool success = save_exr(temp_out_buffer, width, height, file_name);
				if (!success) printf("!Unable to save exr file.\n");

				frame++;

			}

			denoise = false;
		}

		// Update texture for display.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
		glBindTexture(GL_TEXTURE_2D, display_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		check_success(glGetError() == GL_NO_ERROR);

		// Render the quad.

		glClear(GL_COLOR_BUFFER_BIT);
		glBindVertexArray(quad_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		check_success(glGetError() == GL_NO_ERROR);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		//break;
	}

	//Cleanup imgui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	// Cleanup CUDA.
	if (env_tex) {
		check_success(cudaDestroyTextureObject(kernel_params.env_tex) == cudaSuccess);
		check_success(cudaFreeArray(env_tex_data) == cudaSuccess);
	}
	check_success(cudaFree(accum_buffer) == cudaSuccess);
	check_success(cudaFree(raw_buffer) == cudaSuccess);
	check_success(cudaFree(debug_buffer) == cudaSuccess);

	// Cleanup OpenGL.
	glDeleteVertexArrays(1, &quad_vao);
	glDeleteBuffers(1, &quad_vertex_buffer);
	glDeleteProgram(program);
	check_success(glGetError() == GL_NO_ERROR);
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
