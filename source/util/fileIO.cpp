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
//	Version 1.0: Sergen Eren, 15/12/2019
//
// File: This is the implementation file for fileIO
//
//-----------------------------------------------

#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "helper_math.h"
#include "helper_cuda.h"

#include "fileIO.h"

bool save_texture_jpg(float3 * buffer, std::string filename, const int width, const int height)
{
	unsigned char *data = new unsigned char[width*height * 3];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].z * 255)), 255);

		}
	}
	stbi_flip_vertically_on_write(1);
	
	int res = stbi_write_jpg(filename.c_str(), width, height, 3, (void*)data, 100);
	delete[] data;

	if (res) return true;
	return false;
}

bool save_texture_jpg(float4 * buffer, std::string filename, const int width, const int height)
{

	unsigned char *data = new unsigned char[width*height * 3];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].z * 255)), 255);

		}
	}
	stbi_flip_vertically_on_write(1);

	int res = stbi_write_jpg(filename.c_str(), width, height, 3, (void*)data, 100);
	delete[] data;

	if (res) return true;
	return false;
}

bool save_texture_png(float3 * buffer, std::string filename, const int width, const int height)
{
	unsigned char *data = new unsigned char[width*height * 3];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].z * 255)), 255);

		}
	}
	stbi_flip_vertically_on_write(1);

	int res = stbi_write_png(filename.c_str(), width, height, 3, (void*)data, 0);
	delete[] data;

	if (res) return true;
	return false;

}

bool save_texture_png(float4 * buffer, std::string filename, const int width, const int height)
{
	unsigned char *data = new unsigned char[width*height * 4];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].z * 255)), 255);
			data[idx++] = fminf(fmaxf(0, unsigned char(buffer[index].w * 255)), 255);
		}
	}
	stbi_flip_vertically_on_write(1);
	int res = stbi_write_png(filename.c_str(), width, height, 4, (void*)data, 0);
	delete[] data;

	if (res) return true;
	return false;
}

bool save_texture_tga(float3 * buffer, std::string filename, const int width, const int height)
{

	unsigned char *data = new unsigned char[width*height * 3];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].x, 0.0f), 1.0f));
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].y, 0.0f), 1.0f));
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].z, 0.0f), 1.0f));

		}
	}
	stbi_flip_vertically_on_write(1);

	int res = stbi_write_tga(filename.c_str(), width, height, 3, data);
	delete[] data;

	if (res) return true;
	return false;

}

bool save_texture_tga(float4 * buffer, std::string filename, const int width, const int height)
{
	unsigned char *data = new unsigned char[width*height * 4];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].x, 0.0f), 1.0f));
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].y, 0.0f), 1.0f));
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].z, 0.0f), 1.0f));
			data[idx++] = (unsigned int)(255.0f * fminf(fmaxf(buffer[i].w, 0.0f), 1.0f));

		}
	}
	stbi_flip_vertically_on_write(1);

	int res = stbi_write_tga(filename.c_str(), width, height, 4, data);
	delete[] data;

	if (res) return true;
	return false;
}

bool save_texture_exr(float3 *buffer, std::string filename, const int width, const int height, bool flip)
{
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
			if (flip) i = height - i - 1;
			int idx2 = i * width + j;

			images[0][idx] = buffer[idx2].x;
			images[1][idx] = buffer[idx2].y;
			images[2][idx] = buffer[idx2].z;
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

bool save_texture_exr(float4 *buffer, std::string filename, const int width, const int height, bool flip)
{
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
			if(flip) i = height - i - 1;
			int idx2 = i * width + j;

			images[0][idx] = buffer[idx2].x;
			images[1][idx] = buffer[idx2].y;
			images[2][idx] = buffer[idx2].z;
			images[3][idx] = buffer[idx2].w;

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

bool load_texture_exr(float3 **buffer, std::string filename, int &width, int &height, bool flip)
{

	float *rgba;
	const char *err;
	
	int ret = LoadEXR(&rgba, &width, &height, filename.c_str(), &err);
	printf("loaded file %s, width: %i, height: %i \n", filename.c_str(), width, height);

	if (ret != 0) {
		printf("err: %s\n", err);
		return false;
	}

	*buffer = new float3[width*height];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;

			if (flip) idx = (width*height) - idx - 1;

			(*buffer)[idx].x = rgba[float_idx++]; // r
			(*buffer)[idx].y = rgba[float_idx++]; // g
			(*buffer)[idx].z = rgba[float_idx++]; // b
			float_idx++; // alpha
		}
	}

	delete[] rgba;

	return true;
}

bool load_texture_exr(float4 **buffer, std::string filename, int &width, int &height, bool flip)
{
	float *rgba;
	const char *err;

	int ret = LoadEXR(&rgba, &width, &height, filename.c_str(), &err);
	printf("loaded file %s, width: %i, height: %i \n", filename.c_str(), width, height);

	if (ret != 0) {
		printf("err: %s\n", err);
		return false;
	}

	*buffer = new float4[width*height];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;

			if (flip) idx = (width*height) - idx - 1;

			(*buffer)[idx].x = rgba[float_idx++]; // r
			(*buffer)[idx].y = rgba[float_idx++]; // g
			(*buffer)[idx].z = rgba[float_idx++]; // b
			(*buffer)[idx].w = rgba[float_idx++]; // b
		}
	}

	delete[] rgba;

	return true;
}

bool load_texture_exr_gpu(float3 ** buffer, std::string filename, int & width, int & height, bool flip)
{
	float *rgba;
	const char *err;
	int ret = LoadEXR(&rgba, &width, &height, filename.c_str(), &err);
	printf("loaded file %s, width: %i, height: %i \n", filename.c_str(), width, height);
	if (ret != 0) {
		printf("err: %s\n", err);
		return false;
	}

	float3 *values = new float3[height * width];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (flip) idx = (width*height) - idx - 1;
			values[idx].x = rgba[float_idx++]; // r
			values[idx].y = rgba[float_idx++]; // g
			values[idx].z = rgba[float_idx++]; // b
			float_idx++; // alpha
		}
	}

	checkCudaErrors(cudaMalloc(buffer, width * height * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(*buffer, values, width * height * sizeof(float3), cudaMemcpyHostToDevice));

	delete[] rgba;
	delete[] values;

	return true;
}

bool load_texture_exr_gpu(float4 ** buffer, std::string filename, int & width, int & height, bool flip)
{
	float *rgba;
	const char *err;
	int ret = LoadEXR(&rgba, &width, &height, filename.c_str(), &err);
	printf("loaded file %s, width: %i, height: %i \n", filename.c_str(), width, height);
	if (ret != 0) {
		printf("err: %s\n", err);
		return false;
	}

	float4 *values = new float4[height * width];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (flip) idx = (width*height) - idx - 1;
			values[idx].x = rgba[float_idx++]; // r
			values[idx].y = rgba[float_idx++]; // g
			values[idx].z = rgba[float_idx++]; // b
			values[idx].w = rgba[float_idx++]; // b
		}
	}

	checkCudaErrors(cudaMalloc(buffer, width * height * sizeof(float4)));
	checkCudaErrors(cudaMemcpy(*buffer, values, width * height * sizeof(float4), cudaMemcpyHostToDevice));

	delete[] rgba;
	delete[] values;

	return true;
}
