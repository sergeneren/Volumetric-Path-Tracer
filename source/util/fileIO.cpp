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

#include <OpenImageIO/imageio.h>
#include "OpenImageIO/imagebufalgo.h"
using namespace OIIO;

#include "bitmap_image.h"

#include "helper_math.h"
#include "helper_cuda.h"

#include "fileIO.h"
#include "logger.h"

bool save_texture_jpg(float3* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 3, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	// Flip
	ImageBuf A(filename);
	ImageBuf B;
	B = ImageBufAlgo::flip(A);
	B.write(filename);

	return true;
}

bool save_texture_jpg(float4* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	// Flip
	ImageBuf A(filename);
	ImageBuf B;
	B = ImageBufAlgo::flip(A);
	B.write(filename);

	return true;
}

bool save_texture_jpg(uint32_t* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::UINT32);
	out->open(filename, spec);
	out->write_image(TypeDesc::UINT8, buffer);
	out->close();

	return true;
}

bool save_texture_png(float3* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 3, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::UINT8, buffer);
	out->close();

	return true;
}

bool save_texture_png(float4* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_png(uint32_t* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::UINT32);
	out->open(filename, spec);
	out->write_image(TypeDesc::UINT8, buffer);
	out->close();

	return true;
}

bool save_texture_tga(float3* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 3, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_tga(float4* buffer, std::string filename, const int width, const int height)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::FLOAT);
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_exr(float3* buffer, std::string filename, const int width, const int height, bool flip)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 3, TypeDesc::FLOAT);
	spec.channelnames.push_back("R");
	spec.channelnames.push_back("G");
	spec.channelnames.push_back("B");
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_exr(float3* buffer, float* depth, std::string filename, const int width, const int height, bool flip)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::FLOAT);
	spec.channelnames.push_back("R");
	spec.channelnames.push_back("G");
	spec.channelnames.push_back("B");
	spec.channelnames.push_back("Z");
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_exr(float4* buffer, std::string filename, const int width, const int height, bool flip)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}

	ImageSpec spec(width, height, 4, TypeDesc::FLOAT);
	spec.channelnames.emplace_back("R");
	spec.channelnames.emplace_back("G");
	spec.channelnames.emplace_back("B");
	spec.channelnames.emplace_back("A");
	
	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, buffer);
	out->close();

	return true;
}

bool save_texture_exr(float4* buffer, float* depth, std::string filename, const int width, const int height, bool flip)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}
	ImageSpec spec(width, height, 5, TypeDesc::FLOAT);
	spec.channelnames.push_back("R");
	spec.channelnames.push_back("G");
	spec.channelnames.push_back("B");
	spec.channelnames.push_back("A");
	spec.channelnames.push_back("Z");

	std::vector<float> combined(width * height * 5);
	for (size_t x = 0; x < width * height; x++)
	{
		combined.push_back(buffer[x].x);
		combined.push_back(buffer[x].y);
		combined.push_back(buffer[x].z);
		combined.push_back(buffer[x].w);
		combined.push_back(depth[x]);
	}

	out->open(filename, spec);
	out->write_image(TypeDesc::FLOAT, combined.data());
	out->close();

	return true;
}

bool save_texture_exr(uint32_t* buffer, std::string filename, const int width, const int height, bool flip)
{
	std::unique_ptr<ImageOutput> out = ImageOutput::create(filename);
	if (!out) {
		return false;
	}
	ImageSpec spec(width, height, 4, TypeDesc::UINT32);
	spec.channelnames.push_back("R");
	spec.channelnames.push_back("G");
	spec.channelnames.push_back("B");
	spec.channelnames.push_back("A");
	out->open(filename, spec);
	out->write_image(TypeDesc::UINT32, buffer);
	out->close();

	return true;
}

bool load_texture_exr(float3** buffer, std::string filename, int& width, int& height, bool flip)
{
	auto in = ImageInput::open(filename);
	if (!in) {
		return false;
	}
	const ImageSpec& specInput = in->spec();
	width = specInput.width;
	height = specInput.height;
	int channels = specInput.nchannels;
	std::vector<float> pixels(height * width * channels);
	in->read_image(TypeDesc::FLOAT, &pixels[0]);
	in->close();

	*buffer = new float3[width * height];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;

			if (flip) idx = (width * height) - idx - 1;

			(*buffer)[idx].x = pixels[float_idx++]; // r
			(*buffer)[idx].y = pixels[float_idx++]; // g
			(*buffer)[idx].z = pixels[float_idx++]; // b
			float_idx++; // alpha
		}
	}

	return true;
}

bool load_texture_exr(float4** buffer, std::string filename, int& width, int& height, bool flip)
{
	auto in = ImageInput::open(filename);
	if (!in) {
		return false;
	}
	const ImageSpec& specInput = in->spec();
	width = specInput.width;
	height = specInput.height;
	int channels = specInput.nchannels;
	std::vector<float> pixels(height * width * channels);
	in->read_image(TypeDesc::FLOAT, &pixels[0]);
	in->close();

	*buffer = new float4[width * height];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;

			if (flip) idx = (width * height) - idx - 1;

			(*buffer)[idx].x = pixels[float_idx++]; // r
			(*buffer)[idx].y = pixels[float_idx++]; // g
			(*buffer)[idx].z = pixels[float_idx++]; // b
			(*buffer)[idx].w = pixels[float_idx++]; // b
		}
	}

	return true;
}

bool load_texture_exr_gpu(float3** buffer, std::string filename, int& width, int& height, bool flip)
{
	auto in = ImageInput::open(filename);
	if (!in) {
		return false;
	}
	const ImageSpec& specInput = in->spec();
	width = specInput.width;
	height = specInput.height;
	int channels = specInput.nchannels;
	std::vector<float> pixels(height * width * channels);
	in->read_image(TypeDesc::FLOAT, &pixels[0]);
	in->close();

	float3* values = new float3[height * width];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (flip) idx = (width * height) - idx - 1;
			values[idx].x = pixels[float_idx++]; // r
			values[idx].y = pixels[float_idx++]; // g
			values[idx].z = pixels[float_idx++]; // b
			float_idx++; // alpha
		}
	}

	checkCudaErrors(cudaMalloc(buffer, width * height * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(*buffer, values, width * height * sizeof(float3), cudaMemcpyHostToDevice));

	delete[] values;

	return true;
}

bool load_texture_exr_gpu(float4** buffer, std::string filename, int& width, int& height, bool flip)
{
	auto in = ImageInput::open(filename);
	if (!in) {
		return false;
	}
	const ImageSpec& specInput = in->spec();
	width = specInput.width;
	height = specInput.height;
	int channels = specInput.nchannels;
	std::vector<float> pixels(height * width * channels);
	in->read_image(TypeDesc::FLOAT, &pixels[0]);
	in->close();

	float4* values = new float4[height * width];

	int float_idx = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (flip) idx = (width * height) - idx - 1;
			values[idx].x = pixels[float_idx++]; // r
			values[idx].y = pixels[float_idx++]; // g
			values[idx].z = pixels[float_idx++]; // b
			values[idx].w = pixels[float_idx++]; // b
		}
	}

	checkCudaErrors(cudaMalloc(buffer, width * height * sizeof(float4)));
	checkCudaErrors(cudaMemcpy(*buffer, values, width * height * sizeof(float4), cudaMemcpyHostToDevice));

	delete[] values;

	return true;
}

bool load_texture_bmp(float3** buffer, std::string filename, int& width, int& height, bool flip)
{
	// Load blue noise texture from assets directory and send to gpu
	bitmap_image image(filename.c_str());

	if (!image) {
		log("Unable to load file " + filename, VPT_ERROR);
		return false;
	}
	width = image.width();
	height = image.height();

	*buffer = new float3[width * height];

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			rgb_t color;

			image.get_pixel(x, y, color);
			int idx = y * width + x;
			if (flip) idx = (width * height) - idx - 1;
			(*buffer)[idx].x = float(color.red) / 255.0f;
			(*buffer)[idx].y = float(color.blue) / 255.0f;
			(*buffer)[idx].z = float(color.green) / 255.0f;
		}
	}

	log("loaded file " + filename + " width:" + std::to_string(width) + " height:" + std::to_string(height), VPT_LOG);

	return true;
}

bool load_texture_bmp_gpu(float3** buffer, std::string filename, int& width, int& height, bool flip)
{
	// Load blue noise texture from assets directory and send to gpu
	bitmap_image image(filename.c_str());

	if (!image) {
		log("Unable to load file " + filename, VPT_ERROR);
		return false;
	}

	width = image.width();
	height = image.height();

	float3* values = new float3[height * width];

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			rgb_t color;

			image.get_pixel(x, y, color);
			int idx = y * width + x;
			if (flip) idx = (width * height) - idx - 1;
			values[idx].x = float(color.red) / 255.0f;
			values[idx].y = float(color.blue) / 255.0f;
			values[idx].z = float(color.green) / 255.0f;
		}
	}

	checkCudaErrors(cudaMalloc(buffer, width * height * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(*buffer, values, width * height * sizeof(float3), cudaMemcpyHostToDevice));

	delete[] values;

	log("loaded file " + filename + " width:" + std::to_string(width) + " height:" + std::to_string(height), VPT_LOG);

	return true;
}