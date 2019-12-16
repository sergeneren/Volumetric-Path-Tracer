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
// File: This is the header file for fileIO operations.
//		 these functions load and save several types of files with cuda buffers.
//
//-----------------------------------------------


#ifndef _FILEIO_H_
#define _FILEIO_H_

#include <cuda.h>
#include "texture_types.h"
#include <vector>
#include <string>

// Saves a jpg with given float3 buffer 
bool save_texture_jpg(float3 *buffer, std::string filename, const int width, const int height);

// Saves a jpg with given float4 buffer ignores w component 
bool save_texture_jpg(float4 *buffer, std::string filename, const int width, const int height);

// Saves a png with given float3 buffer ignores alpha
bool save_texture_png(float3 *buffer, std::string filename, const int width, const int height);

// Saves a png with given float4 buffer
bool save_texture_png(float4 *buffer, std::string filename, const int width, const int height);

// Saves a tga with given float3 buffer ignores alpha
bool save_texture_tga(float3 *buffer, std::string filename, const int width, const int height);

// Saves a tga with given float4 buffer
bool save_texture_tga(float4 *buffer, std::string filename, const int width, const int height);

// Saves a exr with given float3 buffer
bool save_texture_exr(float3 *buffer, std::string filename, const int width, const int height, bool flip);

// Saves a exr with given float4 buffer
bool save_texture_exr(float4 *buffer, std::string filename, const int width, const int height, bool flip);

// Loads an exr to a float3 buffer
bool load_texture_exr(float3 **buffer, std::string filename, int &width, int &height, bool flip);

// Loads an exr to a float4 buffer
bool load_texture_exr(float4 **buffer, std::string filename, int &width, int &height, bool flip);

// Loads an exr to a float3 buffer and sends it to gpu
bool load_texture_exr_gpu(float3 **buffer, std::string filename, int &width, int &height, bool flip);

// Loads an exr to a float4 buffer and sends it to gpu
bool load_texture_exr_gpu(float4 **buffer, std::string filename, int &width, int &height, bool flip);

// Loads an bmp to a float3 buffer
bool load_texture_bmp(float3 **buffer, std::string filename, int &width, int &height, bool flip);

// Loads an bmp to a float3 buffer and sends it to gpu
bool load_texture_bmp_gpu(float3 **buffer, std::string filename, int &width, int &height, bool flip);

#endif // !_FILEIO_H_
