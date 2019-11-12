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
//	Version 1.0: Sergen Eren, 11/11/2019
//
// File: This is the header file for atmosphere class that implements 
//		 bruneton model sky in cuda. This file contains device side function 
//		 definitions and host side function declerations
//
//-----------------------------------------------


#ifndef  __ATMOSPHERE_H__
#define __ATMOSPHERE_H__

#include "texture_types.h"
#include "definitions.h"
#include <cuda.h>

enum atmosphere_error_t {

	ATMO_INIT_ERR,
	ATMO_INIT_FUNC_ERR,
	ATMO_RECOMPUTE_ERR,
	ATMO_FILL_TEX_ERR,
	ATMO_NO_ERR

};

class atmosphere {


public:
	
	atmosphere();
	~atmosphere();

	atmosphere_error_t init();
	atmosphere_error_t init_functions(CUmodule &cuda_module);
	atmosphere_error_t recompute(float azimuth, float elevation, float exposure);
	atmosphere_error_t fill_transmittance_texture();
	atmosphere_error_t fill_scattering_texture();
	atmosphere_error_t fill_irradiance_texture();

private:
	
	AtmosphereParameters atmosphere_parameters;

	CUfunction *transmittance_texture_function;
	CUfunction *scattering_texture_function;
	CUfunction *irradiance_texture_function;

	float3 *transmittance_buffer;
	float4 *scattering_buffer;
	float3 *irradiance_buffer;

	AtmosphereTextures atmosphere_textures;

};

#endif // ! __ATMOSPHERE_H__
