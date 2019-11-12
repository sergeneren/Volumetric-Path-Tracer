
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
//	Version 1.0: Sergen Eren, 12/11/2019
//
// File: This is the implementation file for atmosphere class functions 
//
//-----------------------------------------------


#include <vector>
#include <string>

#include "atmosphere.h"



atmosphere_error_t atmosphere::init_functions(CUmodule &cuda_module) {

	CUresult error;
	error = cuModuleGetFunction(transmittance_texture_function, cuda_module, "fill_transmittance_texture");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;
	
	error = cuModuleGetFunction(scattering_texture_function, cuda_module, "fill_scattering_texture");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;

	error = cuModuleGetFunction(irradiance_texture_function, cuda_module, "fill_irradiance_texture");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;
	
	return ATMO_NO_ERR;

}

atmosphere::~atmosphere() {

	transmittance_texture_function = nullptr;
	scattering_texture_function = nullptr;
	irradiance_texture_function = nullptr;

}

atmosphere::atmosphere() {

	transmittance_texture_function = new CUfunction;
	scattering_texture_function = new CUfunction;
	irradiance_texture_function = new CUfunction;
}