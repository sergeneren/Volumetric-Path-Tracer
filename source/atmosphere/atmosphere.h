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
#include "matrix_math.h"
#include <string>

#include "helper_math.h"
#include "constants.h"


#define ALIGN(x)	__align__(x)


struct ALIGN(16) DensityProfileLayer {

	__device__ __host__ DensityProfileLayer() : DensityProfileLayer(.0f, .0f, .0f, .0f, .0f) {}
	__device__ __host__ DensityProfileLayer(float width, float exp_term, float exp_scale, 
											float linear_term, float const_term)
		: width(width), exp_term(exp_term), exp_scale(exp_scale), linear_term(linear_term), const_term(const_term){}

	float width;
	float exp_term;
	float exp_scale;
	float linear_term;
	float const_term;
};

struct ALIGN(16) DensityProfile {

	DensityProfileLayer layers[2];

};


struct ALIGN(16) AtmosphereParameters {

	float3 solar_irradiance;
	float angle;
	float bottom_radius;
	float top_radius;
	
	DensityProfile rayleigh_density;
	float3 rayleigh_scattering;

	DensityProfile mie_density;
	float3 mie_scattering;
	float3 mie_extinction;
	float mie_phase_function_g; 

	DensityProfile absorption_density;
	float3 absorption_extinction; 

	float3 ground_albedo;

	float mu_s_min;

};





class atmosphere {



public:

	__device__ __host__ atmosphere(){}
	__device__ __host__ ~atmosphere(){}

	__device__ float3 getIrradiance(
		const AtmosphereParameters atmosphere,
		const cudaTextureObject_t irradiance_texture, 
		float r, float mu_s);
	
	__device__ float getTransmittanceToSun(
		const AtmosphereParameters atmosphere,
		const cudaTextureObject_t transmittance_texture, 
		float r, float mu_s);

	__device__ float3 getSunAndSkyIrradiance(
		const AtmosphereParameters atmosphere, 
		const cudaTextureObject_t transmittance_texture,
		const cudaTextureObject_t irradiance_texture,
		float3 position, float3 direction, float3 sun_direction,
		float3 &sky_irradiance
	);



	__host__ bool init();
	__host__ bool precompute();





	// Variables that can be modified in main 

	float sun_zenith_angle; // in radians 
	float sun_azimuth_angle; // in radians
	float sun_angular_radius;



private:
	
	uint num_precomputed_wavelengths;

	cudaTextureObject_t transmittance_texture;
	cudaTextureObject_t scattering_texture;
	cudaTextureObject_t optional_mie_scattering_texture;
	cudaTextureObject_t irradiance_texture;

};



__device__ float3 atmosphere::getIrradiance(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t irradiance_texture,
	float r, float mu_s) {


	float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);

	const float3 texval = tex2D<float3>(irradiance_texture, uv.x, uv.y);

	return texval;
}


__device__ float getTransmittanceToSun(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	float r, float mu_s) {

	float sin_theta_h = atmosphere.bottom_radius / r;
	float cos_theta_h = -sqrtf(max(1.0 - sin_theta_h * sin_theta_h, 0.0));

	return GetTransmittanceToTopAtmosphereBoundary(	atmosphere, transmittance_texture, r, mu_s) *
		smoothstep(-sin_theta_h * sun_angular_radius / rad,
			sin_theta_h * sun_angular_radius / rad,
			mu_s - cos_theta_h);

}



__device__ float3 atmosphere::getSunAndSkyIrradiance(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	const cudaTextureObject_t irradiance_texture,
	const float3 position, const float3 normal, const float3 sun_direction,
	float3 &sky_irradiance) {

	float r = length(position);
	float mu_s = dot(position, sun_direction) / r; 

	sky_irradiance = getIrradiance(atmosphere, irradiance_texture, r, mu_s) * (1.0f + dot(normal, position) / r) * 0.5f;

	return atmosphere.solar_irradiance * getTransmittanceToSun(atmosphere, transmittance_texture, r, mu_s) * fmaxf(dot(normal, sun_direction), .0f);

}























#endif // ! __ATMOSPHERE_H__
