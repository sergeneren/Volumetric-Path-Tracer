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
#include "texture_indirect_functions.h"
#include "texture_fetch_functions.h"
#include "matrix_math.h"
#include <string>

#include "helper_math.h"
#include "constants.h"
#include "definitions.h"



class atmosphere {



public:

	__device__ __host__ atmosphere(){}
	__device__ __host__ ~atmosphere(){}

	__device__ float ClampCosine(float mu);

	__device__ float ClampDistance(float d);
	
	__device__ float ClampRadius(const AtmosphereParameters atmosphere, float r);
	
	__device__ float SafeSqrt(float a);
	
	__device__ float DistanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu);
	
	__device__ float DistanceToBottomAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu);
	
	__device__ bool	  RayIntersectsGround(const AtmosphereParameters atmosphere, float r, float mu);
	
	__device__ float GetLayerDensity(const DensityProfileLayer layer, float altitude);

	__device__ float GetProfileDensity(const DensityProfile profile, float altitude);

	__device__ float ComputeOpticalLengthToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, const DensityProfile profile, float r, float mu);

	__device__ float3 ComputeTransmittanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu);

	__device__ float GetTextureCoordFromUnitRange(float x, int texture_size);

	__device__ float GetUnitRangeFromTextureCoord(float u, int texture_size);

	__device__ float2 GetTransmittanceTextureUvFromRMu(const AtmosphereParameters atmosphere, float r, float mu);

	__device__ void GetRMuFromTransmittanceTextureUv(const AtmosphereParameters atmosphere, float2 uv, float & r, float & mu);

	__device__ float3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(const AtmosphereParameters atmosphere, float2 frag_coord);

	__device__ float3 GetTransmittanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu);

	__device__ float3 GetTransmittance(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu, float d, bool ray_r_mu_intersects_ground);

	__device__ float3 GetTransmittanceToSun(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu_s);

	__device__ void ComputeSingleScatteringIntegrand(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu, float mu_s, float nu, float d, bool ray_r_mu_intersects_ground, float3 & rayleigh, float3 & mie);

	__device__ float DistanceToNearestAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu, bool ray_r_mu_intersects_ground);

	__device__ void ComputeSingleScattering(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, float3 & rayleigh, float3 & mie);

	__device__ float RayleighPhaseFunction(float nu);

	__device__ float MiePhaseFunction(float g, float nu);

	__device__ float4 GetScatteringTextureUvwzFromRMuMuSNu(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground);

	__device__ void GetRMuMuSNuFromScatteringTextureUvwz(const AtmosphereParameters atmosphere, float4 uvwz, float & r, float & mu, float & mu_s, float & nu, bool & ray_r_mu_intersects_ground);

	__device__ void GetRMuMuSNuFromScatteringTextureFragCoord(const AtmosphereParameters atmosphere, float3 frag_coord, float & r, float & mu, float & mu_s, float & nu, bool & ray_r_mu_intersects_ground);

	__device__ void ComputeSingleScatteringTexture(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float3 frag_coord, float3 & rayleigh, float3 & mie);

	__device__ float3 GetScattering(const AtmosphereParameters atmosphere, const cudaTextureObject_t scattering_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground);

	__device__ float3 GetScattering(const AtmosphereParameters atmosphere, const cudaTextureObject_t single_rayleigh_scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, const cudaTextureObject_t multiple_scattering_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, int scattering_order);

	__device__ float3 GetIrradiance(const AtmosphereParameters atmosphere, const cudaTextureObject_t irradiance_texture, float r, float mu_s);

	__device__ float3 GetCombinedScattering(const AtmosphereParameters atmosphere, const cudaTextureObject_t scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, float3 & single_mie_scattering);

	__device__ float3 GetSkyRadiance(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, float3 camera, float3 view_ray, float shadow_length, float3 sun_direction, float3 & transmittance);

	__device__ float3 ComputeScatteringDensity(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t single_rayleigh_scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, const cudaTextureObject_t multiple_scattering_texture, const cudaTextureObject_t irradiance_texture, float r, float mu, float mu_s, float nu, int scattering_order);

	__device__ float3 ComputeMultipleScattering(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t scattering_density_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground);

	__device__ float3 ComputeScatteringDensityTexture(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t single_rayleigh_scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, const cudaTextureObject_t multiple_scattering_texture, const cudaTextureObject_t irradiance_texture, float3 frag_coord, int scattering_order);

	__device__ float3 ComputeMultipleScatteringTexture(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t scattering_density_texture, float3 frag_coord, float & nu);

	__device__ float3 ComputeDirectIrradiance(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float r, float mu_s);

	__device__ float3 ComputeIndirectIrradiance(const AtmosphereParameters atmosphere, const cudaTextureObject_t single_rayleigh_scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, const cudaTextureObject_t multiple_scattering_texture, float r, float mu_s, int scattering_order);

	__device__ float2 GetIrradianceTextureUvFromRMuS(const AtmosphereParameters atmosphere, float r, float mu_s);

	__device__ void GetRMuSFromIrradianceTextureUv(const AtmosphereParameters atmosphere, float2 uv, float & r, float & mu_s);

	__device__ float3 ComputeDirectIrradianceTexture(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, float2 frag_coord);

	__device__ float3 ComputeIndirectIrradianceTexture(const AtmosphereParameters atmosphere, const cudaTextureObject_t single_rayleigh_scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, const cudaTextureObject_t multiple_scattering_texture, float2 frag_coord, int scattering_order);

	__device__ float3 GetSkyRadianceToPoint(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t scattering_texture, const cudaTextureObject_t single_mie_scattering_texture, float3 camera, float3 point, float shadow_length, float3 sun_direction, float3 & transmittance);

	__device__ float3 atmosphere::GetSunAndSkyIrradiance(const AtmosphereParameters atmosphere, const cudaTextureObject_t transmittance_texture, const cudaTextureObject_t irradiance_texture, float3 point, float3 normal, float3 sun_direction, float3 &sky_irradiance);








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

#endif // ! __ATMOSPHERE_H__
