//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Sergen Eren, 26/3/2019
//----------------------------------------------------------------------------------
// 
// File: Custom path trace kernel: 
//       Performs custom path tracing
//
//-----------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdio.h>
#include "cuda_math.cuh"
#include <float.h>
#include <cuda_runtime.h> 
#include <curand_kernel.h>


typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 

#include "render_kernel.h"



#define BLACK			make_float3(0.0f, 0.0f, 0.0f)
#define WHITE			make_float3(1.0f, 1.0f, 1.0f)
#define RED				make_float3(1.0f, 0.0f, 0.0f)
#define GREEN			make_float3(0.0f, 1.0f, 0.0f)
#define BLUE			make_float3(0.0f, 0.0f, 1.0f)
#define EPS				0.001f

#include <curand_kernel.h>
typedef curandStatePhilox4_32_10_t Rand_state;
#define rand(state) curand_uniform(state)

// Helper functions

__device__ inline void coordinate_system(float3 v1, float3 &v2, float3 &v3) {

	if (fabsf(v1.x) > fabsf(v1.y)) v2 = make_float3(-v1.z , 0 , v1.x) / sqrtf(v1.x * v1.x + v1.z +v1.z);
	else v2 = make_float3(0, v1.z, -v1.y) / sqrtf(v1.y * v1.y + v1.z + v1.z);

	v3 = cross(v1, v2);

}

__device__ inline float3 spherical_direction(
						float sinTheta, 
						float cosTheta, 
						float phi, 
						float3 x, 
						float3 y,
						float3 z)
{

	return (x * sinTheta * cosf(phi)) + (y * sinTheta * sinf(phi)) + (z * cosTheta);

}

__device__ inline bool solveQuadratic(float a, float b, float c, float& x1, float& x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return false;
		x1 = 0; x2 = sqrt(-c / a);
		return true;
	}

	float discr = b * b - 4 * a * c;

	if (discr < 0) return false;

	float q = (b < 0.f) ? -0.5f * (b - sqrt(discr)) : -0.5f * (b + sqrt(discr));
	x1 = q / a;
	x2 = c / q;

	return true;
}

__device__ bool raySphereIntersect(const float3& orig, const float3& dir, const float& radius, float& t0, float& t1) {
	
	float A = squared_length(dir);
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

__device__ inline float degree_to_radians(float degree) {

	return degree * M_PI / 180.0f;

}


__device__ inline float3 degree_to_cartesian(float azimuth, float elevation) {


	azimuth =degree_to_radians(clamp(azimuth, 0.0f, 360.0f));
	elevation =degree_to_radians(clamp(elevation, .0f, 180.0f));
		
	float x = sinf(elevation) * cosf(azimuth);
	float y = sinf(elevation) * sinf(azimuth);
	float z = cosf(elevation);

	return make_float3(x, y, z);
}

//Phase functions pdf 

__device__ inline float isotropic() {

	return M_PI_4;

}

__device__ inline float henyey_greenstein(float cos_theta, float g) {

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}

__device__ inline float double_henyey_greenstein(float cos_theta, float f, float g1, float g2) {

	return (1 - f)*henyey_greenstein(cos_theta, g2) + f * henyey_greenstein(cos_theta, g1);

}

__device__ inline float schlick(float cos_theta, float k) { // simpler hg phase function Note: -1<k<1   

	float denominator = 1 + k * cos_theta;

	return M_PI_4 * (1 - k * k) / (denominator*denominator);

}

__device__ inline float rayleigh(float cos_sq_theta, float lambda) // rayleigh scattering
{

	return 3 * (1 + cos_sq_theta) / 4 * lambda*lambda*lambda*lambda; // 

}

__device__ inline float cornette_shanks(float cos_theta, float cos_sq_theta, float g) {

	float first_part = (1 - g * g) / (2 + g * g);
	float second_part = (1 + cos_sq_theta) / pow((1 + g * g - cos_theta), 1.5f);

	return M_PI_4 * 1.5f * first_part * second_part;

}
// End phase functions pdf

//Phase function direction samplers

__device__ inline float sample_hg(float3 &wi, Rand_state &randstate, float &cosTheta ,float g) {

	float cos_theta;
	
	if (fabsf(g) < 0.001f) cos_theta = 1 - 2 * rand(&randstate);
	else {

		float sqr_term = (1 - g * g) / (1 - g + 2 * g * rand(&randstate));
		cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g);
	}
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
	float phi = (float)(2.0 * M_PI) * rand(&randstate);
	float3 v1, v2;
	coordinate_system(wi, v1, v2);
	spherical_direction(sin_theta, cos_theta, phi, v1, v2, wi);
	wi = make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);

	return henyey_greenstein(cos_theta, g);
}

__device__ inline float sample_double_hg(float3 &wi, Rand_state randstate, float f, float g1, float g2) {

	float3 v1, v2; 
	float cos_theta1, cos_theta2;
	sample_hg(v1, randstate, cos_theta1, g1);
	sample_hg(v2, randstate, cos_theta2, g2);

	wi = lerp(v1, v2, 1-f);
	float cos_theta = lerp(cos_theta1, cos_theta2, 1-f);
	return double_henyey_greenstein(cos_theta, f, g1, g2);
}


// End phase function samplers

__device__ inline bool in_volume_bbox(
		const VDBInfo gvdb, 
		const float3 pos) 
{

	return pos.x >= gvdb.bmin.x && pos.y >= gvdb.bmin.y && pos.z >= gvdb.bmin.z && pos.x < gvdb.bmax.x && pos.y < gvdb.bmax.y && pos.z < gvdb.bmax.z;
}

__device__ inline float get_extinction(
	const Kernel_params &kernel_params,
	VDBInfo *gvdb,
	const float3 &p)
{

	float density = 0.0f; 

	//brick node variables 
	float3 vmin; //root pos of brick node
	uint64 nodeid; // brick id 
	float3 offset; // brick offset
	float3 vdel; // i.e. voxel size 

	VDBNode* brick_node = getNodeAtPoint(gvdb, p, &offset, &vmin, &vdel, &nodeid);

	if (brick_node != 0x0) {

		float3 brick_pos = (p - vmin) / vdel;
		float3 atlas_pos = make_float3(brick_node->mValue);
		density = tex3D<float>(gvdb->volIn[0], brick_pos.x + atlas_pos.x, brick_pos.y + atlas_pos.y, brick_pos.z + atlas_pos.z) * kernel_params.max_extinction;
	}

	return density;
}

__device__ inline bool integrate(
			Rand_state &rand_state,
			float3 &ray_pos,
			const float3 &ray_dir,
			const Kernel_params &kernel_params,
			VDBInfo &gvdb) 
{

	float t = 0.0f; 
	float3 pos; 


	do {
		t -= logf(1.0f - rand(&rand_state)) / kernel_params.max_extinction;
		pos = ray_pos + ray_dir * t; 
		if (!in_volume_bbox(gvdb, pos)) return false;
	} while (get_extinction(kernel_params, &gvdb, pos) < rand(&rand_state) * kernel_params.max_extinction);

	ray_pos = pos;
	return true; 


}


// Light Samplers 

__device__ inline float3 sample_atmosphere(const Kernel_params &kernel_params ,const float3 orig, const float3 dir, float tmin, float tmax) {

	// initial parameters
	float	atmosphereRadius = 6420e3f;
	float3	sunDirection = degree_to_cartesian(kernel_params.light_pos.x, kernel_params.light_pos.y);
	float	earthRadius = 6360e3f;
	float	Hr = 7994.0f;
	float	Hm = 1200.0f;
	float3	betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f) ;
	float3	betaM = make_float3(21e-6f);
	//


	float t0, t1;

	if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0) return make_float3(1.0f, .0f, .0f);
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
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float g = 0.76f;

	float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

	for (uint i = 0; i < numSamples; ++i) {
		float3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
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


	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * 20 ;
}


// Volume functions 

__device__ inline float3 transmittance(Rand_state &rand_state, float3 pos , float3 dir ,  const Kernel_params &kernel_params, VDBInfo &gvdb) {

	float3 p = pos; 
	float t = 0.0f; 
	bool terminated = false; 


	do {

		float zeta = rand(&rand_state);
		t -= logf(1.0f - zeta) / kernel_params.max_extinction;
		p += dir * t;
		if (!in_volume_bbox(gvdb, p)) break;

		float density = get_extinction(kernel_params, &gvdb, p);

		float xi = rand(&rand_state);
		if ( xi < density / kernel_params.max_extinction) terminated = true;

	} while ( !terminated);

	if(terminated) return make_float3(0.0f);
	else return kernel_params.extinction;
}

__device__ inline float3 sample(Rand_state &rand_state, float3 pos, float3 dir, const Kernel_params &kernel_params, VDBInfo &gvdb) {

	float3 p = pos;
	float t = 0.0f;
	bool terminated = false;


	do {

		float zeta = rand(&rand_state);
		t -= logf(1.0f - zeta) / kernel_params.max_extinction;
		p += dir * t;
		if (!in_volume_bbox(gvdb, p)) break;

		float density = get_extinction(kernel_params, &gvdb, p);

		float xi = rand(&rand_state);
		if (xi < density / kernel_params.max_extinction) terminated = true;

	} while (!terminated);

	if (terminated) return make_float3(0.0f);
	else return kernel_params.extinction;
}


__device__ inline float3 trace_volume(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params kernel_params,
	VDBInfo gvdb)
{
	
	float3 t = rayBoxIntersect(ray_pos, ray_dir, gvdb.bmin, gvdb.bmax);
	float3 w = make_float3(1.0f);
	float3 Tr = make_float3(1.0f); 
	float3 Sun_light = make_float3(0.0f, 0.0f, 0.0f);

	float phase_pdf = 1.0f;

	if (t.z != NOHIT) {
		
		ray_pos +=  ray_dir * t.x;
		uint num_interactions = 0; 
		
		while (integrate(rand_state, ray_pos, ray_dir, kernel_params, gvdb)) {

			// Is the path length exeeded?
			if (num_interactions++ >= kernel_params.max_interactions)
				return make_float3(0.0f, 0.0f, 0.0f);

			//Tr *= transmittance(rand_state , ray_pos, kernel_params, gvdb);

			w *= (float3)kernel_params.albedo;
			
			float cos_theta;
			phase_pdf = sample_hg(ray_dir, rand_state, cos_theta, kernel_params.phase_g1);
			//phase_pdf = sample_double_hg(ray_dir, rand_state, kernel_params.phase_f, kernel_params.phase_g1, kernel_params.phase_g2 );

		}

	}
	//Sun_light = kernel_params.light_energy * Tr;

	// Lookup environment.

	if (kernel_params.environment_type == 0) {
		float t0, t1, tmax = FLT_MAX;
		float3 pos = ray_pos;
		pos.y += 1000 + 6360e3f;

		if (raySphereIntersect(pos, ray_dir, 6360e3f, t0, t1) && t1 > .0f) tmax = fmaxf(.0f, t0);
		float3 L = sample_atmosphere(kernel_params, pos, ray_dir, 0, tmax);

		return L * w / phase_pdf;
	}
	else {
		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
		return make_float3(texval.x , texval.y , texval.z )* w / phase_pdf + Sun_light;
	}
	*/
}

extern "C" __global__ void volume_rt_kernel(VDBInfo gvdb, const Kernel_params kernel_params) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= kernel_params.resolution.x || y >= kernel_params.resolution.y)
		return;

	// Initialize pseudorandom number generator (PRNG); assume we need no more than 4096 random numbers.
	const unsigned int idx = y * scn.width + x;
	Rand_state rand_state;
	curand_init(idx, 0, kernel_params.iteration * 4096, &rand_state);
	
	float3 ray_dir =  normalize(getViewRay((float(scn.width - x) + 0.5) / scn.width, (float(scn.height - y) + 0.5) / scn.height));
	
	float3 value = trace_volume(rand_state, scn.campos, ray_dir, kernel_params, gvdb);

	// Accumulate.
	if (kernel_params.iteration == 0)
		kernel_params.accum_buffer[idx] = value;
	else
		kernel_params.accum_buffer[idx] = kernel_params.accum_buffer[idx] +
		(value - kernel_params.accum_buffer[idx]) / (float)(kernel_params.iteration + 1);

	// Update display buffer (simple Reinhard tonemapper + gamma).

	float3 val = kernel_params.accum_buffer[idx] * kernel_params.exposure_scale;

	val.x *= (1.0f + val.x * 0.1f) / (1.0f + val.x);
	val.y *= (1.0f + val.y * 0.1f) / (1.0f + val.y);
	val.z *= (1.0f + val.z * 0.1f) / (1.0f + val.z);
	const unsigned int r = (unsigned int)(255.0f * fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int g = (unsigned int)(255.0f * fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int b = (unsigned int)(255.0f * fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	kernel_params.display_buffer[idx] = 0xff000000 | (r << 16) | (g << 8) | b;

}