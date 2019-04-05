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


//Phase functions 

__device__ inline float isotropic() {

	return M_PI_4;

}

__device__ inline float henyey_greenstein(float cos_theta, float g) {

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}

__device__ inline float double_henyey_greenstein(float cos_theta, float f, float g1, float g2) {

	return (1 - f)*henyey_greenstein(cos_theta, g1) + f * henyey_greenstein(cos_theta, g2);

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
// End phase functions



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

__device__ inline float transmittance(Rand_state &rand_state, float3 pos , const Kernel_params &kernel_params, VDBInfo &gvdb) {

	float3 p = pos; 
	float t = 0.0f; 
	float3 Lpos = make_float3(1000.0f, 0.0f, 0.0f);
	float3 L_dir = normalize(Lpos - pos);
	bool terminated = false; 


	do {

		float zeta = rand(&rand_state);
		t -= logf(1.0f - zeta) / kernel_params.max_extinction;
		p = p + L_dir * t;
		if (!in_volume_bbox(gvdb, p)) break;

		float density = get_extinction(kernel_params, &gvdb, p) / kernel_params.max_extinction;

		float xi = rand(&rand_state);
		if (density < xi * kernel_params.max_extinction) terminated = true;

	} while ( !terminated);

	if(terminated) return 0.0f;
	else return 1.0f;
}

__device__ inline float3 trace_volume(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params kernel_params,
	VDBInfo gvdb)
{
	
	float3 t = rayBoxIntersect(ray_pos, ray_dir, gvdb.bmin, gvdb.bmax);
	float w = 1.0f;
	float Tr = 1.0f; 
	float3 Sun_light = make_float3(0.0f, 0.0f, 0.0f);

	if (t.z != NOHIT) {
		
		ray_pos +=  ray_dir * t.x;
		uint num_interactions = 0; 
		
		while (integrate(rand_state, ray_pos, ray_dir, kernel_params, gvdb)) {

			// Is the path length exeeded?
			if (num_interactions++ >= kernel_params.max_interactions)
				return make_float3(0.0f, 0.0f, 0.0f);

			Tr *= transmittance(rand_state , ray_pos, kernel_params, gvdb);

			w *= kernel_params.albedo * (1.0f - Tr);
			
			// Sample isotropic phase function.
			const float phi = (float)(2.0 * M_PI) * rand(&rand_state);
			const float cos_theta = 1.0f - 2.0f * rand(&rand_state);
			const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
			ray_dir = make_float3(
				cosf(phi) * sin_theta,
				sinf(phi) * sin_theta,
				cos_theta);

			Sun_light = make_float3(1.0f,1.0f, 1.0f) * w ;

		}

	}


	// Lookup environment.
	if (kernel_params.environment_type == 0) {
		const float f = (0.5f + 0.5f * ray_dir.y) * w;
		return make_float3(f, f, f) + Sun_light;
	}
	else {
		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
		return make_float3(texval.x * w, texval.y * w, texval.z * w) + Sun_light;
	}
	
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
	const unsigned int r = (unsigned int)(255.0f *
		fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int g = (unsigned int)(255.0f *
		fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int b = (unsigned int)(255.0f *
		fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	kernel_params.display_buffer[idx] = 0xff000000 | (r << 16) | (g << 8) | b;

}