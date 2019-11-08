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
//	Version 1.0: Sergen Eren, 25/10/2019
//
// File: This is the header file for GPU_VDB that converts and loads
//		 a vdb file to CUDA 3d texture and loads it into gpu
//
//-----------------------------------------------



#ifndef __LIGHT_H__
#define __LIGHT_H__


#include <curand_kernel.h>
#include <helper_math.h>
#define _USE_MATH_DEFINES
#include <cmath>


typedef curandStatePhilox4_32_10_t Rand_state;
#define rand(state) curand_uniform(state)


__device__ inline float henyey_greenstein(
	float cos_theta,
	float g)
{

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}
__device__ inline float power_heuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f*f) / (f*f + g * g);
}

class light {

public:
	__host__ __device__ light():pos(make_float3(.0f)), dir(make_float3(.0f)), power(1.0f), color(make_float3(1.0f)) {}
	__host__ __device__ ~light() {};
	__host__ __device__ virtual int get_type() const=0;

	float3 pos;
	float3 dir;
	float power;
	float3 color;

	enum light_type {
		__INIT__	= 0,
		POINT_LIGHT = 1,
		AREA_LIGHT	= 2
	};

};



class point_light : public light {

public:

	__host__ __device__ point_light(){}
	__host__ __device__ point_light(float3 p, float3 cl, float pow) {
		pos = p;
		color = cl;
		power = pow;
	}

	__device__ float3 Le(Rand_state &randstate, float3 ray_pos, float3 ray_dir, float phase_g1, float3 tr, float max_density, float density_mult, float tr_depth) const {

		float3 Ld = make_float3(.0f);
		float3 wi;
		float phase_pdf = .0f;
		float eq_pdf = .0f;
		
		
		// Sample point light with phase pdf  
		wi = normalize(pos - ray_pos);
		float cos_theta = dot(ray_dir, wi);
		phase_pdf = henyey_greenstein(cos_theta, phase_g1);
		
		float falloff = 1 / length(pos*pos - ray_pos * ray_pos);

		float3 Li = color * power * tr  * phase_pdf * falloff;

		// Sample point light with equiangular pdf

		float delta = dot(pos - ray_pos, ray_dir);
		float D = length(ray_pos + ray_dir * delta - pos);

		float inv_max_density = 1.0f / max_density;
		float inv_density_mult = 1.0f / density_mult;

		float max_t = .0f;
		max_t -= logf(1 - rand(&randstate)) * inv_max_density * inv_density_mult * tr_depth;

		float thetaA = atan2f(.0f - delta, D);
		float thetaB = atan2f(max_t - delta, D);

		float t = D * tanf(lerp(thetaA, thetaB, rand(&randstate)));

		eq_pdf = D / ((thetaB - thetaA) * (D*D + t * t));
		float3 Leq = color * power * tr  * eq_pdf * falloff;

		float weight = power_heuristic(1, phase_pdf, 1, eq_pdf);

		Ld = (Li + Leq) * weight;
		
		return Ld;

	}

	__host__ __device__ int get_type() const { return POINT_LIGHT; }

};



class light_list {

public:
	__host__ __device__ light_list(unsigned int n_l){
		num_lights = n_l;	
	}
	__host__  __device__ ~light_list() {}

	unsigned int num_lights;
	point_light *light_ptr;

};




#endif