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
//	Version 1.0: Sergen Eren, 13/12/2019
//
// File: Sphere class with intersection routine 
//
//-----------------------------------------------


#ifndef _SPHERE_H_
#define _SPHERE_H_


#include "cuda_runtime_api.h"
#include "helper_math.h"



__device__ __host__ inline bool find_discr(
	float a,
	float b,
	float c,
	float& x1,
	float& x2)
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


class sphere {
public:

	__device__ __host__ sphere(){
		center = make_float3(.0f);
		radius = 1.0f;
		color = make_float3(1.0f);
		roughness = .0f;
	}

	__device__ __host__ sphere(float3 c, float rad){
		center = c;
		radius = rad;
		color = make_float3(1.0f);
		roughness = .0f;
	}

	__device__ __host__ ~sphere(){}

	__device__ __host__ bool intersect(float3 ray_pos, float3 ray_dir, float &t_min, float &t_max) const {

		float3 orig = ray_pos - center;
		

		float A = ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z;
		float B = 2 * (ray_dir.x * orig.x + ray_dir.y * orig.y + ray_dir.z * orig.z);
		float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

		if (!find_discr(A, B, C, t_min, t_max)) return false;

		if (t_min > t_max) {
			float tempt = t_max;
			t_max = t_min;
			t_min = tempt;
		}
		return true;

	}

	float3 center;
	float radius;
	float3 color;
	float roughness;
};




#endif // !_SPHERE_H_
