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
//	Version 1.0: Sergen Eren, 18/12/2019
//
// File: General geometry class that parents geo types 
//
//-----------------------------------------------


#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_


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



class geometry {

public:

	__device__ virtual int intersect(float3 ray_pos, float3 ray_dir, float& t_min, float& t_max) const = 0;
	__device__ virtual bool scatter(float3& ray_pos, float3& ray_dir, float t_min, float3& normal, float3& atten, Rand_state rand_state) const = 0;
};


class sphere : public geometry {

public:

	__device__ __host__ sphere() {
		center = make_float3(.0f);
		radius = 1.0f;
		color = make_float3(1.0f);
		roughness = .0f;
	}

	__device__ __host__ sphere(float3 c, float rad) {
		center = c;
		radius = rad;
		color = make_float3(1.0f);
		roughness = .0f;
	}

	__device__ __host__ sphere(float3 c, float rad, float3 col) {
		center = c;
		radius = rad;
		color = col;
		roughness = 1.0f;
	}

	__device__ __host__ sphere(float3 c, float rad, float3 col, float r) {
		center = c;
		radius = rad;
		color = col;
		roughness = r;
	}

	__device__ virtual int intersect(float3 ray_pos, float3 ray_dir, float& t_min, float& t_max) const {

		float3 orig = ray_pos - center;

		float A = ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z;
		float B = 2 * (ray_dir.x * orig.x + ray_dir.y * orig.y + ray_dir.z * orig.z);
		float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

		if (!find_discr(A, B, C, t_min, t_max)) return 0;

		if (t_min > t_max) {
			float tempt = t_max;
			t_max = t_min;
			t_min = tempt;
		}

		if (t_min < 0) {
			t_min = t_max;
			if (t_min < 0) return 0;
		}

		return 1;

	};


	__device__ virtual bool scatter(float3& ray_pos, float3& ray_dir, float t_min, float3& normal, float3& atten, Rand_state rand_state) const {

		ray_dir = normalize(ray_dir);
		ray_pos += ray_dir * t_min;
		normal = normalize((ray_pos - center) / radius);
		float3 nl = dot(normal, ray_dir) < 0 ? normal : normal * -1;

		float phi = 2 * M_PI * rand(&rand_state);
		float r2 = rand(&rand_state);
		float r2s = sqrtf(r2);

		float3 w = normalize(nl);
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		float3 hemisphere_dir = normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
		float3 ref = reflect(ray_dir, nl);
		ray_dir = lerp(ref, hemisphere_dir, roughness);

		ray_pos += ray_dir * 0.1;

		atten *= color;

		return true;
	}


	float3 center;
	float radius;
	float3 color;
	float roughness;

};


class geometry_list : public geometry {

public:

	__device__ geometry_list() {};
	__device__ geometry_list(geometry** l, int n) { list = l; list_size = n; };

	__device__ virtual int intersect(float3 ray_pos, float3 ray_dir, float& t_min, float& t_max) const {

		int idx = -1;
		float temp_tmin = FLT_MAX;
		for (int i = 0; i < list_size; i++) {

			if (list[i]->intersect(ray_pos, ray_dir, t_min, t_max)) {

				if (t_min < temp_tmin) {
					temp_tmin = t_min;
					idx = i;
				}

			}

		}

		t_min = temp_tmin;

		return idx;

	}

	__device__ virtual bool scatter(float3& ray_pos, float3& ray_dir, float t_min, float3& normal, float3& atten, Rand_state rand_state) const {

		float t_max;
		int idx = this->intersect(ray_pos, ray_dir, t_min, t_max);

		if (idx > -1) {

			list[idx]->scatter(ray_pos, ray_dir, t_min, normal, atten, rand_state);
			return true;

		}

		return false;

	}



	geometry** list = NULL;
	int list_size = 0;

};



#endif // !_GEOMETRY_H_