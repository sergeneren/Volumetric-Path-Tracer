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
//	Version 1.0: Sergen Eren, 08/12/2019
//
// File: 4 point plane with gpu intersection routine 
//
//-----------------------------------------------

#ifndef _PLANE_H_
#define _PLANE_H_

#include "cuda_runtime_api.h"
#include "helper_math.h"


struct triangle {

	__device__ __host__ triangle(float3 vtx0, float3 vtx1, float3 vtx2):v0(vtx0), v1(vtx1),v2(vtx2) {}
	__device__ __host__ triangle():v0(make_float3(0)), v1(make_float3(0,1,0)),v2(make_float3(1,0,0)) {}

	__device__ __host__ ~triangle(){}

	__device__ __host__ bool intersect(float3 ray_pos, float3 ray_dir , float &t) {

		float3 e1 = v1 - v0;
		float3 e2 = v2 - v0;

		float3 P = cross(ray_dir, e2);
		float det = dot(e1, P);

		// Not culling back-facing triangles
		if (det > -M_EPSILON && det < M_EPSILON) {
			return false;
		}

		float invDet = 1.0f / det;
		float3 T = ray_pos - v0;
		float u = dot(T, P)*invDet;

		if (u < 0.0f || u > 1.0f) {
			return false;
		}

		float3 Q = cross(T, e1);
		float v = dot(ray_dir, Q)*invDet;

		if (v < 0.0f || u + v > 1.0f) {
			return false;
		}

		float t0 = dot(e2, Q) * invDet;

		if (t0 > M_EPSILON && t0 < t) {
			t = t0;
			return true;
		}

		return false;


	}

	float3 v0;
	float3 v1;
	float3 v2;

};



struct plane {

	__device__ __host__ plane(float3 p0, float3 p1, float3 p2, float3 p3) {
		
		tri1 = triangle(p0, p1, p2);
		tri2 = triangle(p1, p2, p3);

	}
	__device__ __host__ plane() {

		float3 p0 = make_float3(0, 0, 0);
		float3 p1 = make_float3(1, 0, 0);
		float3 p2 = make_float3(0, 1, 0);
		float3 p3 = make_float3(1, 1, 0);

		tri1 = triangle(p0, p1, p2);
		tri2 = triangle(p1, p2, p3);

	}

	__device__ __host__ ~plane(){}
	   	  
	__device__ __host__ bool intersect(float3 ray_pos, float3 ray_dir, float &t) {

		if (tri1.intersect(ray_pos, ray_dir, t) || tri2.intersect(ray_pos, ray_dir, t)) return true;
		
		return false;

	}
	
	triangle tri1;
	triangle tri2;

};


#endif // !_PLANE_H_
