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
//	Version 1.0: Sergen Eren, 28/11/2019
//
// File: Axis Aligned Bounding Box for bvh contruction
//		 from https://github.com/henrikdahlberg/GPUPathTracer
//
//-----------------------------------------------

#ifndef _AABB_H_
#define _AABB_H_


#include "cuda_runtime_api.h"
#include "helper_math.h"

struct AABB {

	__host__ __device__ AABB() {
		pmin = make_float3(M_INF);
		pmax = make_float3(-M_INF);
	}
	
	__host__ __device__ AABB(const float3 &p) {
		pmin = p;
		pmax = p;
	}
	__host__ __device__ AABB(const float3 &p1, const float3 &p2) {

		pmin = p1;
		pmax = p2;
	}

	__host__ __device__ const float3 &operator[](int i) const {return (i == 0) ? pmin : pmax; }

	__host__ __device__ float3 &operator[](int i) {return (i == 0) ? pmin : pmax; }

	__host__ __device__ bool operator==(const AABB &b) const {
		return ((b.pmin.x == pmin.x && b.pmax.x == pmax.x) &&
			(b.pmin.y == pmin.y && b.pmax.y == pmax.y) &&
			(b.pmin.z == pmin.z && b.pmax.z == pmax.z));
	}

	__host__ __device__ bool operator!=(const AABB &b) const {
		return ((b.pmin.x != pmin.x || b.pmax.x != pmax.x) ||
			(b.pmin.y != pmin.y || b.pmax.y != pmax.y) ||
			(b.pmin.z != pmin.z || b.pmax.z != pmax.z));
	}

	__host__ __device__ float3 Diagonal() const { return pmax - pmin; }

	__host__ __device__ float3 Centroid() const { return 0.5f*(pmax + pmin); }

	__host__ __device__ float SurfaceArea() const {
		float3 d = Diagonal();
		return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
	}

	__host__ __device__ float Volume() const {
		float3 d = Diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ int MaximumDimension() const {
		float3 d = Diagonal();
		return (d.x > d.y && d.x > d.z) ? 0 : ((d.y > d.z) ? 1 : 2);
	}

	__host__ __device__ float3 Offset(const float3 &p) const {
		float3 o = p - pmin;
		if (pmax.x > pmin.x) o.x /= pmax.x - pmin.x;
		if (pmax.y > pmin.y) o.y /= pmax.y - pmin.y;
		if (pmax.z > pmin.z) o.z /= pmax.z - pmin.z;
		return o;
	}

	__host__ __device__ bool Intersect(const float3 &origin, const float3 &direction) const;

	float3 pmin;
	float3 pmax;

};

//////////////////////////////////////////////////////////////////////////
// Geometry inline functions
//////////////////////////////////////////////////////////////////////////


__host__ __device__ inline AABB UnionP(const AABB &b, const float3 &p) {
	return AABB(make_float3(fminf(b.pmin.x, p.x),	fminf(b.pmin.y, p.y), fminf(b.pmin.z, p.z)), 
				make_float3(fmaxf(b.pmax.x, p.x),	fmaxf(b.pmax.y, p.y), fmaxf(b.pmax.z, p.z)));
}

__host__ __device__ inline AABB UnionB(const AABB &b1, const AABB &b2) {
	return AABB(make_float3(fminf(b1.pmin.x, b2.pmin.x), fminf(b1.pmin.y, b2.pmin.y), fminf(b1.pmin.z, b2.pmin.z)), 
				make_float3(fmaxf(b1.pmax.x, b2.pmax.x), fmaxf(b1.pmax.y, b2.pmax.y), fmaxf(b1.pmax.z, b2.pmax.z)));
}

__host__ __device__ inline AABB Intersection(const AABB &b1, const AABB &b2) {
	return AABB(make_float3(fmaxf(b1.pmin.x, b2.pmin.x),	fmaxf(b1.pmin.y, b2.pmin.y),	fmaxf(b1.pmin.z, b2.pmin.z)),
				make_float3(fminf(b1.pmax.x, b2.pmax.x), fminf(b1.pmax.y, b2.pmax.y), fminf(b1.pmax.z, b2.pmax.z)));
}

__host__ __device__ inline bool Overlaps(const AABB &b1, const AABB &b2) {
	bool x = (b1.pmax.x >= b2.pmin.x) && (b1.pmin.x <= b2.pmax.x);
	bool y = (b1.pmax.y >= b2.pmin.y) && (b1.pmin.y <= b2.pmax.y);
	bool z = (b1.pmax.z >= b2.pmin.z) && (b1.pmin.z <= b2.pmax.z);
	return (x && y && z);
}

__host__ __device__ inline bool Contains(const AABB &b, const float3 &p) {
	return (p.x >= b.pmin.x && p.x <= b.pmax.x &&
		p.y >= b.pmin.y && p.y <= b.pmax.y &&
		p.z >= b.pmin.z && p.z <= b.pmax.z);

}

__host__ __device__ inline void BoundingSphere(const AABB &b, float3* position, float* radius) {
	*position = (b.pmin + b.pmax) * 0.5f;
	*radius = Contains(b, *position) ? length(*position - b.pmax) : 0;
}

__host__ __device__ inline bool ContainsExclusive(const AABB &b, const float3 &p) {
	return (p.x >= b.pmin.x && p.x < b.pmax.x &&
		p.y >= b.pmin.y && p.y < b.pmax.y &&
		p.z >= b.pmin.z && p.z < b.pmax.z);
}

__host__ __device__ inline AABB Expand(const AABB &b, const float delta) {
	return AABB(b.pmin - make_float3(delta),
		b.pmax + make_float3(delta));
}

__host__ __device__ inline bool AABB::Intersect(const float3 &origin, const float3 &direction) const {

	float3 directionInv = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);

	float t1 = (pmin.x - origin.x) * directionInv.x;
	float t2 = (pmax.x - origin.x) * directionInv.x;
	float t3 = (pmin.y - origin.y) * directionInv.y;
	float t4 = (pmax.y - origin.y) * directionInv.y;
	float t5 = (pmin.z - origin.z) * directionInv.z;
	float t6 = (pmax.z - origin.z) * directionInv.z;
	float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
	float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
	if (tmax <= 0.0f) return false; // box is behind
	if (tmin > tmax) return false; // ray missed

	return true;
}




struct AABBUnion {
	__host__ __device__ AABB operator()(const AABB &b1, const AABB &b2) const {
		return UnionB(b1, b2);
	}
};


struct OCTNode {

	__host__ __device__ bool isLeaf() { return !children; }

	int num_volumes = 0;
	int *vol_indices = new int[128];
	float max_extinction = .0f;
	float min_extinction = .0f;

	OCTNode *children[8];
	OCTNode *parent;
	AABB bbox = AABB();
};


#endif // !_AABB_H_

