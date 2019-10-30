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
//	Version 1.0: Sergen Eren, 28/10/2019
//
// File: This is the header file for Camera class 
//
//-----------------------------------------------

#ifndef __CAMERA_H_
#define __CAMERA_H_


#include <curand_kernel.h>
#include <helper_math.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef curandStatePhilox4_32_10_t Rand_state;
#define rand(state) curand_uniform(state)


__device__ float vanDerCorput(Rand_state *local_rand_state, int base = 2) {

	int n = int(rand(local_rand_state) * 100);
	float rand_int = 0, denom = 1, invBase = 1.f / base;

	while (n) {

		denom *= base;
		rand_int += (n%base) / denom;
		n *= invBase;

	}
	return rand_int;
}


__device__ float3 random_in_unit_disk(Rand_state *local_rand_state) {

	float3 p;
	do {

		p = 2.0f * make_float3(vanDerCorput(local_rand_state), vanDerCorput(local_rand_state, 3), 0) - make_float3(1.0f, 1.0f, 0.0f);

	} while (dot(p, p) >= 1.0);
	return p;

}

class ray
{
public:
	__device__ ray() {};
	__device__ ray(const float3& a, const float3& b, float ti = 0.0) { A = a; B = b; _time = ti; };
	__device__ float3 origin() const { return A; }
	__device__ float3 direction() const { return B; }
	__device__ float time() const { return _time; }
	__device__ float3 point_at_parameter(float t) const { return A + t * B; }

	float3 A;
	float3 B;
	float _time;
};



class camera
{
public:
	__host__ __device__ camera():
		time0(.0f), time1(1.0f), 
		origin(make_float3(10.0f, .0f, .0f)),
		lower_left_corner(make_float3(.0f, .0f, .0f)),
		horizontal(make_float3(-1.0f, .0f, .0f)),
		vertical(make_float3(.0f, 1.0f, .0f)),
		u(make_float3(1.0f, .0f, .0f)),
		v(make_float3(.0f, 1.0f, .0f)),
		w(make_float3(.0f, .0f, 1.0f)),
		lens_radius(25.0f){}



	__host__ __device__ void update_camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect, float aperture) {

		float focus_dist = length(lookfrom - lookat);
		lens_radius = aperture / 2.0f;
		float theta = vfov * float(M_PI) / 180.0f;
		float half_height = tan(theta / 2.0f);
		float half_width = aspect * half_height;
		origin = lookfrom;

		w = normalize(lookfrom - lookat);
		u = normalize(cross(vup, w));
		v = cross(w, u);

		lower_left_corner = origin - half_width * focus_dist*u - half_height * focus_dist*v - focus_dist * w;

		horizontal = 2.0f*half_width*focus_dist*u;

		vertical = 2.0f*half_height*focus_dist*v;
		
	}

	__device__ ray get_ray(float s, float t, Rand_state *local_rand_state) const{
		float3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		float3 offset = u * rd.x + v * rd.y;
		float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
	}


	float time1, time0;
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 u, v, w;
	float lens_radius;
};

#endif