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
//	Version 1.0: Sergen Eren, 09/12/2019
//
// File: This class builds a shadow pyramid out of given direction and 
//		 atmosphere class. the shadow box consists of 12 planes and 
//		 is used for fog shadow calculations 
//
//-----------------------------------------------



#include "shadow_box.h"
#include "helper_cuda.h"


float degree_to_radians(float degree)
{
	return degree * M_PI / 180.0f;
}


float3 degree_to_cartesian(float azimuth, float elevation)
{
	float az = clamp(azimuth, .0f, 360.0f);
	float el = clamp(elevation, -90.0f, 90.0f);

	az = degree_to_radians(az);
	el = degree_to_radians(90.0f - el);

	float x = sinf(el) * cosf(az);
	float y = cosf(el);
	float z = sinf(el) * sinf(az);

	return normalize(make_float3(x, y, z));
}

bool solveQuadratic(
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


bool raySphereIntersect(
	const float3& orig,
	const float3& dir,
	const float& radius,
	float& t0,
	float& t1)
{

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


float3 project_to_earth(float3 point, float3 dir, float bottom_radius, float top_radius) {

	float t0, t1;

	if (raySphereIntersect(point + bottom_radius, dir, bottom_radius, t0, t1)) {
		point += dir * t0;
		return point;
	}
	if (raySphereIntersect(point + bottom_radius, dir, top_radius, t0, t1)) point += dir * t0;

	return point;

}

shadow_box::shadow_box(){}

shadow_box::~shadow_box() {

	cudaFree(cuda_planes);
	delete[] bbox_planes;

}

void shadow_box::build_planes(float azimuth, float elevation, AABB root_box, float bottom_radius, float top_radius) {


	if(cuda_planes) cudaFree(cuda_planes);

	float3 p0;
	float3 p1;
	float3 pr_0;
	float3 pr_1;
	
	float3 pmin = root_box.pmin;
	float3 pmax = root_box.pmax;

	//Find sun direction 
	float3 l_dir = -normalize(degree_to_cartesian(azimuth, elevation));

	// First Plane 
	p0 = pmin;
	p1 = make_float3(pmin.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[0] = plane(p0, p1, pr_1, pr_0);

	// Second Plane 
	p0 = pmin;
	p1 = make_float3(pmax.x, pmin.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[1] = plane(p0, p1, pr_1, pr_0);

	// Third Plane 
	p0 = pmin;
	p1 = make_float3(pmin.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[2] = plane(p0, p1, pr_1, pr_0);

	// Fourth Plane 
	p0 = pmax;
	p1 = make_float3(pmax.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[3] = plane(p0, p1, pr_1, pr_0);

	// Fifth Plane 
	p0 = pmax;
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[4] = plane(p0, p1, pr_1, pr_0);

	// Sixth Plane 
	p0 = pmax;
	p1 = make_float3(pmin.x, pmax.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[5] = plane(p0, p1, pr_1, pr_0);




	// Seventh Plane 
	p0 = make_float3(pmin.x, pmax.y, pmin.z);
	p1 = make_float3(pmin.x, pmax.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[6] = plane(p0, p1, pr_1, pr_0);

	// Eighth Plane 
	p0 = make_float3(pmin.x, pmax.y, pmin.z);
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[7] = plane(p0, p1, pr_1, pr_0);

	// Nineth Plane 
	p0 = make_float3(pmin.x, pmax.y, pmax.z);
	p1 = make_float3(pmin.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[8] = plane(p0, p1, pr_1, pr_0);

	// Tenth Plane 
	p0 = make_float3(pmin.x, pmin.y, pmax.z);
	p1 = make_float3(pmax.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[9] = plane(p0, p1, pr_1, pr_0);

	// Eleventh Plane 
	p0 = make_float3(pmax.x, pmin.y, pmax.z);
	p1 = make_float3(pmax.x, pmin.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[10] = plane(p0, p1, pr_1, pr_0);

	// Twelfth Plane 
	p0 = make_float3(pmax.x, pmin.y, pmin.z);
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, bottom_radius, top_radius);
	pr_1 = project_to_earth(p1, l_dir, bottom_radius, top_radius);
	bbox_planes[11] = plane(p0, p1, pr_1, pr_0);


	checkCudaErrors(cudaMalloc(&cuda_planes, 12 * sizeof(plane)));
	checkCudaErrors(cudaMemcpy(cuda_planes, &bbox_planes, 12 * sizeof(plane), cudaMemcpyHostToDevice));


}