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
//	Version 1.0.1: Sergen Eren, 18/12/2019
//
// File: Geometry creation and processing kernels
//
//-----------------------------------------------

#define DDA_STEP_TRUE

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdio.h>
#include <float.h>

// Cuda includes
#include <cuda_runtime.h> 
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "helper_math.h"

typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

// Internal includes
#include "kernel_params.h"
#include "atmosphere/definitions.h"
#include "atmosphere/constants.h"
#include "gpu_vdb.h"
#include "camera.h"
#include "light.h"
#include "bvh/bvh.h"
//#include "geometry/sphere.h"
#include "geometry/geometry.h"

#define BLACK			make_float3(0.0f, 0.0f, 0.0f)
#define WHITE			make_float3(1.0f, 1.0f, 1.0f)
#define RED				make_float3(1.0f, 0.0f, 0.0f)
#define GREEN			make_float3(0.0f, 1.0f, 0.0f)
#define BLUE			make_float3(0.0f, 0.0f, 1.0f)
#define EPS				0.001f

#define INV_2_PI		1.0f / (2.0f * M_PI) 
#define INV_4_PI		1.0f / (4.0f * M_PI) 
#define INV_PI			1.0f / M_PI 


extern "C" __global__ void create_geometry_list(sphere **d_list, geometry_list **d_geo_list){

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		float3 center = make_float3(100, 320, -200);
		float radius = 100;

		d_list[0] = new sphere(center , radius, make_float3(0.18f), .001f);
		d_list[1] = new sphere(center+make_float3(0, 150 , 0) , radius, make_float3(0.18f), .001f);

		*d_geo_list = new geometry_list(*d_list, 2);
		
	}
}

// Test to see if geo_list is filled right 
extern "C" __global__ void test_geometry_list(
	const camera cam,
	const light_list lights,
	const GPU_VDB * gpu_vdb,
	const sphere & sphere,
	const geometry_list * *geo_list,
	BVHNode * root_node,
	OCTNode * oct_root,
	const AtmosphereParameters atmosphere,
	const Kernel_params kernel_params) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		float t_min, t_max;
		if ((*geo_list)->intersect(make_float3(0, 320, -200), make_float3(1, 0, 0), t_min, t_max)) printf("geometry test OK...\n");
		else printf("Error! geometry didn't create properly. \n");
	}
}