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

#ifndef _GPU_VDB_H_
#define _GPU_VDB_H_

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "texture_types.h"
#include "matrix_math.h"
#include <string>

#include "helper_math.h"
#include "bvh/AABB.h"

#define ALIGN(x)	__align__(x)

#define NOHIT			1.0e10f


// Transform matrix for gpu_vdb instance

struct ALIGN(16) VDB_INFO {

	float	voxelsize;
	int3	dim;
	float3	bmin;
	float3	bmax;
	float	max_density;
	float	min_density;

	bool has_color;
	bool has_emission;
	bool matte;

	cudaTextureObject_t density_texture; 
	cudaTextureObject_t emission_texture;
	cudaTextureObject_t color_texture;

};


class GPU_VDB {

public:
	__host__ ~GPU_VDB();
	__host__ GPU_VDB();
	__host__ GPU_VDB(const GPU_VDB& copy);

	// Device functions
	__device__ float3 rayBoxIntersect(float3 ray_pos, float3 ray_dir) const {

		// World space to object space
		ray_pos = xform.transpose().inverse().transform_point(ray_pos);
		ray_dir = xform.transpose().inverse().transform_vector(ray_dir);

		register float ht[8];
		ht[0] = (vdb_info.bmin.x - ray_pos.x) / ray_dir.x;
		ht[1] = (vdb_info.bmax.x - ray_pos.x) / ray_dir.x;
		ht[2] = (vdb_info.bmin.y - ray_pos.y) / ray_dir.y;
		ht[3] = (vdb_info.bmax.y - ray_pos.y) / ray_dir.y;
		ht[4] = (vdb_info.bmin.z - ray_pos.z) / ray_dir.z;
		ht[5] = (vdb_info.bmax.z - ray_pos.z) / ray_dir.z;
		ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
		ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));
		ht[6] = (ht[6] < 0) ? 0.0f : ht[6];
				
		return make_float3(ht[6], ht[7], (ht[7] < ht[6] || ht[7] < 0) ? NOHIT : 0);
			
	}

	__device__ bool inVolumeBbox(float3 ray_pos) const {

		// World space to object space
		ray_pos = xform.transpose().inverse().transform_point(ray_pos);

		float3 min = vdb_info.bmin;
		float3 max = vdb_info.bmax;

		return ray_pos.x >= min.x && ray_pos.y >= min.y && ray_pos.z >= min.z && ray_pos.x < max.x && ray_pos.y < max.y && ray_pos.z < max.z;
	}

	// Host functions
	__host__ bool loadVDB(std::string file_name, std::string density_channel, std::string emission_channel="", std::string color_channel="");
	
	__host__ VDB_INFO * get_vdb_info() ;
	
	__host__ GPU_VDB * clone();

	// Host and device functions
	__host__ __device__ mat4 get_xform() const { return this->xform; }
	
	__host__ __device__ void set_xform(mat4 &matrix) { this->xform = matrix; }
	
	__host__ __device__ AABB Bounds() const {
		
		// OOB to AABB conversion from
		// https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/

		float3 center = (vdb_info.bmax + vdb_info.bmin) * 0.5;
		float3 extent = (vdb_info.bmax - vdb_info.bmin) * 0.5;

		float3 new_center = xform.transpose().transform_point(center);
		float3 new_extent = xform.abs().transpose().transform_vector(extent);

		AABB bounding_box(new_center - new_extent, new_center + new_extent);
	
		return bounding_box;

	}

	VDB_INFO vdb_info;

private:

	mat4 xform;

};


class GPU_PROC_VOL : virtual public GPU_VDB {


public: 

	__host__ GPU_PROC_VOL();
	__host__ GPU_PROC_VOL(const GPU_PROC_VOL& copy);
	__host__ ~GPU_PROC_VOL();

	__host__ bool create_volume(float3 min, float3 max, float res);

private:

	CUmodule texture_module;
	CUfunction fill_buffer_function;
	float *device_density_buffer;
	float resolution;
	int3 dimensions;
};


#endif //endif _GPU_VDB_H_