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

#include "cuda_runtime_api.h"

#include "texture_types.h"
#include "matrix_math.h"
#include <string>


#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/Metadata.h>

#include "helper_math.h"

#define ALIGN(x)	__align__(x)

#define NOHIT			1.0e10f


// Transform matrix for gpu_vdb instance

struct ALIGN(16) VDB_INFO {

	float	voxelsize;
	int3	dim;
	float	epsilon;
	float3	bmin;
	float3	bmax;

	cudaTextureObject_t density_texture; 
	cudaTextureObject_t emission_texture;

};


class GPU_VDB {

public:
	__host__ __device__ ~GPU_VDB();
	__host__ __device__ GPU_VDB();

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
	__host__ bool loadVDB(std::string file_name, std::string density_channel, std::string emission_channel="");
	__host__ VDB_INFO * get_vdb_info() ;
	
	// Host and device functions
	__host__ __device__ mat4 get_xform() const { return this->xform; }
	
	VDB_INFO vdb_info;
private:

	__host__ void fill_texture(openvdb::GridBase::Ptr gridBase, cudaTextureObject_t &texture);
	
	

	__host__ inline void set_vec3s(float3 &fl, openvdb::Vec3s vec) {

		fl.x = vec[0];
		fl.y = vec[1];
		fl.z = vec[2];
	}
	__host__ inline void set_vec3i(int3 &dim, openvdb::Vec3i vec) {

		dim.x = vec[0];
		dim.y = vec[1];
		dim.z = vec[2];
	}

	__host__ inline void set_xform(mat4 &xform, openvdb::Mat4R matrix) {

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				xform[i][j] = float(matrix(j,i));
			}
		}

	}
	
	mat4 xform;

};

#endif //endif _GPU_VDB_H_