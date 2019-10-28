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


//#ifdef max
//	#undef max
//	#ifdef min
//		#undef min
//	#endif
//#endif




#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/Metadata.h>

#define ALIGN(x)	__align__(x)

// Transform matrix for gpu_vdb instance

struct ALIGN(16) VDB_INFO {

	float	voxelsize;
	int		dim[3];
	float	epsilon;
	float3	bmin;
	float3	bmax;

	cudaTextureObject_t density_texture; 
	cudaTextureObject_t emission_texture;

};


class GPU_VDB {

public:
	~GPU_VDB();
	GPU_VDB();
	__host__ bool loadVDB(std::string file_name, std::string density_channel, std::string emission_channel="");
	
	__host__ VDB_INFO * get_vdb_info();
	
	VDB_INFO vdb_info;

	__host__ __device__ mat4 get_xform() { return this->xform; }

private:

	__host__ void fill_texture(openvdb::GridBase::Ptr gridBase, cudaTextureObject_t &texture);
	
	mat4 xform;

	__host__ inline void set_vec3s(float3 &fl, openvdb::Vec3s vec) {

		fl.x = vec[0];
		fl.y = vec[1];
		fl.z = vec[2];
	}
	__host__ inline void set_vec3i(int *dim, openvdb::Vec3i vec) {

		dim[0] = vec[0];
		dim[1] = vec[1];
		dim[2] = vec[2];
	}

	__host__ inline void set_xform(mat4 &xform, openvdb::Mat4R matrix) {

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				xform[i][j] = matrix(i,j);
			}
		}

	}
};

#endif //endif _GPU_VDB_H_