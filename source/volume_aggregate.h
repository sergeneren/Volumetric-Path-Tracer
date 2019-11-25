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
//	Version 1.0: Sergen Eren, 25/11/2019
//
// File: This is the header file for volume_aggregate that is an octree to 
//		 hold gpu_vdb instances. It creates a bvh for ray traversal and provides 
//		 functions for intersection and metadata providing. 
//
//-----------------------------------------------



#ifndef __VOLUME_AGGREGATE_H__
#define __VOLUME_AGGREGATE_H__

#include "cuda_runtime_api.h"
#include "gpu_vdb/gpu_vdb.h"

class volume_aggregate {


	class iterator {

		__device__ __host__ float start() const;
		__device__ __host__ float end() const;
		__device__ __host__ float max_extinction() const;
		__device__ __host__ float min_extinction() const;
		__device__ __host__ bool finished() const;
		__device__ __host__ void operator++();
		__device__ __host__ int num_volumes() const ;
		__device__ __host__ GPU_VDB* volumes() const;


	};


	__device__ __host__ void insert(const GPU_VDB *vdbs);
	iterator iterator() const;

};





#endif // !__VOLUME_AGGREGATE_H__
