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
//	Version 1.0: Sergen Eren, 29/11/2019
//
// File: Contains the kernels for construction of volume bvh on gpu 
//		 from https://github.com/henrikdahlberg/GPUPathTracer
//
//-----------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <helper_math.h>

#include "bvh.h"
#include "gpu_vdb.h"

#define BLOCK_SIZE 32

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}


//////////////////////////////////////////////////////////////////////////
// Device functions
//////////////////////////////////////////////////////////////////////////

/**
* Longest common prefix for Morton code
*/
__device__ int LongestCommonPrefix(int i, int j, int numTriangles,
	MortonCode* mortonCodes, int* triangleIDs) {
	if (i < 0 || i > numTriangles - 1 || j < 0 || j > numTriangles - 1) {
		return -1;
	}

	MortonCode mi = mortonCodes[i];
	MortonCode mj = mortonCodes[j];

	if (mi == mj) {
		return __clzll(mi ^ mj) + __clzll(triangleIDs[i] ^ triangleIDs[j]);
	}
	else {
		return __clzll(mi ^ mj);
	}
}
/**
* Expand bits, used in Morton code calculation
*/
__device__ MortonCode bitExpansion(MortonCode i) {
	i = (i * 0x00010001u) & 0xFF0000FFu;
	i = (i * 0x00000101u) & 0x0F00F00Fu;
	i = (i * 0x00000011u) & 0xC30C30C3u;
	i = (i * 0x00000005u) & 0x49249249u;
	return i;
}

/**
* Compute morton code given volume centroid scaled to [0,1] of scene bounding box
*/
__device__ MortonCode ComputeMortonCode(float x, float y, float z) {

	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	MortonCode xx = bitExpansion((MortonCode)x);
	MortonCode yy = bitExpansion((MortonCode)y);
	MortonCode zz = bitExpansion((MortonCode)z);
	return xx * 4 + yy * 2 + zz;

}

__device__ AABB divide_bbox(int idx, float3 pmin, float3 pmax) {

	float3 min = make_float3(.0f);
	float3 max = make_float3(.0f);
	float half_x = (pmin.x + pmax.x)*0.5;
	float half_y = (pmin.y + pmax.y)*0.5;
	float half_z = (pmin.z + pmax.z)*0.5;

	if (idx == 0) {
		min = make_float3(pmin.x, half_y, pmin.z);
		max = make_float3(half_x, pmax.y, half_z);
	}

	if (idx == 1) {
		min = make_float3(half_x, half_y, pmin.z);
		max = make_float3(pmax.x, pmax.y, half_z);
	}

	if (idx == 2) {
		min = pmin;
		max = make_float3(half_x, half_y, half_z);
	}

	if (idx == 3) {
		min = make_float3(half_x, pmin.y, pmin.z);
		max = make_float3(pmax.x, half_y, half_z);
	}


	if (idx == 4) {
		min = make_float3(pmin.x, half_y, half_z);
		max = make_float3(half_x, pmax.y, pmax.z);
	}

	if (idx == 5) {
		min = make_float3(half_x, half_y, half_z);
		max = pmax;
	}

	if (idx == 6) {
		min = make_float3(pmin.x, pmin.y, half_z);
		max = make_float3(half_x, half_y, pmax.z);
	}

	if (idx == 7) {
		min = make_float3(half_x, pmin.y, half_z);
		max = make_float3(pmax.x, half_y, pmax.z);

	}

	return AABB(min, max);

}

__device__ void build_octree_recursive(GPU_VDB *vdbs, int num_volumes, OCTNode *root, int depth, bool m_debug) {

	if (depth > 0) {
		if (root->num_volumes > 0) {
			for (int i = 0; i < 8; ++i) {

				root->children[i] = new OCTNode;
				root->children[i]->parent = root;
				root->children[i]->depth = depth;
				float3 pmin = root->bbox.pmin;
				float3 pmax = root->bbox.pmax;
				root->children[i]->bbox = divide_bbox(i, pmin, pmax);

				int idx = 0;
				for (int y = 0; y < num_volumes; ++y) {
					if (Overlaps(root->children[i]->bbox, vdbs[y].Bounds())) {
						root->children[i]->num_volumes++;
						root->children[i]->vol_indices[idx] = y;
						root->children[i]->max_extinction = fmaxf(root->children[i]->max_extinction, vdbs[y].vdb_info.max_density);
						root->children[i]->voxel_size = fminf(root->children[i]->voxel_size, vdbs[y].vdb_info.voxelsize);
						idx++;
					}
				}
				if (root->children[i]->num_volumes>0) root->children[i]->has_children = true;
				if (m_debug) {
					printf("num volumes for child %d-%d is %d ", depth, i, root->children[i]->num_volumes);
					if (root->children[i]->num_volumes > 0) {
						printf("volume indices: ");
						for (int x = 0; x < root->children[i]->num_volumes; ++x) {
							printf("%d ", root->children[i]->vol_indices[x]);
						}
					}
					printf(" max extinction: %f\n", root->children[i]->max_extinction);
				}
				
				build_octree_recursive(vdbs, num_volumes, root->children[i], depth - 1, m_debug);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// Kernels
//////////////////////////////////////////////////////////////////////////


__global__ void ComputeBoundingBoxes(GPU_VDB* volumes,
	int numVolumes,
	AABB* boundingBoxes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numVolumes) boundingBoxes[i] = volumes[i].Bounds();
}

__global__ void DebugBVH(BVHNode* BVHLeaves, BVHNode* BVHNodes, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// do in serial
	if (i == 0) {
		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = BVHLeaves + j;
			printf("BBox for volumeIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}
		
		//parents:
		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent;
			printf("BBox for parent node of volumeIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}
		
		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent;
			printf("BBox for parents parent node of volumeIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}

		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent->parent;
			printf("BBox for parents parents parent node of volumeIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}
		
	}

}

__global__ void ComputeMortonCodes(const GPU_VDB* volumes, int numTriangles, AABB sceneBounds, MortonCode* mortonCodes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (i < numTriangles) {

		// Compute volume centroid
		float3 centroid = volumes[i].Bounds().Centroid();

		// Normalize triangle centroid to lie within [0,1] of scene bounding box
		float x = (centroid.x - sceneBounds.pmin.x) / (sceneBounds.pmax.x - sceneBounds.pmin.x);
		float y = (centroid.y - sceneBounds.pmin.y) / (sceneBounds.pmax.y - sceneBounds.pmin.y);
		float z = (centroid.z - sceneBounds.pmin.z) / (sceneBounds.pmax.z - sceneBounds.pmin.z);

		// Compute morton code
		mortonCodes[i] = ComputeMortonCode(x, y, z);
	}
	
}

__global__ void ConstructBVH(BVHNode* BVHNodes, BVHNode* BVHLeaves, int* nodeCounter, GPU_VDB* volumes, int* volumeIDs, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes) {
		BVHNode* leaf = BVHLeaves + i;

		int volumeIdx = volumeIDs[i];
		// Handle leaf first
		leaf->volIndex = volumeIdx;
		leaf->boundingBox = volumes[volumeIdx].Bounds();

		BVHNode* current = leaf->parent;
		int currentIndex = current - BVHNodes;

		int res = atomicAdd(nodeCounter + currentIndex, 1);

		// Go up and handle internal nodes
		while (true) {
			if (res == 0) {
				return;
			}
			AABB leftBoundingBox = current->leftChild->boundingBox;
			AABB rightBoundingBox = current->rightChild->boundingBox;

			// Compute current bounding box
			current->boundingBox = UnionB(leftBoundingBox, rightBoundingBox);

			// If current is root, return
			if (current == BVHNodes) {
				return;
			}
			current = current->parent;
			currentIndex = current - BVHNodes;
			res = atomicAdd(nodeCounter + currentIndex, 1);
		}
	}
}

__global__ void BuildRadixTree(BVHNode* radixTreeNodes, BVHNode* radixTreeLeaves, MortonCode* mortonCodes, int* volumeIds, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes - 1) {
		// Run radix tree construction algorithm
		// Determine direction of the range (+1 or -1)
		int d = LongestCommonPrefix(i, i + 1, numVolumes, mortonCodes, volumeIds) -
			LongestCommonPrefix(i, i - 1, numVolumes, mortonCodes, volumeIds) >= 0 ? 1 : -1;

		// Compute upper bound for the length of the range
		int deltaMin = LongestCommonPrefix(i, i - d, numVolumes, mortonCodes, volumeIds);
		//int lmax = 128;
		int lmax = 2;

		while (LongestCommonPrefix(i, i + lmax * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
			//lmax = lmax * 4;
			lmax = lmax * 2;
		}

		// Find the other end using binary search
		int l = 0;
		int divider = 2;
		for (int t = lmax / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (l + t) * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
				l = l + t;
			}
			if (t == 1) break;
			t = lmax / divider;
		}

		int j = i + l * d;

		// Find the split position using binary search
		int deltaNode = LongestCommonPrefix(i, j, numVolumes, mortonCodes, volumeIds);
		int s = 0;
		divider = 2;
		for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (s + t) * d, numVolumes, mortonCodes, volumeIds) > deltaNode) {
				s = s + t;
			}
			if (t == 1) break;
			t = (l + (divider - 1)) / divider;
		}

		int gamma = i + s * d + min(d, 0);

		//printf("i:%d, d:%d, deltaMin:%d, deltaNode:%d, lmax:%d, l:%d, j:%d, gamma:%d. \n", i, d, deltaMin, deltaNode, lmax, l, j, gamma);

		// Output child pointers
		BVHNode* current = radixTreeNodes + i;

		if (min(i, j) == gamma) {
			current->leftChild = radixTreeLeaves + gamma;
			(radixTreeLeaves + gamma)->parent = current;
		}
		else {
			current->leftChild = radixTreeNodes + gamma;
			(radixTreeNodes + gamma)->parent = current;
		}

		if (max(i, j) == gamma + 1) {
			current->rightChild = radixTreeLeaves + gamma + 1;
			(radixTreeLeaves + gamma + 1)->parent = current;
		}
		else {
			current->rightChild = radixTreeNodes + gamma + 1;
			(radixTreeNodes + gamma + 1)->parent = current;
		}

		current->minId = min(i, j);
		current->maxId = max(i, j);
	}
}

__global__ void pass_octree(GPU_VDB *volumes, int num_volumes, OCTNode *root, int depth, bool m_debug) {

	build_octree_recursive(volumes, num_volumes, root, depth, m_debug);
}

extern "C" void BuildBVH(BVH& bvh, GPU_VDB* volumes, int numVolumes, AABB &sceneBounds, bool debug_bvh) {

	int blockSize = BLOCK_SIZE;
	int gridSize = (numVolumes + blockSize - 1) / blockSize;

	// Timing metrics
	float total = 0;
	float elapsed;
	cudaEvent_t start, stop;

	std::cout << "Number of volumes: " << numVolumes << std::endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Pre-process stage, scene bounding box
	// TODO: add check if this has been done already
	//		 if we already have scenebounds and have new/modified triangles, no need to start over
	// Should only do this if scene has changed (added tris, moved tris)

	// Compute bounding boxes
	
	std::cout << "Computing volume bounding boxes...";
	cudaEventRecord(start, 0);
	thrust::device_vector<AABB> boundingBoxes(numVolumes);
	ComputeBoundingBoxes <<<gridSize, blockSize>>> (volumes, numVolumes, boundingBoxes.data().get());
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;

	thrust::host_vector<AABB> bounding_boxes_h = boundingBoxes;

	// Compute scene bounding box
	std::cout << "Computing scene bounding box...";
	cudaEventRecord(start, 0);
	sceneBounds = thrust::reduce(boundingBoxes.begin(), boundingBoxes.end(), AABB(), AABBUnion());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;
	std::cout << "Total pre-computation time for scene was " << total << " ms.\n" << std::endl;
	total = 0;

	std::cout << "Scene boundingbox:\n";
	std::cout << "pmin: " << sceneBounds.pmin.x << ", " << sceneBounds.pmin.y << ", " << sceneBounds.pmin.z << std::endl;
	std::cout << "pmax: " << sceneBounds.pmax.x << ", " << sceneBounds.pmax.y << ", " << sceneBounds.pmax.z << std::endl;

	// Pre-process done, start building BVH

	// Compute Morton codes
	thrust::device_vector<MortonCode> mortonCodes(numVolumes);
	std::cout << "Computing Morton codes...";
	cudaEventRecord(start, 0);
	ComputeMortonCodes <<<gridSize, blockSize>>> (volumes, numVolumes, sceneBounds, mortonCodes.data().get());
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Sort triangle indices with Morton code as key
	thrust::device_vector<int> triangleIDs(numVolumes);
	thrust::sequence(triangleIDs.begin(), triangleIDs.end());
	std::cout << "Sort volumes...";
	cudaEventRecord(start, 0);
	try {
		thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), triangleIDs.begin());
	}
	catch (thrust::system_error e) {
		std::cout << "Error inside sort: " << e.what() << std::endl;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Sorting took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build radix tree of BVH nodes
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHNodes, (numVolumes - 1) * sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHLeaves, numVolumes * sizeof(BVHNode)));
	std::cout << "Building radix tree...";
	cudaEventRecord(start, 0);
	BuildRadixTree <<<gridSize, blockSize>>> (bvh.BVHNodes, bvh.BVHLeaves, mortonCodes.data().get(), triangleIDs.data().get(), numVolumes);
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build BVH
	thrust::device_vector<int> nodeCounters(numVolumes);
	std::cout << "Building BVH...";
	cudaEventRecord(start, 0);
	ConstructBVH <<<gridSize, blockSize >>> (bvh.BVHNodes, bvh.BVHLeaves,	nodeCounters.data().get(), volumes,	triangleIDs.data().get(), numVolumes);
	CudaCheckError();
	checkCudaErrors(cudaDeviceSynchronize());
	
	if(debug_bvh) DebugBVH << <gridSize, blockSize >> >(bvh.BVHLeaves, bvh.BVHNodes, numVolumes);
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	std::cout << "Total BVH construction time was " << total << " ms.\n" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

extern "C" void build_octree(OCTNode *root, GPU_VDB *volumes, int num_volumes, int depth, bool m_debug) {

	pass_octree << <1, 1 >> > (volumes, num_volumes, root, depth, m_debug);

}