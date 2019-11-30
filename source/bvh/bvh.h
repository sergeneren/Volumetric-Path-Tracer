#ifndef _BVH_H_
#define _BVH_H_

#include <cuda_runtime.h>
#include "AABB.h"


typedef unsigned long long MortonCode;


struct BVHNode {

	__host__ __device__ inline bool IsLeaf() { return !leftChild && !rightChild; }

	int minId;
	int maxId;
	int volIndex;

	BVHNode *leftChild;
	BVHNode *rightChild;
	BVHNode *parent;

	AABB boundingBox;
};

class BVH {

public:
	BVH(){}
	virtual ~BVH(){}

	BVHNode* getRoot();

	BVHNode *BVHNodes;
	BVHNode *BVHLeaves;
	int numVolumes;
};

#endif
