
#ifndef _OCTREE_H_
#define _OCTREE_H_

#include <cuda_runtime.h>
#include "AABB.h"


struct OCTNode {

	__host__ __device__ bool isLeaf() {return !children;}
	
	int num_volumes;
	int *vol_indices;
	float max_extinction;
	float min_extinction;

	OCTNode *children[8];
	OCTNode *parent;
	AABB bbox;
};


class OCTree {

	OCTree() {};
	OCTree() {};

	OCTNode* getRoot() { return root_node; }

	OCTNode *root_node;
	
};

#endif // !_OCTREE_H_
