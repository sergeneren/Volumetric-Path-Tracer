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
//	Version 1.0: Sergen Eren, 01/12/2019
//
// File: Implementation file for OCTree class functions  
//
//-----------------------------------------------

#include "bvh/octree.h"
#include <iostream>


// Returns root node 
OCTNode* OCTree::getRoot() {

	return root_node;
}

// Calculates children bounding boxes based on an index
//						    +------------+ pmax
//						   /  4   /  5  /|
//						  /______/_____/ |
//	      y				 /      /     /| |
//		|	z			+------------+ |/|
//		|  /			|   0  |  1  | / |
//		| /				|______|_____|/| /
//		|/				|      |     | |/
//		------ > x		|   2  |  3  | /
//				   pmin +------------+

AABB OCTree::divide_bbox(int idx, float3 pmin, float3 pmax) {

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

void OCTree::create_tree(std::vector<GPU_VDB> vdbs, OCTNode *root, int depth)
{
	if (depth > 0) {
		if (root->num_volumes > 0) {
			for (int i = 0; i < 8; ++i) {
				root->children[i] = new OCTNode;
				root->children[i]->parent = root;
				float3 pmin = root->bbox.pmin;
				float3 pmax = root->bbox.pmax;
				root->children[i]->bbox = divide_bbox(i, pmin, pmax);

				int idx = 0;
				for (int y = 0; y < vdbs.size(); ++y) {
					if (Overlaps(root->children[i]->bbox, vdbs.at(y).Bounds())) {
						root->children[i]->num_volumes++;
						root->children[i]->vol_indices[idx] = y;
						root->children[i]->max_extinction = fmaxf(root->children[i]->max_extinction, vdbs.at(y).vdb_info.max_density);
						idx++;
					}
				}

				if(m_debug) std::cout << "num volumes for child " << i << " at depth "<< depth << " is " << root->children[i]->num_volumes << "\n";

				create_tree(vdbs, root->children[i], depth - 1);
			}
		}
	}
}
