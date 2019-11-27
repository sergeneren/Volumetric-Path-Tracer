/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CudaBVH.h"
#include "SceneLoader.h"  // required for triangles and vertices

//Nodes / BVHLayout_Compact  (12 floats + 4 ints = 64 bytes)
// innernode contains two childnodes c0 and c1, each having x,y,z coords for AABBhi and AABBlo, 2*2*3 = 12 floats 
//      
//		nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)  // 4 floats = 16 bytes
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)  // increment nodes array index with 16
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)  // either inner or leaf, two dummy zeros at the end
//		CudaBVH Compact: Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//		CudaBVH Compact: Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//		CudaBVH Compact: Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//		CudaBVH Compact: Vec4f(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)
//		BVH node bounds: c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y, c0.lo.z, c0.hi.z

static int woopcount = 0;  // counts Woopified triangles

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
	: m_layout(layout)
{
	FW_ASSERT(layout >= 0 && layout < BVHLayout_Max);

	if (layout == BVHLayout_Compact)
	{
		createCompact(bvh, 1);
		return;
	}

	if (layout == BVHLayout_Compact2)
	{
		createCompact(bvh, 16);
		return;
	}
}

CudaBVH::~CudaBVH(void)
{
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getNodeSubArray(int idx) const
{
	FW_ASSERT(idx >= 0 && idx < 4);
	S32 size = (S32)m_nodes.getSize();

	if (m_layout == BVHLayout_SOA_AOS || m_layout == BVHLayout_SOA_SOA)
		return Vec2i((size >> 2) * idx, (size >> 2));
	return Vec2i(0, size);
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getTriWoopSubArray(int idx) const
{
	FW_ASSERT(idx >= 0 && idx < 4);
	S32 size = (S32)m_triWoop.getSize();

	if (m_layout == BVHLayout_AOS_SOA || m_layout == BVHLayout_SOA_SOA)
		return Vec2i((size >> 2) * idx, (size >> 2));
	return Vec2i(0, size);
}

//------------------------------------------------------------------------

CudaBVH& CudaBVH::operator=(CudaBVH& other)  
{
	if (&other != this)
	{
		m_layout = other.m_layout;
		m_nodes = other.m_nodes;
		m_triWoop = other.m_triWoop;
		m_triIndex = other.m_triIndex;
	}
	return *this;
}

namespace detail
{
struct StackEntry
{
    const BVHNode*  node;
    S32             idx;

    StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
};
}

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
{
    using namespace detail; // for StackEntry

	int leafcount = 0; // counts leafnodes

	// construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function). 

	Array<Vec4i> nodeData(NULL, 4); 
	Array<Vec4i> triWoopData;
	Array<Vec4i> triDebugData; // array for regular (non-woop) triangles
	Array<S32> triIndexData;

	// construct a stack (array of stack entries) to help in filling the data arrays
	Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0)); // initialise stack to rootnode

	while (stack.getSize()) // while stack is not empty
	{
		StackEntry e = stack.removeLast(); // pop the stack
		FW_ASSERT(e.node->getNumChildNodes() == 2);
		const AABB* cbox[2];   
		int cidx[2]; // stores indices to both children

		// Process children.

		// for each child in entry e
		for (int i = 0; i < 2; i++)
		{
			const BVHNode* child = e.node->getChildNode(i); // current childnode
			cbox[i] = &child->m_bounds; // current child's AABB

			////////////////////////////
			/// INNER NODE
			//////////////////////////////

			// Inner node => push to stack.

			if (!child->isLeaf()) // no leaf, thus an inner node
			{   // compute childindex
				cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv; // nodeOffsetSizeDiv is 1 for Fermi kernel, 16 for Kepler kernel		
				
				// push the current child on the stack
				stack.add(StackEntry(child, nodeData.getSize()));   
				nodeData.add(NULL, 4); /// adds 4 * Vec4i per inner node or 4 * 16 bytes/Vec4i = 64 bytes of empty data per inner node
				continue; // process remaining childnode (if any)
			}



			//////////////////////
			/// LEAF NODE
			/////////////////////

			// Leaf => append triangles.

			const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
			
			// index of a leafnode is a negative number, hence the ~
			cidx[i] = ~triWoopData.getSize();  // leafs must be stored as negative (bitwise complement) in order to be recognised by pathtracer as a leaf
		
			// for each triangle in leaf, range of triangle index j from m_lo to m_hi 
			for (int j = leaf->m_lo; j < leaf->m_hi; j++) 
			{
				// transform the triangle's vertices to Woop triangle (simple transform to right angled triangle, see paper by Sven Woop)
				woopifyTri(bvh, j);  /// j is de triangle index in triIndex array
				
				if (m_woop[0].x == 0.0f) m_woop[0].x = 0.0f;  // avoid degenerate coordinates
				// add the transformed woop triangle to triWoopData
				
				triWoopData.add((Vec4i*)m_woop, 3);  
				
				triDebugData.add((Vec4i*)m_debugtri, 3);  

				// add tri index for current triangle to triIndexData	
				triIndexData.add(bvh.getTriIndices()[j]); 
				triIndexData.add(0); // zero padding because CUDA kernel uses same index for vertex array (3 vertices per triangle)
				triIndexData.add(0); // and array of triangle indices
			}

			// Leaf node terminator to indicate end of leaf, stores hexadecimal value 0x80000000 (= 2147483648 in decimal)
			triWoopData.add(0x80000000); // leafnode terminator code indicates the last triangle of the leaf node
			triDebugData.add(0x80000000); 
			
			// add extra zero to triangle indices array to indicate end of leaf
			triIndexData.add(0);  // terminates triIndexdata for current leaf

			leafcount++;
		}

		// Write entry for current node.  
		/// 4 Vec4i per node (according to compact bvh node layout)
		Vec4i* dst = nodeData.getPtr(e.idx);
		///std::cout << "e.idx: " << e.idx << " cidx[0]: " << cidx[0] << " cidx[1]: " << cidx[1] << "\n";
		dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
		dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
		dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
		dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);

	} // end of while loop, will iteratively empty the stack


	m_leafnodecount = leafcount;
	m_tricount = woopcount;

	// Write data arrays to arrays of CudaBVH

	m_gpuNodes = (Vec4i*) malloc(nodeData.getNumBytes());
	m_gpuNodesSize = nodeData.getSize();
	
	for (int i = 0; i < nodeData.getSize(); i++){	
		m_gpuNodes[i].x = nodeData.get(i).x;
		m_gpuNodes[i].y = nodeData.get(i).y;
		m_gpuNodes[i].z = nodeData.get(i).z;
		m_gpuNodes[i].w = nodeData.get(i).w; // child indices
	}	

	m_gpuTriWoop = (Vec4i*) malloc(triWoopData.getSize() * sizeof(Vec4i));
	m_gpuTriWoopSize = triWoopData.getSize();

	for (int i = 0; i < triWoopData.getSize(); i++){
		m_gpuTriWoop[i].x = triWoopData.get(i).x;
		m_gpuTriWoop[i].y = triWoopData.get(i).y;
		m_gpuTriWoop[i].z = triWoopData.get(i).z;
		m_gpuTriWoop[i].w = triWoopData.get(i).w;
	}

	m_debugTri = (Vec4i*)malloc(triDebugData.getSize() * sizeof(Vec4i));
	m_debugTriSize = triDebugData.getSize();

	for (int i = 0; i < triDebugData.getSize(); i++){
		m_debugTri[i].x = triDebugData.get(i).x;
		m_debugTri[i].y = triDebugData.get(i).y;
		m_debugTri[i].z = triDebugData.get(i).z;
		m_debugTri[i].w = triDebugData.get(i).w; 
	}

	m_gpuTriIndices = (S32*) malloc(triIndexData.getSize() * sizeof(S32));
	m_gpuTriIndicesSize = triIndexData.getSize();

	for (int i = 0; i < triIndexData.getSize(); i++){
		m_gpuTriIndices[i] = triIndexData.get(i);
	}
}

//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int triIdx)
{	
	woopcount++;

	// fetch the 3 vertex indices of this triangle
	const Vec3i& vtxInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[triIdx]).vertices; 
  const Vec3f& v0 = bvh.getScene()->getVertex(vtxInds.x);
  const Vec3f& v1 = bvh.getScene()->getVertex(vtxInds.y);
	const Vec3f& v2 = bvh.getScene()->getVertex(vtxInds.z);
	
	// regular triangles (for debugging only)
	m_debugtri[0] = Vec4f(v0.x, v0.y, v0.z, 0.0f);
	m_debugtri[1] = Vec4f(v1.x, v1.y, v1.z, 0.0f);
	m_debugtri[2] = Vec4f(v2.x, v2.y, v2.z, 0.0f);

	Mat4f mtx;
	// compute edges and transform them with a matrix 
	mtx.setCol(0, Vec4f(v0 - v2, 0.0f)); // sets matrix column 0 equal to a Vec4f(Vec3f, 0.0f )
	mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
	mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
	mtx.setCol(3, Vec4f(v2, 1.0f));
	mtx = invert(mtx);   

	/// m_woop[3] stores 3 transformed triangle edges
	m_woop[0] = Vec4f(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3)); // elements of 3rd row of inverted matrix
 	m_woop[1] = mtx.getRow(0); 
	m_woop[2] = mtx.getRow(1); 
}

//------------------------------------------------------------------------
