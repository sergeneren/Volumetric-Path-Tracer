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

#pragma once

#include "BVH.h"
#include "CudaRenderKernel.h"
#include "Array.h"

//------------------------------------------------------------------------
// Nodes / BVHLayout_Compact  (12 floats + 4 ints = 64 bytes)
// innernode contains two childnodes c0 and c1, each having x,y,z coords for AABBhi and AABBlo, 2*2*3 = 12 floats 
//      
//		nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)  // 4 floats = 16 bytes
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)  // increment nodes array index with 16
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)  // either inner or leaf, 
//4 ints = 16 bytes, two dummy zeros at the end
//
// TriWoop / BVHLayout_Compact  (16 floats = 64 bits)
//      triWoop[triOfs*16 + 0 ] = Vec4f(woopZ)   // 4 floats = 16 bytes
//      triWoop[triOfs*16 + 16] = Vec4f(woopU)
//      triWoop[triOfs*16 + 32] = Vec4f(woopV)
//      triWoop[endOfs*16 + 0 ] = Vec4f(-0.0f, -0.0f, -0.0f, -0.0f)
//
// TriIndex / BVHLayout_Compact
//      triIndex[triOfs*4] = origIdx
//
//------------------------------------------------------------------------
//
// Following layouts used on older GPUs (pre-Fermi architecture)
//
// Nodes / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA  (64 bytes: 16 floats or 12 flats + 4 ints)
//      nodes[node*64  + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[node*64  + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*64  + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*64 + 48] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0) // either inner or leaf
//      nodes[leaf*64  + 48] = Vec4i(triStart, triEnd, 0, 0)
//
// Nodes / BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      nodes[node*16  + size*0/4] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)  // 16 ipv 64, geen bytes maar floats
//      nodes[node*16  + size*1/4] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*16  + size*2/4] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*16 + size*3/4] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0) // either inner or leaf
//      nodes[leaf*16  + size*3/4] = Vec4i(triStart, triEnd, 0, 0)
//
// TriWoop / BVHLayout_AOS_AOS, BVHLayout_SOA_AOS
//      triWoop[tri*64 + 0 ] = Vec4f(woopZ)
//      triWoop[tri*64 + 16] = Vec4f(woopU)
//      triWoop[tri*64 + 32] = Vec4f(woopV)
//
// TriWoop / BVHLayout_AOS_SOA, BVHLayout_SOA_SOA
//      triWoop[tri*16 + size*0/4] = Vec4f(woopZ)
//      triWoop[tri*16 + size*1/4] = Vec4f(woopU)
//      triWoop[tri*16 + size*2/4] = Vec4f(woopV)
//
// TriIndex / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA, BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      triIndex[tri*4] = origIdx
//------------------------------------------------------------------------

class CudaBVH
{
public:
	enum
	{
		Align = 4096
	};

public:
	explicit    CudaBVH(const BVH& bvh, BVHLayout layout);
	CudaBVH(CudaBVH& other)        { operator=(other); }
	~CudaBVH(void);

	BVHLayout   getLayout(void) const            { return m_layout; }
	Array<Vec4i>&  getNodeBuffer(void)            { return m_nodes; }
	Array<Vec4i>&  getTriWoopBuffer(void)         { return m_triWoop; }
	Array<S32>&    getTriIndexBuffer(void)        { return m_triIndex; }

	Vec4i*  getGpuNodes(void)            { return m_gpuNodes; }
	Vec4i*  getGpuTriWoop(void)         { return m_gpuTriWoop; }
	Vec4i*  getDebugTri(void)			{ return m_debugTri;  }
	S32*    getGpuTriIndices(void)        { return m_gpuTriIndices; }

	U32    getGpuNodesSize(void)			{ return m_gpuNodesSize; }
	U32    getGpuTriWoopSize(void)			{ return m_gpuTriWoopSize; }
	U32    getDebugTriSize(void)			{ return m_debugTriSize; }
	U32    getGpuTriIndicesSize(void)        { return m_gpuTriIndicesSize; }
	U32    getLeafnodeCount(void)			{ return m_leafnodecount; }
	U32    getTriCount(void)			{ return m_tricount; }

	// AOS: idx ignored, returns entire buffer
	// SOA: 0 <= idx < 4, returns one subarray  // idx between 0 and 4
	Vec2i       getNodeSubArray(int idx) const; // (ofs, size)
	Vec2i       getTriWoopSubArray(int idx) const; // (ofs, size)

	CudaBVH&    operator=(CudaBVH& other);

private:
	void        createNodeBasic(const BVH& bvh);
	void        createTriWoopBasic(const BVH& bvh);
	void        createTriIndexBasic(const BVH& bvh);
	void        createCompact(const BVH& bvh, int nodeOffsetSizeDiv);
	void        woopifyTri(const BVH& bvh, int idx);

private:
	BVHLayout   m_layout;
	
	Array<Vec4i>      m_nodes; 
	Array<Vec4i>      m_triWoop;
	Array<S32>        m_triIndex;

	Vec4i*	m_gpuNodes;
	Vec4i*  m_gpuTriWoop;
	Vec4i*  m_debugTri;
	S32*	m_gpuTriIndices;

	U32     m_gpuNodesSize;
	U32		m_gpuTriWoopSize;
	U32     m_debugTriSize;
	U32		m_gpuTriIndicesSize;
	U32		m_leafnodecount;
	U32     m_tricount;

	Vec4f   m_woop[3];
	Vec4f	m_debugtri[3];
};

//------------------------------------------------------------------------
