#pragma once

#include "linear_math.h"

struct Vertex : public Vec3f
{
	Vec3f _normal;
	// ambient occlusion of this vertex (pre-calculated in e.g. MeshLab)

	Vertex(float x, float y, float z, float nx, float ny, float nz, float amb = 60.f)
		:
		Vec3f(x, y, z), _normal(Vec3f(nx, ny, nz))
	{
		// assert |nx,ny,nz| = 1
	}
};

struct Triangle {
	// indexes in vertices array
	unsigned _idx1;
	unsigned _idx2;
	unsigned _idx3;
};

