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
//	Version 1.0: Sergen Eren, 27/10/2019
//
// File: Column major 4x4 matrix for CUDA.
// source https://gist.github.com/mattatz/1f3234e3f978706b46eb32493b3f9fa9
//
//-----------------------------------------------

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#ifdef _MSC_VER
#pragma warning( disable : 4244)
#endif // _MSC_VER

#include "cuda_runtime.h"
#include "helper_cuda.h"

struct mat4 {
	float m[4][4];

	__host__ __device__ __forceinline__ mat4() {
		m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0; m[3][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0; m[3][2] = 0.0;
		m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 1.0;
	}

	__host__ __device__ __forceinline__ mat4(
		const float m11, const float m12, const float m13, const float m14,
		const float m21, const float m22, const float m23, const float m24,
		const float m31, const float m32, const float m33, const float m34,
		const float m41, const float m42, const float m43, const float m44
	) {
		m[0][0] = m11; m[1][0] = m12; m[2][0] = m13; m[3][0] = m14;
		m[0][1] = m21; m[1][1] = m22; m[2][1] = m23; m[3][1] = m24;
		m[0][2] = m31; m[1][2] = m32; m[2][2] = m33; m[3][2] = m34;
		m[0][3] = m41; m[1][3] = m42; m[2][3] = m43; m[3][3] = m44;
	}

	__host__ __device__ __forceinline__ float* operator[] (const size_t idx) {
		return m[idx];
	}

	__host__ __device__ __forceinline__ float4 operator*(const float4& v) const {
		float4 ret;
		ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w;
		ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w;
		ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w;
		ret.w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w;
		return ret;
	}

	__host__ __device__ __forceinline__ float3 operator*(const float3& v) const {
		float3 ret;

		ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
		ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
		ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];

		return ret;
	}

	__host__ __device__ __forceinline__ mat4 operator*(const float f) const {
		mat4 ret;
		ret[0][0] = m[0][0] * f; ret[1][0] = m[1][0] * f; ret[2][0] = m[2][0] * f; ret[3][0] = m[3][0] * f;
		ret[0][1] = m[0][1] * f; ret[1][1] = m[1][1] * f; ret[2][1] = m[2][1] * f; ret[3][1] = m[3][1] * f;
		ret[0][2] = m[0][2] * f; ret[1][2] = m[1][2] * f; ret[2][2] = m[2][2] * f; ret[3][2] = m[3][2] * f;
		ret[0][3] = m[0][3] * f; ret[1][3] = m[1][3] * f; ret[2][3] = m[2][3] * f; ret[3][3] = m[3][3] * f;
		return ret;
	}

	__host__ __device__ __forceinline__ mat4 operator/(const float f) const {
		mat4 ret;
		ret[0][0] = m[0][0] / f; ret[1][0] = m[1][0] / f; ret[2][0] = m[2][0] / f; ret[3][0] = m[3][0] / f;
		ret[0][1] = m[0][1] / f; ret[1][1] = m[1][1] / f; ret[2][1] = m[2][1] / f; ret[3][1] = m[3][1] / f;
		ret[0][2] = m[0][2] / f; ret[1][2] = m[1][2] / f; ret[2][2] = m[2][2] / f; ret[3][2] = m[3][2] / f;
		ret[0][3] = m[0][3] / f; ret[1][3] = m[1][3] / f; ret[2][3] = m[2][3] / f; ret[3][3] = m[3][3] / f;
		return ret;
	}

	__host__ __device__ __forceinline__ mat4 operator+(const mat4& other) const {
		mat4 ret;
		ret[0][0] = m[0][0] + other.m[0][0]; ret[1][0] = m[1][0] + other.m[1][0]; ret[2][0] = m[2][0] + other.m[2][0]; ret[3][0] = m[3][0] + other.m[3][0];
		ret[0][1] = m[0][1] + other.m[0][1]; ret[1][1] = m[1][1] + other.m[1][1]; ret[2][1] = m[2][1] + other.m[2][1]; ret[3][1] = m[3][1] + other.m[3][1];
		ret[0][2] = m[0][2] + other.m[0][2]; ret[1][2] = m[1][2] + other.m[1][2]; ret[2][2] = m[2][2] + other.m[2][2]; ret[3][2] = m[3][2] + other.m[3][2];
		ret[0][3] = m[0][3] + other.m[0][3]; ret[1][3] = m[1][3] + other.m[1][3]; ret[2][3] = m[2][3] + other.m[2][3]; ret[3][3] = m[3][3] + other.m[3][3];
		return ret;
	}

	__host__ __device__ __forceinline__ mat4 operator-(const mat4& other) const {
		mat4 ret;
		ret[0][0] = m[0][0] - other.m[0][0]; ret[1][0] = m[1][0] - other.m[1][0]; ret[2][0] = m[2][0] - other.m[2][0]; ret[3][0] = m[3][0] - other.m[3][0];
		ret[0][1] = m[0][1] - other.m[0][1]; ret[1][1] = m[1][1] - other.m[1][1]; ret[2][1] = m[2][1] - other.m[2][1]; ret[3][1] = m[3][1] - other.m[3][1];
		ret[0][2] = m[0][2] - other.m[0][2]; ret[1][2] = m[1][2] - other.m[1][2]; ret[2][2] = m[2][2] - other.m[2][2]; ret[3][2] = m[3][2] - other.m[3][2];
		ret[0][3] = m[0][3] - other.m[0][3]; ret[1][3] = m[1][3] - other.m[1][3]; ret[2][3] = m[2][3] - other.m[2][3]; ret[3][3] = m[3][3] - other.m[3][3];
		return ret;
	}

	__host__ __device__ __forceinline__ mat4 operator*(const mat4& other) const {
		auto a11 = m[0][0], a12 = m[1][0], a13 = m[2][0], a14 = m[3][0];
		auto a21 = m[0][1], a22 = m[1][1], a23 = m[2][1], a24 = m[3][1];
		auto a31 = m[0][2], a32 = m[1][2], a33 = m[2][2], a34 = m[3][2];
		auto a41 = m[0][3], a42 = m[1][3], a43 = m[2][3], a44 = m[3][3];

		auto b11 = other.m[0][0], b12 = other.m[1][0], b13 = other.m[2][0], b14 = other.m[3][0];
		auto b21 = other.m[0][1], b22 = other.m[1][1], b23 = other.m[2][1], b24 = other.m[3][1];
		auto b31 = other.m[0][2], b32 = other.m[1][2], b33 = other.m[2][2], b34 = other.m[3][2];
		auto b41 = other.m[0][3], b42 = other.m[1][3], b43 = other.m[2][3], b44 = other.m[3][3];

		mat4 ret;
		ret[0][0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
		ret[0][1] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
		ret[0][2] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
		ret[0][3] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

		ret[1][0] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
		ret[1][1] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
		ret[1][2] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
		ret[1][3] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

		ret[2][0] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
		ret[2][1] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
		ret[2][2] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
		ret[2][3] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

		ret[3][0] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
		ret[3][1] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
		ret[3][2] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
		ret[3][3] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
		return ret;
	}

	__host__ __device__ __forceinline__ mat4 transpose() const {
		mat4 ret;
		ret[0][0] = m[0][0]; ret[0][1] = m[1][0]; ret[0][2] = m[2][0]; ret[0][3] = m[3][0];
		ret[1][0] = m[0][1]; ret[1][1] = m[1][1]; ret[1][2] = m[2][1]; ret[1][3] = m[3][1];
		ret[2][0] = m[0][2]; ret[2][1] = m[1][2]; ret[2][2] = m[2][2]; ret[2][3] = m[3][2];
		ret[3][0] = m[0][3]; ret[3][1] = m[1][3]; ret[3][2] = m[2][3]; ret[3][3] = m[3][3];
		return ret;
	}

	__host__ __device__ __forceinline__ float det() const {
		auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
		auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
		auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
		auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

		return (
			n41 * (
				+n14 * n23 * n32
				- n13 * n24 * n32
				- n14 * n22 * n33
				+ n12 * n24 * n33
				+ n13 * n22 * n34
				- n12 * n23 * n34
				) +
			n42 * (
				+n11 * n23 * n34
				- n11 * n24 * n33
				+ n14 * n21 * n33
				- n13 * n21 * n34
				+ n13 * n24 * n31
				- n14 * n23 * n31
				) +
			n43 * (
				+n11 * n24 * n32
				- n11 * n22 * n34
				- n14 * n21 * n32
				+ n12 * n21 * n34
				+ n14 * n22 * n31
				- n12 * n24 * n31
				) +
			n44 * (
				-n13 * n22 * n31
				- n11 * n23 * n32
				+ n11 * n22 * n33
				+ n13 * n21 * n32
				- n12 * n21 * n33
				+ n12 * n23 * n31
				)
			);
	}

	__host__ __device__ __forceinline__ mat4 inverse() const {
		auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
		auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
		auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
		auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

		auto t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
		auto t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
		auto t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
		auto t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

		auto det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
		auto idet = 1.0f / det;

		mat4 ret;

		ret[0][0] = t11 * idet;
		ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
		ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
		ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

		ret[1][0] = t12 * idet;
		ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
		ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
		ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

		ret[2][0] = t13 * idet;
		ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
		ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
		ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

		ret[3][0] = t14 * idet;
		ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
		ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
		ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

		return ret;
	}

	__host__ __device__ __forceinline__ void zero() {
		m[0][0] = 0.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 0.0; m[2][1] = 0.0; m[3][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 0.0; m[3][2] = 0.0;
		m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 0.0;
	}

	__host__ __device__ __forceinline__ void identity() {
		m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0; m[3][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0; m[3][2] = 0.0;
		m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 1.0;
	}

	__host__ __device__ __forceinline__ mat4 abs() const {
		mat4 ret;

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				float temp = fabsf(m[j][i]);
				ret[j][i] = temp;
			}
		}

		return ret;
	}

	__host__ __device__ __forceinline__ void print() {
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				printf("%f ", m[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	__host__ __device__ __forceinline__ float3 transform_point(const float3& pt) const {
		float4 temp = make_float4(pt.x, pt.y, pt.z, 1.0f);
		temp = *this * temp;
		return make_float3(temp.x, temp.y, temp.z);
	}
	__host__ __device__ __forceinline__ float3 transform_vector(const float3& pt) const {
		float4 temp = make_float4(pt.x, pt.y, pt.z, 0.0f);
		temp = *this * temp;
		return make_float3(temp.x, temp.y, temp.z);
	}

	__host__ __device__ __forceinline__ mat4 rotate_zyx(const float3& angs) {
		float cx, sx, cy, sy, cz, sz;
		cx = (float)cos(angs.x);
		sx = (float)sin(angs.x);
		cy = (float)cos(angs.y);
		sy = (float)sin(angs.y);
		cz = (float)cos(angs.z);
		sz = (float)sin(angs.z);

		m[0][0] = cz * cy;
		m[1][0] = sz * cy;
		m[2][0] = -sy;

		m[0][1] = -sz * cx + cz * sy * sx;
		m[1][1] = cz * cx - sz * sy * sz;
		m[2][1] = -cy * sx;

		m[0][2] = -sz * sx + cz * sy * cx;
		m[1][2] = cz * sx + sz * sy * cx;
		m[2][2] = cy * cx;

		return *this;
	}

	__host__ __device__ __forceinline__ float3 extract_translate() {
		return make_float3(m[0][3], m[1][3], m[2][3]);
	}

	__host__ __device__ __forceinline__ mat4 translate(const float3& val) {
		m[0][3] += val.x;
		m[1][3] += val.y;
		m[2][3] += val.z;

		return *this;
	}

	__host__ __device__ __forceinline__ mat4 scale(const float3& val) {
		m[0][0] *= val.x;
		m[1][1] *= val.y;
		m[2][2] *= val.z;

		return *this;
	}

	__host__ __device__ __forceinline__ mat4& operator*=(const float f) { return *this = *this * f; }
	__host__ __device__ __forceinline__ mat4& operator/=(const float f) { return *this = *this / f; }
	__host__ __device__ __forceinline__ mat4& operator+=(const mat4& m) { return *this = *this + m; }
	__host__ __device__ __forceinline__ mat4& operator-=(const mat4& m) { return *this = *this - m; }
	__host__ __device__ __forceinline__ mat4& operator*=(const mat4& m) { return *this = *this * m; }
};

//////////////////////////////////////////////////////////////////////////
// Inline Functions
//////////////////////////////////////////////////////////////////////////

__host__ __device__ __forceinline__ mat4 toMatrix(double* arr) {
	mat4 ret;

	ret[0][0] = arr[0]; ret[1][0] = arr[1]; ret[2][0] = arr[2]; ret[3][0] = 0.0f;
	ret[0][1] = arr[3]; ret[1][1] = arr[4]; ret[2][1] = arr[5]; ret[3][1] = 0.0f;
	ret[0][2] = arr[6]; ret[1][2] = arr[7]; ret[2][2] = arr[8]; ret[3][2] = 0.0f;
	ret[0][3] = .0f; ret[1][3] = .0f; ret[2][3] = 0.0f; ret[3][3] = 1.0f;

	return ret;
}

__host__ __device__ __forceinline__ mat4 toMatrix(float* arr) {
	mat4 ret;

	ret[0][0] = arr[0]; ret[1][0] = arr[1]; ret[2][0] = arr[2]; ret[3][0] = 0.0f;
	ret[0][1] = arr[3]; ret[1][1] = arr[4]; ret[2][1] = arr[5]; ret[3][1] = 0.0f;
	ret[0][2] = arr[6]; ret[1][2] = arr[7]; ret[2][2] = arr[8]; ret[3][2] = 0.0f;
	ret[0][3] = .0f; ret[1][3] = .0f; ret[2][3] = 0.0f; ret[3][3] = 1.0f;

	return ret;
}

__host__ __device__ __forceinline__ mat4 quaternion_to_mat4(double x, double y, double z, double w) {
	const double n = 1.0 / sqrtf(x * x + y * y + z * z + w * w);

	x *= n;
	y *= n;
	z *= n;
	w *= n;

	float m11 = float(1.0f - 2.0f * y * y - 2.0f * z * z);
	float m12 = float(2.0f * x * y + 2.0f * z * w);
	float m13 = float(2.0f * x * z - 2.0f * y * w);
	float m14 = 0.0f;

	float m21 = float(2.0f * x * y - 2.0f * z * w);
	float m22 = float(1.0f - 2.0f * x * x - 2.0f * z * z);
	float m23 = float(2.0f * y * z + 2.0f * x * w);
	float m24 = 0.0f;

	float m31 = float(2.0f * x * z + 2.0f * y * w);
	float m32 = float(2.0f * y * z - 2.0f * x * w);
	float m33 = float(1.0f - 2.0f * x * x - 2.0f * y * y);
	float m34 = 0.0f;

	float m41 = 0.0f;
	float m42 = 0.0f;
	float m43 = 0.0f;
	float m44 = 1.0f;

	mat4 ret(m11, m12, m13, m14,
		m21, m22, m23, m24,
		m31, m32, m33, m34,
		m41, m42, m43, m44);

	return ret;
}

__host__ __device__ __forceinline__ mat4 quaternion_to_mat4(float4 quaternion) {
	float x = quaternion.x;
	float y = quaternion.y;
	float z = quaternion.z;
	float w = quaternion.w;

	const float n = 1.0f / sqrtf(x * x + y * y + z * z + w * w);

	x *= n;
	y *= n;
	z *= n;
	w *= n;

	mat4 m1 = mat4(w, z, -y, x,
		-z, w, x, y,
		y, -x, w, z,
		-x, -y, -z, w);

	mat4 m2 = mat4(w, z, -y, -x,
		-z, w, x, -y,
		y, -x, w, -z,
		x, y, z, w);

	mat4 ret = m1 * m2;

	return ret;
}

// Rotation by quaternion about point
__host__ __device__ __forceinline__ mat4 rotate_by_point(float4 q, float3 center) {
	mat4 ret;

	float sqw = q.w * q.w;
	float sqx = q.x * q.x;
	float sqy = q.y * q.y;
	float sqz = q.z * q.z;

	ret[0][0] = sqx - sqy - sqz + sqw; // since sqw + sqx + sqy + sqz =1
	ret[1][1] = -sqx + sqy - sqz + sqw;
	ret[2][2] = -sqx - sqy + sqz + sqw;

	float tmp1 = q.x * q.y;
	float tmp2 = q.z * q.w;
	ret[1][0] = 2.0f * (tmp1 + tmp2);
	ret[0][1] = 2.0f * (tmp1 - tmp2);

	tmp1 = q.x * q.z;
	tmp2 = q.y * q.w;
	ret[2][0] = 2.0f * (tmp1 - tmp2);
	ret[0][2] = 2.0f * (tmp1 + tmp2);

	tmp1 = q.y * q.z;
	tmp2 = q.x * q.w;
	ret[2][1] = 2.0f * (tmp1 + tmp2);
	ret[1][2] = 2.0f * (tmp1 - tmp2);

	float a1 = center.x, a2 = center.y, a3 = center.z;

	ret[3][0] = a1 - a1 * ret[0][0] - a2 * ret[1][0] - a3 * ret[2][0];
	ret[3][1] = a2 - a1 * ret[0][1] - a2 * ret[1][1] - a3 * ret[2][1];
	ret[3][2] = a3 - a1 * ret[0][2] - a2 * ret[1][2] - a3 * ret[2][2];
	ret[0][3] = ret[1][3] = ret[2][3] = 0.0f;
	ret[3][3] = 1.0f;

	return ret;
}

struct mat3 {
	float m[3][3];

	__host__ __device__ __forceinline__ mat3() {
		m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0;
	}

	__host__ __device__ __forceinline__ mat3(
		const float m11, const float m12, const float m13,
		const float m21, const float m22, const float m23,
		const float m31, const float m32, const float m33
	) {
		m[0][0] = m11; m[1][0] = m12; m[2][0] = m13;
		m[0][1] = m21; m[1][1] = m22; m[2][1] = m23;
		m[0][2] = m31; m[1][2] = m32; m[2][2] = m33;
	}

	__host__ __device__ __forceinline__ float* operator[] (const size_t idx) {
		return m[idx];
	}

	__host__ __device__ __forceinline__ float3 operator*(const float3& v) const {
		float3 ret;

		ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z;
		ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z;
		ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z;

		return ret;
	}

	__host__ __device__ __forceinline__ mat3 operator*(const float f) const {
		mat3 ret;
		ret[0][0] = m[0][0] * f; ret[1][0] = m[1][0] * f; ret[2][0] = m[2][0] * f;
		ret[0][1] = m[0][1] * f; ret[1][1] = m[1][1] * f; ret[2][1] = m[2][1] * f;
		ret[0][2] = m[0][2] * f; ret[1][2] = m[1][2] * f; ret[2][2] = m[2][2] * f;
		return ret;
	}

	__host__ __device__ __forceinline__ mat3 operator/(const float f) const {
		mat3 ret;
		ret[0][0] = m[0][0] / f; ret[1][0] = m[1][0] / f; ret[2][0] = m[2][0] / f;
		ret[0][1] = m[0][1] / f; ret[1][1] = m[1][1] / f; ret[2][1] = m[2][1] / f;
		ret[0][2] = m[0][2] / f; ret[1][2] = m[1][2] / f; ret[2][2] = m[2][2] / f;
		return ret;
	}

	__host__ __device__ __forceinline__ mat3 operator+(const mat3& other) const {
		mat3 ret;
		ret[0][0] = m[0][0] + other.m[0][0]; ret[1][0] = m[1][0] + other.m[1][0]; ret[2][0] = m[2][0] + other.m[2][0];
		ret[0][1] = m[0][1] + other.m[0][1]; ret[1][1] = m[1][1] + other.m[1][1]; ret[2][1] = m[2][1] + other.m[2][1];
		ret[0][2] = m[0][2] + other.m[0][2]; ret[1][2] = m[1][2] + other.m[1][2]; ret[2][2] = m[2][2] + other.m[2][2];
		return ret;
	}

	__host__ __device__ __forceinline__ mat3 operator-(const mat3& other) const {
		mat3 ret;
		ret[0][0] = m[0][0] - other.m[0][0]; ret[1][0] = m[1][0] - other.m[1][0]; ret[2][0] = m[2][0] - other.m[2][0];
		ret[0][1] = m[0][1] - other.m[0][1]; ret[1][1] = m[1][1] - other.m[1][1]; ret[2][1] = m[2][1] - other.m[2][1];
		ret[0][2] = m[0][2] - other.m[0][2]; ret[1][2] = m[1][2] - other.m[1][2]; ret[2][2] = m[2][2] - other.m[2][2];
		return ret;
	}

	__host__ __device__ __forceinline__ mat3 operator*(const mat3& other) const {
		auto a11 = m[0][0], a12 = m[1][0], a13 = m[2][0];
		auto a21 = m[0][1], a22 = m[1][1], a23 = m[2][1];
		auto a31 = m[0][2], a32 = m[1][2], a33 = m[2][2];

		auto b11 = other.m[0][0], b12 = other.m[1][0], b13 = other.m[2][0];
		auto b21 = other.m[0][1], b22 = other.m[1][1], b23 = other.m[2][1];
		auto b31 = other.m[0][2], b32 = other.m[1][2], b33 = other.m[2][2];

		mat3 ret;
		ret[0][0] = a11 * b11 + a12 * b21 + a13 * b31;
		ret[0][1] = a11 * b12 + a12 * b22 + a13 * b32;
		ret[0][2] = a11 * b13 + a12 * b23 + a13 * b33;

		ret[1][0] = a21 * b11 + a22 * b21 + a23 * b31;
		ret[1][1] = a21 * b12 + a22 * b22 + a23 * b32;
		ret[1][2] = a21 * b13 + a22 * b23 + a23 * b33;

		ret[2][0] = a31 * b11 + a32 * b21 + a33 * b31;
		ret[2][1] = a31 * b12 + a32 * b22 + a33 * b32;
		ret[2][2] = a31 * b13 + a32 * b23 + a33 * b33;

		return ret;
	}

	__host__ __device__ __forceinline__ mat3 transpose() const {
		mat3 ret;
		ret[0][0] = m[0][0]; ret[0][1] = m[1][0]; ret[0][2] = m[2][0];
		ret[1][0] = m[0][1]; ret[1][1] = m[1][1]; ret[1][2] = m[2][1];
		ret[2][0] = m[0][2]; ret[2][1] = m[1][2]; ret[2][2] = m[2][2];
		return ret;
	}

	__host__ __device__ __forceinline__ float det() const {
		auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0];
		auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1];
		auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2];

		return (n11 * (n22 * n33 - n23 * n32) - n12 * (n21 * n33 - n23 * n31) + n13 * (n21 * n32 - n22 * n31));
	}

	__host__ __device__ __forceinline__ mat3 inverse() const {
		float det = this->det();

		if (det == 0) return *this;

		mat3 ret;

		ret = this->transpose();

		auto det1 = ret[2][2] * ret[1][1] - ret[2][1] * ret[1][2];
		auto det2 = ret[0][1] * ret[2][2] - ret[2][1] * ret[0][2];
		auto det3 = ret[0][1] * ret[1][1] - ret[1][1] * ret[0][2];

		auto det4 = ret[1][0] * ret[2][2] - ret[2][0] * ret[1][2];
		auto det5 = ret[0][0] * ret[2][2] - ret[2][0] * ret[0][2];
		auto det6 = ret[0][0] * ret[1][2] - ret[0][1] * ret[0][2];

		auto det7 = ret[1][0] * ret[2][1] - ret[2][0] * ret[1][1];
		auto det8 = ret[0][0] * ret[2][1] - ret[2][0] * ret[0][1];
		auto det9 = ret[0][0] * ret[1][1] - ret[1][0] * ret[0][1];

		ret = mat3(det1, det2, det3, det4, det5, det6, det7, det8, det9);
		ret = ret * mat3(1, -1, 1, -1, 1, -1, 1, -1, 1); // Adjoint matrix
		ret = ret * (1 / det);

		return ret;
	}

	__host__ __device__ __forceinline__ void zero() {
		m[0][0] = 0.0; m[1][0] = 0.0; m[2][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 0.0; m[2][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 0.0;
	}

	__host__ __device__ __forceinline__ void identity() {
		m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0;
		m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0;
		m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0;
	}

	__host__ __device__ __forceinline__ void print() {
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				printf("%f ", m[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	__host__ __device__ __forceinline__ float3 transform_point(const float3& pt) const {
		float3 temp = *this * pt;
		return make_float3(temp.x, temp.y, temp.z);
	}
	__host__ __device__ __forceinline__ float3 transform_vector(const float3& pt) const {
		float3 temp = *this * pt;
		return make_float3(temp.x, temp.y, temp.z);
	}
	__host__ __device__ __forceinline__ mat3 rotate_zyx(const float3& angs) {
		float cx, sx, cy, sy, cz, sz;
		cx = (float)cos(angs.x);
		sx = (float)sin(angs.x);
		cy = (float)cos(angs.y);
		sy = (float)sin(angs.y);
		cz = (float)cos(angs.z);
		sz = (float)sin(angs.z);

		m[0][0] = cz * cy;
		m[1][0] = sz * cy;
		m[2][0] = -sy;

		m[0][1] = -sz * cx + cz * sy * sx;
		m[1][1] = cz * cx - sz * sy * sz;
		m[2][1] = -cy * sx;

		m[0][2] = -sz * sx + cz * sy * cx;
		m[1][2] = cz * sx + sz * sy * cx;
		m[2][2] = cy * cx;

		return *this;
	}

	__host__ __device__ __forceinline__ mat3 translate(const float3& val) {
		m[0][2] += val.x;
		m[1][2] += val.y;
		m[2][2] += val.z;

		return *this;
	}

	__host__ __device__ __forceinline__ mat3 scale(const float3& val) {
		m[0][0] *= val.x;
		m[1][1] *= val.y;
		m[2][2] *= val.z;

		return *this;
	}

	__host__ __device__ __forceinline__ mat3 toMatrix(double* arr) {
		m[0][0] = arr[0]; m[1][0] = arr[1]; m[2][0] = arr[2];
		m[0][1] = arr[3]; m[1][1] = arr[4]; m[2][1] = arr[5];
		m[0][2] = arr[6]; m[1][2] = arr[7]; m[2][2] = arr[8];

		return *this;
	}

	__host__ __device__ __forceinline__ mat3 toMatrix(float* arr) {
		m[0][0] = arr[0]; m[1][0] = arr[1]; m[2][0] = arr[2];
		m[0][1] = arr[3]; m[1][1] = arr[4]; m[2][1] = arr[5];
		m[0][2] = arr[6]; m[1][2] = arr[7]; m[2][2] = arr[8];

		return *this;
	}

	__host__ __device__ __forceinline__ mat3& operator*=(const float f) { return *this = *this * f; }
	__host__ __device__ __forceinline__ mat3& operator/=(const float f) { return *this = *this / f; }
	__host__ __device__ __forceinline__ mat3& operator+=(const mat3& m) { return *this = *this + m; }
	__host__ __device__ __forceinline__ mat3& operator-=(const mat3& m) { return *this = *this - m; }
	__host__ __device__ __forceinline__ mat3& operator*=(const mat3& m) { return *this = *this * m; }
};
#endif