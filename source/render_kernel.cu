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
//	Version 1.0: Sergen Eren, 26/3/2019
//
// File: Custom path trace kernel: 
//       Performs custom path tracing
//
//-----------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdio.h>
#include "cuda_math.cuh"
#include <float.h>
#include <cuda_runtime.h> 
#include <curand_kernel.h>


typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 

#include "render_kernel.h"

#define BLACK			make_float3(0.0f, 0.0f, 0.0f)
#define WHITE			make_float3(1.0f, 1.0f, 1.0f)
#define RED				make_float3(1.0f, 0.0f, 0.0f)
#define GREEN			make_float3(0.0f, 1.0f, 0.0f)
#define BLUE			make_float3(0.0f, 0.0f, 1.0f)
#define EPS				0.001f

#define INV_2_PI		1.0f / (2.0f * M_PI) 
#define INV_4_PI		1.0f / (4.0f * M_PI) 
#define INV_PI			1.0f / M_PI 

#include <curand_kernel.h>
typedef curandStatePhilox4_32_10_t Rand_state;
#define rand(state) curand_uniform(state)

// Helper functions

__device__ inline void coordinate_system(
	float3 v1,
	float3 &v2,
	float3 &v3)
{
	if (fabsf(v1.x) > fabsf(v1.y))	v2 = make_float3(-v1.z, 0.0f, v1.x);
	else							v2 = make_float3(0.0f, v1.z, -v1.y);
	v2 = normalize(v2);
	v3 = normalize(cross(v1, v2));

}

__device__ inline float3 spherical_direction(
	float sinTheta,
	float cosTheta,
	float phi,
	float3 x,
	float3 y,
	float3 z)
{

	return x * sinTheta * cosf(phi) + y * sinTheta * sinf(phi) + z * cosTheta;

}

__device__ inline bool solveQuadratic(
	float a,
	float b,
	float c,
	float& x1,
	float& x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return false;
		x1 = 0; x2 = sqrt(-c / a);
		return true;
	}

	float discr = b * b - 4 * a * c;

	if (discr < 0) return false;

	float q = (b < 0.f) ? -0.5f * (b - sqrt(discr)) : -0.5f * (b + sqrt(discr));
	x1 = q / a;
	x2 = c / q;

	return true;
}

__device__ bool raySphereIntersect(
	const float3& orig,
	const float3& dir,
	const float& radius,
	float& t0,
	float& t1)
{

	float A = squared_length(dir);
	float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
	float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

	if (!solveQuadratic(A, B, C, t0, t1)) return false;

	if (t0 > t1) {
		float tempt = t1;
		t1 = t0;
		t0 = tempt;
	}
	return true;
}

__device__ inline float degree_to_radians(
	float degree)
{

	return degree * M_PI / 180.0f;

}


__device__ inline float3 degree_to_cartesian(
	float azimuth,
	float elevation)
{

	float az = clamp(azimuth, .0f, 360.0f);
	float el = clamp(elevation, .0f, 90.0f);

	az = degree_to_radians(az);
	el = degree_to_radians(90.0f - el);

	float x = sinf(el) * cosf(az);
	float y = cosf(el);
	float z = sinf(el) * sinf(az);

	return normalize(make_float3(x, y, z));
}

__device__ inline float tex_lookup_1d(
	cudaTextureObject_t tex,
	float v)
{
	const float texval = tex1D<float>(tex, v);

	return texval;
}

__device__ inline float tex_lookup_2d(
	cudaTextureObject_t tex,
	float u,
	float v) {


	const float texval = tex2D<float>(tex, u, v);

	return texval;
}

__device__ inline float draw_sample_from_distribution(
	Kernel_params kernel_params,
	Rand_state rand_state,
	float3 &wo) {

	float xi = rand(&rand_state);
	float zeta = rand(&rand_state);

	float pdf = 1.0f;
	int v = 0;
	int res = kernel_params.env_sample_tex_res;

	// Find marginal row number

	// Find interval

	int first = 0, len = res;

	while (len > 0) {

		int half = len >> 1, middle = first + half;

		if (tex_lookup_1d(kernel_params.env_marginal_cdf_tex, middle) <= xi) {
			first = middle + 1;
			len -= half + 1;
		}
		else len = half;

	}
	v = clamp(first - 1, 0, res - 2);



	float dv = xi - tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v);
	float d_cdf_marginal = tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v + 1) - tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v);
	if (d_cdf_marginal > .0f) dv /= d_cdf_marginal;

	// Calculate marginal pdf
	float marginal_pdf = tex_lookup_1d(kernel_params.env_marginal_func_tex, v + dv) / kernel_params.env_marginal_int;

	// calculate Φ (elevation)
	float theta = ((float(v) + dv) / float(res)) * M_PI;

	// v is now our row number. find the conditional value and pdf from v

	int u;
	first = 0, len = res;
	while (len > 0) {

		int half = len >> 1, middle = first + half;

		if (tex_lookup_2d(kernel_params.env_cdf_tex, middle, v) <= zeta) {
			first = middle + 1;
			len -= half + 1;
		}
		else len = half;

	}
	u = clamp(first - 1, 0, res - 2);

	float du = zeta - tex_lookup_2d(kernel_params.env_cdf_tex, u, v);

	float d_cdf_conditional = tex_lookup_2d(kernel_params.env_cdf_tex, u + 1, v) - tex_lookup_2d(kernel_params.env_cdf_tex, u, v);
	if (d_cdf_conditional > 0) du /= d_cdf_conditional;

	//Calculate conditional pdf
	float conditional_pdf = tex_lookup_2d(kernel_params.env_func_tex, u + du, v) / tex_lookup_1d(kernel_params.env_marginal_func_tex, v);

	// Find the θ (azimuth)
	float phi = ((float(u) + du) / float(res)) * M_PI * 2.0f;



	float cos_theta = cosf(theta);
	float sin_theta = sinf(theta);
	float sin_phi = sinf(phi);
	float cos_phi = cosf(phi);

	float3 sundir = normalize(make_float3(sinf(kernel_params.azimuth) * cosf(kernel_params.elevation),
		sinf(kernel_params.azimuth) * sinf(kernel_params.elevation), cosf(kernel_params.azimuth)));

	wo = normalize(make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta));
	pdf = (marginal_pdf * conditional_pdf) / (2 * M_PI * M_PI * sin_theta);
	if (kernel_params.debug) printf("\n%f	%f	%f	%d	%d", ((float(u) + du) / float(res)), ((float(v) + dv) / float(res)), pdf, u, v);
	//if (kernel_params.debug) printf("\n%f	%f	%f	%f", wo.x, wo.y,wo.z, dot(wo, sundir));
	return pdf;
}

__device__ inline float draw_pdf_from_distribution(Kernel_params kernel_params, float2 point)
{
	int res = kernel_params.env_sample_tex_res;

	int iu = clamp(int(point.x * res), 0, res - 1);
	int iv = clamp(int(point.y * res), 0, res - 1);

	float conditional = tex_lookup_2d(kernel_params.env_func_tex, iu, iv);
	float marginal = tex_lookup_1d(kernel_params.env_marginal_func_tex, iv);

	return conditional / marginal;
}

__device__ inline float power_heuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f*f) / (f*f + g * g);
}

//Phase functions pdf 

__device__ inline float isotropic() {

	return INV_4_PI;

}

__device__ inline float henyey_greenstein(
	float cos_theta,
	float g)
{

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}

__device__ inline float double_henyey_greenstein(
	float cos_theta,
	float f,
	float g1,
	float g2)
{

	return f * henyey_greenstein(cos_theta, g1) + (1 - f) * henyey_greenstein(cos_theta, g2);

}


//Phase function direction samplers

__device__ inline float sample_spherical(
	Rand_state rand_state,
	float3 &wi)
{
	float phi = (float)(2.0f * M_PI) * rand(&rand_state);
	float cos_theta = 1.0f - 2.0f * rand(&rand_state);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	wi = make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);

	return isotropic();
}

__device__ inline float sample_hg(
	float3 &wo,
	Rand_state &randstate,
	float g)
{

	float cos_theta;

	if (fabsf(g) < EPS) cos_theta = 1 - 2 * rand(&randstate);
	else {
		float sqr_term = (1 - g * g) / (1 - g + 2 * g * rand(&randstate));
		cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g);
	}
	float sin_theta = sqrtf(fmaxf(.0f, 1.0f - cos_theta * cos_theta));
	float phi = (float)(2.0 * M_PI) * rand(&randstate);
	float3 v1, v2;
	coordinate_system(wo * -1.0f, v1, v2);
	wo = spherical_direction(sin_theta, cos_theta, phi, v1, v2, wo);
	return henyey_greenstein(-cos_theta, g);
}


__device__ inline float sample_double_hg(
	float3 &wi,
	Rand_state randstate,
	float f,
	float g1,
	float g2)
{
	wi *= -1.0f;
	float3 v1 = wi, v2 = wi;
	float cos_theta1, cos_theta2;


	if (f > 0.9999f) {

		cos_theta1 = sample_hg(v1, randstate, g1);
		wi = v1;
		return henyey_greenstein(cos_theta1, g1);
	}
	else if (f < EPS)
	{
		cos_theta2 = sample_hg(v2, randstate, g2);
		wi = v2;
		return henyey_greenstein(cos_theta2, g2);
	}
	else {

		cos_theta1 = sample_hg(v1, randstate, g1);
		cos_theta2 = sample_hg(v2, randstate, g2);

		wi = lerp(v1, v2, 1 - f);
		float cos_theta = lerp(cos_theta1, cos_theta2, 1 - f);
		return double_henyey_greenstein(cos_theta, f, g1, g2);
	}

}


// Volume accessors
__device__ inline bool in_volume_bbox(
	const VDBInfo gvdb,
	const float3 pos)
{

	return pos.x >= gvdb.bmin.x && pos.y >= gvdb.bmin.y && pos.z >= gvdb.bmin.z && pos.x < gvdb.bmax.x && pos.y < gvdb.bmax.y && pos.z < gvdb.bmax.z;
}

__device__ inline float get_extinction(
	const Kernel_params &kernel_params,
	VDBInfo *gvdb,
	const float3 &p)
{

	float density = 0.0f;

	//brick node variables 
	float3 vmin; //root pos of brick node
	uint64 nodeid; // brick id 
	float3 offset; // brick offset
	float3 vdel; // i.e. voxel size 

	VDBNode* brick_node = getNodeAtPoint(gvdb, p, &offset, &vmin, &vdel, &nodeid);

	if (brick_node != 0x0) {

		float3 brick_pos = (p - vmin) / vdel;
		float3 atlas_pos = make_float3(brick_node->mValue);
		density = tex3D<float>(gvdb->volIn[0], brick_pos.x + atlas_pos.x, brick_pos.y + atlas_pos.y, brick_pos.z + atlas_pos.z);
	}

	return density;
}


// Light Samplers 

__device__ inline float3 sample_atmosphere(
	const Kernel_params &kernel_params,
	const float3 orig,
	const float3 dir,
	const float3 intensity)
{

	// initial parameters
	float	atmosphereRadius = 6420e3f;
	float3	sunDirection = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);
	float	earthRadius = 6360e3f;
	float	Hr = 7994.0f;
	float	Hm = 1200.0f;
	float3	betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
	float3	betaM = make_float3(21e-6f);
	//


	float t0, t1;
	float tmin, tmax = FLT_MAX;
	float3 pos = orig;
	pos.y += 1000 + 6360e3f;

	if (raySphereIntersect(pos, dir, 6360e3f, t0, t1) && t1 > .0f) tmax = fmaxf(.0f, t0);
	tmin = .0f;
	if (!raySphereIntersect(pos, dir, atmosphereRadius, t0, t1) || t1 < 0) return make_float3(1.0f, .0f, .0f);
	if (t0 > tmin && t0 > 0) tmin = t0;
	if (t1 < tmax) tmax = t1;

	uint numSamples = 16;
	uint numSamplesLight = 8;

	float segmentLength = (tmax - tmin) / numSamples;
	float tCurrent = tmin;
	float3 sumR = make_float3(0.0f, .0f, .0f); // Rayleigh contribution
	float3 sumM = make_float3(0.0f, .0f, .0f); // Mie contribution

	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float g = 0.76f;

	float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

	for (uint i = 0; i < numSamples; ++i) {
		float3 samplePosition = pos + (tCurrent + segmentLength * 0.5f) * dir;
		float height = length(samplePosition) - earthRadius;
		// compute optical depth for light
		float hr = exp(-height / Hr) * segmentLength;
		float hm = exp(-height / Hm) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float t0Light, t1Light;
		raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
		float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
		float opticalDepthLightR = 0, opticalDepthLightM = 0;
		uint j;
		for (j = 0; j < numSamplesLight; ++j) {
			float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
			float heightLight = length(samplePositionLight) - earthRadius;
			if (heightLight < 0) break;
			opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
			tCurrentLight += segmentLengthLight;
		}
		if (j == numSamplesLight) {
			float3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
			float3 attenuation = make_float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
		tCurrent += segmentLength;
	}


	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * intensity;
}

__device__ inline float3 sample_env_tex(
	const Kernel_params kernel_params,
	const float3 wi)
{

	const float4 texval = tex2D<float4>(
		kernel_params.env_tex,
		atan2f(wi.z, wi.x) * (float)(0.5 / M_PI) + 0.5f,
		acosf(fmaxf(fminf(wi.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
	return make_float3(texval.x, texval.y, texval.z);


}

__device__ inline float get_density(
	const Kernel_params &kernel_params,
	VDBInfo *gvdb,
	const float3 &p)
{

	float density = 0.0f;

	//brick node variables 
	float3 vmin; //root pos of brick node
	uint64 nodeid; // brick id 
	float3 offset; // brick offset
	float3 vdel; // i.e. voxel size 

	VDBNode* brick_node = getNodeAtPoint(gvdb, p, &offset, &vmin, &vdel, &nodeid);

	if (brick_node != 0x0) {

		float3 brick_pos = (p - vmin) / vdel;
		float3 atlas_pos = make_float3(brick_node->mValue);
		density = tex3D<float>(gvdb->volIn[0], brick_pos.x + atlas_pos.x, brick_pos.y + atlas_pos.y, brick_pos.z + atlas_pos.z);
	}

	return density;

}

__device__ inline float3 Tr(
	Rand_state &rand_state,
	float3 pos,
	float3 dir,
	const Kernel_params &kernel_params,
	VDBInfo &gvdb)
{

	// Run ratio tracking to estimate transmittance

	float3 tr = WHITE;
	float3 p = pos;
	float t = 0.0f;
	float inv_max_density = 1 / kernel_params.max_extinction;

	//int k = 1;
	//kernel_params.debug_buffer[0] = WHITE;

	while (true) {
		if (tr.x < 0.0000001) break;
		t -= logf(1 - rand(&rand_state)) * inv_max_density * kernel_params.tr_depth / kernel_params.extinction.x;
		p += dir * t;
		if (!in_volume_bbox(gvdb, p)) break;
		float density = get_density(kernel_params, &gvdb, p);
		tr *= 1 - fmaxf(.0f, density*inv_max_density);
		//kernel_params.debug_buffer[k] = tr;
		//k++;
	}
	return tr;
}


__device__ inline float pdf_li(
	Kernel_params kernel_params,
	float3 wi)
{
	float theta = acosf(clamp(wi.y, -1.0f, 1.0f));
	float phi = atan2f(wi.z, wi.x);
	float sin_theta = sinf(theta);

	if (sin_theta == .0f) return .0f;
	float2 polar_pos = make_float2(phi * INV_2_PI, theta * INV_PI) / (2.0f * M_PI * M_PI * sin_theta);
	return draw_pdf_from_distribution(kernel_params, polar_pos);

}

__device__ inline float3 estimate_sky(
	Kernel_params kernel_params,
	Rand_state &randstate,
	const float3 &ray_pos,
	float3 &ray_dir,
	VDBInfo &gvdb)
{
	float3 Ld = BLACK;

	for (int i = 0; i < 1; i++) {

		float3 Li = BLACK;
		float3 wi;

		float light_pdf = .0f, phase_pdf = .0f;

		float az = rand(&randstate) * 360.0f;
		float el = rand(&randstate) * 180.0f;

		// Sample light source with multiple importance sampling 

		if (kernel_params.environment_type == 0) {
			light_pdf = draw_sample_from_distribution(kernel_params, randstate, wi);
			Li = sample_atmosphere(kernel_params, ray_pos, wi, kernel_params.sky_color);
		}
		else {
			light_pdf = sample_spherical(randstate, wi);
			Li = sample_env_tex(kernel_params, wi);
		}

		if (light_pdf > .0f && !isBlack(Li)) {

			float cos_theta = dot(ray_dir, wi);
			phase_pdf = henyey_greenstein(cos_theta, kernel_params.phase_g1);

			if (phase_pdf > .0f) {
				float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gvdb);
				Li *= tr;

				if (!isBlack(Li)) {

					float weight = power_heuristic(1, light_pdf, 1, phase_pdf);
					Ld += Li * phase_pdf * weight / light_pdf;
				}

			}

		}


		// Sample BSDF with multiple importance sampling 
		wi = ray_dir;
		phase_pdf = sample_hg(wi, randstate, kernel_params.phase_g1);
		float3 f = make_float3(phase_pdf);
		if (phase_pdf > .0f) {
			Li = BLACK;
			float weight = 1.0f;
			if (kernel_params.environment_type == 0)
			{
				light_pdf = pdf_li(kernel_params, wi);
			}
			else light_pdf = isotropic();

			if (light_pdf == 0.0f) return Ld;
			weight = power_heuristic(1, phase_pdf, 1, light_pdf);

			float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gvdb);

			if (kernel_params.environment_type == 0)
			{
				Li = sample_atmosphere(kernel_params, ray_pos, wi, kernel_params.sky_color);
			}
			else Li = sample_env_tex(kernel_params, wi);


			if (!isBlack(Li))
				Ld += Li * tr * weight;
		}


	}

	return Ld;

}


__device__ inline float3 estimate_sun(
	Kernel_params kernel_params,
	Rand_state &randstate,
	const float3 &ray_pos,
	float3 &ray_dir,
	VDBInfo &gvdb)
{
	float3 Ld = BLACK;
	float3 wi;
	float phase_pdf = .0f;

	// sample sun light with multiple importance sampling

	//Find sun direction 
	wi = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

	// find scattering pdf
	float cos_theta = dot(ray_dir, wi);
	phase_pdf = henyey_greenstein(cos_theta, kernel_params.phase_g1);

	// Check visibility of light source 
	float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gvdb);

	// Ld = Li * visibility.Tr * scattering_pdf / light_pdf  
	Ld = kernel_params.sun_color * tr  * phase_pdf;

	// No need for sampling BSDF with importance sampling
	// please see: http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Direct_Lighting.html#fragment-SampleBSDFwithmultipleimportancesampling-0

	return Ld;

}

__device__ inline float3 uniform_sample_one_light(
	Kernel_params kernel_params,
	const float3 &ray_pos,
	float3 &ray_dir,
	Rand_state &randstate,
	VDBInfo &gvdb)
{

	int nLights = 2; // number of lights
	bool light_num = rand(&randstate) < .5f;
	//bool light_num = 0; 

	float3 L = BLACK;

	if (light_num) {

		L += estimate_sun(kernel_params, randstate, ray_pos, ray_dir, gvdb) * kernel_params.sun_mult;
	}
	else {

		L += estimate_sky(kernel_params, randstate, ray_pos, ray_dir, gvdb) * kernel_params.sky_mult;
	}

	return L * (float)nLights;

}


__device__ inline float3 sample(
	Rand_state &rand_state,
	float3 &ray_pos,
	const float3 &ray_dir,
	bool &interaction,
	const Kernel_params &kernel_params,
	VDBInfo &gvdb)
{
	// Run delta tracking 

	float t = 0.0f;
	float inv_max_density = 1.0f / kernel_params.max_extinction;
	float inv_density_mult = 1.0f / kernel_params.density_mult;

	while (true) {

		t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult;
		ray_pos += ray_dir * t;
		if (!in_volume_bbox(gvdb, ray_pos)) break;
		float density = get_density(kernel_params, &gvdb, ray_pos);
		if (density * inv_max_density > rand(&rand_state)) {

			interaction = true;
			return kernel_params.albedo / kernel_params.extinction;
		}
	}
	return WHITE;

}


// PBRT Volume Integrator
__device__ inline float3 vol_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params kernel_params,
	VDBInfo gvdb)
{
	float3 L = BLACK;
	float3 beta = WHITE;
	float3 env_pos = ray_pos;
	float3 t = rayBoxIntersect(ray_pos, ray_dir, gvdb.bmin, gvdb.bmax);
	bool mi;

	if (t.z != NOHIT) { // found an intersection
		ray_pos += ray_dir * t.x;

		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, kernel_params, gvdb);
			if (isBlack(beta)) break;


			if (mi) { // medium interaction 

				L += beta * uniform_sample_one_light(kernel_params, ray_pos, ray_dir, rand_state, gvdb);
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
			}



		}


	}

	return L;

}


// From Ray Tracing Gems Vol-28
__device__ inline float3 direct_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params kernel_params,
	VDBInfo gvdb)
{
	float3 L = BLACK;
	float3 beta = WHITE;
	float3 env_pos = ray_pos;
	float3 t = rayBoxIntersect(ray_pos, ray_dir, gvdb.bmin, gvdb.bmax);
	bool mi = false;

	if (t.z != NOHIT) { // found an intersection
		ray_pos += ray_dir * t.x;

		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, kernel_params, gvdb);
			if (isBlack(beta)) break;


			if (mi) { // medium interaction 
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
				if (kernel_params.sun_mult > .0f) L += estimate_sun(kernel_params, rand_state, ray_pos, ray_dir, gvdb) * beta * kernel_params.sun_mult;
			}

		}

	}

	//Sample environment

	if (kernel_params.environment_type == 0) {

		if (mi) L += estimate_sky(kernel_params, rand_state, ray_pos, ray_dir, gvdb) * beta;
		else L += sample_atmosphere(kernel_params, env_pos, ray_dir, kernel_params.sky_color) * beta;
	}
	else {
		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));

		L += make_float3(texval.x, texval.y, texval.z) * kernel_params.sky_color * beta * isotropic();
	}

	return L;

}





// From Art-Directable Multiple Volumetric Scattering Wrenninge - 2015

__device__ inline bool density_sample(
	Rand_state &rand_state,
	float3 &ray_pos,
	const float3 &ray_dir,
	bool &interaction,
	const Kernel_params &kernel_params,
	VDBInfo &gvdb)
{

	float t = 0.0f;
	float inv_max_density = 1.0f / kernel_params.max_extinction;
	float inv_density_mult = 1.0f / kernel_params.density_mult;

	while (true) {

		t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult;
		ray_pos += ray_dir * t;
		if (!in_volume_bbox(gvdb, ray_pos)) return false;
		float density = get_density(kernel_params, &gvdb, ray_pos);
		if (density * inv_max_density > rand(&rand_state)) {
			return true;
		}
	}
	return true;

}

__device__ inline bool track_secondary(
	Rand_state &rand_state,
	float3 &ray_pos,
	const float3 ray_dir,
	bool &interaction,
	const Kernel_params &kernel_params,
	VDBInfo &gvdb)
{
	float t = 0.0f;
	float inv_max_density = 1.0f / kernel_params.max_extinction;
	float inv_density_mult = 1.0f / kernel_params.density_mult;

	while (true) {

		t -= logf(1 - rand(&rand_state)) * inv_density_mult;
		ray_pos += ray_dir * t;
		if (!in_volume_bbox(gvdb, ray_pos)) return false;
		float density = get_density(kernel_params, &gvdb, ray_pos);
		if (density * inv_max_density > rand(&rand_state)) {
			return true;
		}
	}
	return true;

}

__device__ inline float3 art_directable_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params kernel_params,
	VDBInfo gvdb)
{
	float3 L = BLACK;
	float3 beta = WHITE;
	float3 env_pos = ray_pos;
	float3 t = rayBoxIntersect(ray_pos, ray_dir, gvdb.bmin, gvdb.bmax);
	bool mi = false;

	float a = 0.5f, b= 0.5f, c = 0.5f;
	int N = 8;

	if (t.z != NOHIT) { // found an intersection
		ray_pos += ray_dir * t.x;

		// Find the points inside volume based on density tracking
		while (density_sample(rand_state, ray_pos, ray_dir, mi, kernel_params, gvdb)) {

			float3 sec_pos = ray_pos;
			float3 sec_dir = ray_dir;

			for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {

				track_secondary(rand_state, sec_pos, sec_dir, mi, kernel_params, gvdb);
				
				float phase = sample_hg(sec_dir, rand_state, powf(c, depth)*kernel_params.phase_g1);
				float3 tr = Tr(rand_state, sec_pos, sec_dir, kernel_params, gvdb) / powf(a, depth);
				
				L += kernel_params.albedo * powf(b, depth) * estimate_sun(kernel_params, rand_state, sec_pos, sec_dir, gvdb) * phase * tr;
			
			}
					   			 		  
		}

	}

	return L;

}




// Main cuda kernels accessor 


extern "C" __global__ void volume_rt_kernel(
	VDBInfo gvdb,
	const Kernel_params kernel_params) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= kernel_params.resolution.x || y >= kernel_params.resolution.y)
		return;


	// Initialize pseudorandom number generator (PRNG); assume we need no more than 4096 random numbers.
	const unsigned int idx = y * scn.width + x;
	Rand_state rand_state;
	curand_init(idx, 0, kernel_params.iteration * 4096, &rand_state);

	float3 ray_dir = normalize(getViewRay((float(scn.width - x) + .5f) / scn.width, (float(scn.height - y) + .5f) / scn.height));

	float3 value = WHITE;

	if (kernel_params.iteration < kernel_params.max_interactions && kernel_params.render)
	{
		value = direct_integrator(rand_state, scn.campos, ray_dir, kernel_params, gvdb);

	}

	// Accumulate.
	if (kernel_params.iteration == 0)
		kernel_params.accum_buffer[idx] = value;
	else if (kernel_params.iteration < kernel_params.max_interactions) {
		kernel_params.accum_buffer[idx] = kernel_params.accum_buffer[idx] +
			(value - kernel_params.accum_buffer[idx]) / (float)(kernel_params.iteration + 1);
	}

	// Update display buffer (simple Reinhard tonemapper + gamma).

	float3 val = kernel_params.accum_buffer[idx] * kernel_params.exposure_scale;

	val.x *= (1.0f + val.x * 0.1f) / (1.0f + val.x);
	val.y *= (1.0f + val.y * 0.1f) / (1.0f + val.y);
	val.z *= (1.0f + val.z * 0.1f) / (1.0f + val.z);
	const unsigned int r = (unsigned int)(255.0f * fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int g = (unsigned int)(255.0f * fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int b = (unsigned int)(255.0f * fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	kernel_params.display_buffer[idx] = 0xff000000 | (r << 16) | (g << 8) | b;

}