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
//	Version 1.0.1: Sergen Eren, 30/10/2019
//
// File: Custom path trace kernel: 
//       Performs custom path tracing
//
//-----------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdio.h>
#include <float.h>

// Cuda includes
#include <cuda_runtime.h> 
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "helper_math.h"

typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

// Internal includes
#include "kernel_params.h"
#include "atmosphere/definitions.h"
#include "atmosphere/constants.h"
#include "gpu_vdb.h"
#include "camera.h"
#include "light.h"
#include "bvh/bvh.h"
#include "plane.h"

#define BLACK			make_float3(0.0f, 0.0f, 0.0f)
#define WHITE			make_float3(1.0f, 1.0f, 1.0f)
#define RED				make_float3(1.0f, 0.0f, 0.0f)
#define GREEN			make_float3(0.0f, 1.0f, 0.0f)
#define BLUE			make_float3(0.0f, 0.0f, 1.0f)
#define EPS				0.001f

#define INV_2_PI		1.0f / (2.0f * M_PI) 
#define INV_4_PI		1.0f / (4.0f * M_PI) 
#define INV_PI			1.0f / M_PI 


// Helper functions

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
	float el = clamp(elevation, -90.0f, 90.0f);

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


// Environment light samplers

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
	//if (kernel_params.debug) printf("\n%f	%f	%f	%d	%d", ((float(u) + du) / float(res)), ((float(v) + dv) / float(res)), pdf, u, v);
	//if (kernel_params.debug) printf("\n%f	%f	%f	%f", wo.x, wo.y,wo.z, dot(wo, sundir));
	return pdf;
}



//Phase functions pdf 
__device__ inline float draw_pdf_from_distribution(Kernel_params kernel_params, float2 point)
{
	int res = kernel_params.env_sample_tex_res;

	int iu = clamp(int(point.x * res), 0, res - 1);
	int iv = clamp(int(point.y * res), 0, res - 1);

	float conditional = tex_lookup_2d(kernel_params.env_func_tex, iu, iv);
	float marginal = tex_lookup_1d(kernel_params.env_marginal_func_tex, iv);

	return conditional / marginal;
}

__device__ inline float isotropic() {

	return INV_4_PI;

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


// Atmosphere Functions

//#define COMBINED_SCATTERING_TEXTURES

__device__  float ClampCosine(float mu)
{
	return clamp(mu, float(-1.0), float(1.0));
}

__device__  float ClampDistance(float d)
{
	return fmaxf(d, 0.0 * m);
}

__device__  float ClampRadius(const AtmosphereParameters atmosphere, float r)
{
	return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

__device__  float SafeSqrt(float a)
{
	return sqrtf(fmaxf(a, 0.0 * m2()));
}

__device__  float DistanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu)
{
	float discriminant = r * r * (mu * mu - 1.0) + atmosphere.top_radius * atmosphere.top_radius;
	return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

__device__  float DistanceToBottomAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu)
{
	float discriminant = r * r * (mu * mu - 1.0) + atmosphere.bottom_radius * atmosphere.bottom_radius;
	return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

__device__  bool RayIntersectsGround(const AtmosphereParameters atmosphere, float r, float mu)
{
	return mu < 0.0 && r * r * (mu * mu - 1.0) +
		atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2();
}

__device__  float GetLayerDensity(const DensityProfileLayer layer, float altitude)
{
	float density = layer.exp_term * exp(layer.exp_scale * altitude) +
		layer.linear_term * altitude + layer.const_term;
	return clamp(density, float(0.0), float(1.0));
}

__device__  float GetProfileDensity(const DensityProfile profile, float altitude)
{
	return altitude < profile.layers[0].width ? GetLayerDensity(profile.layers[0], altitude) : GetLayerDensity(profile.layers[1], altitude);
}

__device__  float GetTextureCoordFromUnitRange(float x, int texture_size)
{
	return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

__device__  float GetUnitRangeFromTextureCoord(float u, int texture_size)
{
	return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

__device__  float2 GetTransmittanceTextureUvFromRMu(const AtmosphereParameters atmosphere, float r, float mu)
{
	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	float H = sqrtf(atmosphere.top_radius * atmosphere.top_radius - atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
	// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
	float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
	float d_min = atmosphere.top_radius - r;
	float d_max = rho + H;
	float x_mu = (d - d_min) / (d_max - d_min);
	float x_r = rho / H;
	return make_float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH), GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

__device__  void GetRMuFromTransmittanceTextureUv(const AtmosphereParameters atmosphere, float2 uv, float &r, float &mu)
{
	float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
	float x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	float H = sqrt(atmosphere.top_radius * atmosphere.top_radius - atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon, from which we can compute r:
	float rho = H * x_r;
	r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
	// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
	// from which we can recover mu:
	float d_min = atmosphere.top_radius - r;
	float d_max = rho + H;
	float d = d_min + x_mu * (d_max - d_min);
	mu = d == 0.0 * m ? float(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
	mu = ClampCosine(mu);
}

__device__  float3 GetTransmittanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu)
{

	float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
	const float3 texval = make_float3(tex2D<float4>(atmosphere.transmittance_texture, uv.x, uv.y));
	return texval;
}

__device__  float3 GetTransmittance(const AtmosphereParameters atmosphere, float r, float mu, float d, bool ray_r_mu_intersects_ground)
{

	float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	float mu_d = ClampCosine((r * mu + d) / r_d);

	if (ray_r_mu_intersects_ground) {
		return fminf(GetTransmittanceToTopAtmosphereBoundary(atmosphere, r_d, -mu_d) / GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, -mu), make_float3(1.0f));
	}
	else {
		return fminf(GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu) / GetTransmittanceToTopAtmosphereBoundary(atmosphere, r_d, mu_d), make_float3(1.0));
	}
}

__device__  float3 GetTransmittanceToSun(const AtmosphereParameters atmosphere, float r, float mu_s)
{

	float sin_theta_h = atmosphere.bottom_radius / r;
	float cos_theta_h = -sqrtf(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
	return GetTransmittanceToTopAtmosphereBoundary(
		atmosphere, r, mu_s) *
		smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
			sin_theta_h * atmosphere.sun_angular_radius / rad,
			mu_s - cos_theta_h);
}

__device__  float DistanceToNearestAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu, bool ray_r_mu_intersects_ground)
{
	if (ray_r_mu_intersects_ground) {
		return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
	}
	else {
		return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
	}
}

__device__  float RayleighPhaseFunction(float nu)
{
	float k = 3.0 / (16.0 * PI * sr);
	return k * (1.0 + nu * nu);
}

__device__  float MiePhaseFunction(float g, float nu)
{
	float k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

__device__  float4 GetScatteringTextureUvwzFromRMuMuSNu(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground)
{

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	float rho =
		SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
	float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

	// Discriminant of the quadratic equation for the intersections of the ray
	// (r,mu) with the ground (see RayIntersectsGround).
	float r_mu = r * mu;
	float discriminant =
		r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
	float u_mu;
	if (ray_r_mu_intersects_ground) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon).
		float d = -r_mu - SafeSqrt(discriminant);
		float d_min = r - atmosphere.bottom_radius;
		float d_max = rho;
		u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon).
		float d = -r_mu + SafeSqrt(discriminant + H * H);
		float d_min = atmosphere.top_radius - r;
		float d_max = rho + H;
		u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}

	float d = DistanceToTopAtmosphereBoundary(
		atmosphere, atmosphere.bottom_radius, mu_s);
	float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	float d_max = H;
	float a = (d - d_min) / (d_max - d_min);
	float A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	float u_mu_s = GetTextureCoordFromUnitRange(
		max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

	float u_nu = (nu + 1.0) / 2.0;
	return make_float4(u_nu, u_mu_s, u_mu, u_r);
}

__device__  void GetRMuMuSNuFromScatteringTextureUvwz(const AtmosphereParameters atmosphere, float4 uvwz, float &r, float &mu, float &mu_s, float &nu, bool &ray_r_mu_intersects_ground)
{

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	float rho =
		H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
	r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

	if (uvwz.z < 0.5) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
		// we can recover mu:
		float d_min = r - atmosphere.bottom_radius;
		float d_max = rho;
		float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? float(-1.0) :
			ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = true;
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon) - from which we can recover mu:
		float d_min = atmosphere.top_radius - r;
		float d_max = rho + H;
		float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? float(1.0) :
			ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = false;
	}

	float x_mu_s =
		GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
	float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	float d_max = H;
	float A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
	float d = d_min + min(a, A) * (d_max - d_min);
	mu_s = d == 0.0 * m ? float(1.0) :
		ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

	nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

__device__  void GetRMuMuSNuFromScatteringTextureFragCoord(const AtmosphereParameters atmosphere, float3 frag_coord, float& r, float& mu, float& mu_s, float& nu, bool& ray_r_mu_intersects_ground) {
	const float4 SCATTERING_TEXTURE_SIZE = make_float4(SCATTERING_TEXTURE_NU_SIZE - 1, SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
	float frag_coord_nu = floor(frag_coord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
	float frag_coord_mu_s = fmodf(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
	float4 uvwz = make_float4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) / SCATTERING_TEXTURE_SIZE;
	GetRMuMuSNuFromScatteringTextureUvwz(atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	// Clamp nu to its valid range of values, given mu and mu_s.
	nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)), mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

__device__  float3 GetIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s);

__device__  float2 GetIrradianceTextureUvFromRMuS(const AtmosphereParameters atmosphere, float r, float mu_s) {
	float x_r = (r - atmosphere.bottom_radius) /
		(atmosphere.top_radius - atmosphere.bottom_radius);
	float x_mu_s = mu_s * 0.5 + 0.5;
	return make_float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
		GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

__device__  void GetRMuSFromIrradianceTextureUv(const AtmosphereParameters atmosphere, float2 uv, float& r, float& mu_s) {
	float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
	float x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);

	r = atmosphere.bottom_radius + x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
	mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

__device__  float3 GetIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s)
{
	float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
	const float3 val = make_float3(tex2D<float4>(atmosphere.irradiance_texture, uv.x, uv.y));
	return val;
}


// Rendering kernels 

#ifdef COMBINED_SCATTERING_TEXTURES
__device__  float3 GetExtrapolatedSingleMieScattering(const AtmosphereParameters atmosphere, const float4 scattering)
{
	if (scattering.x == 0.0) {
		return make_float3(0.0);
	}

	return make_float3(scattering.x, scattering.y, scattering.z) * scattering.w / scattering.x *
		(atmosphere.rayleigh_scattering.x / atmosphere.mie_scattering.x) *
		(atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}
#endif

__device__  float3 GetCombinedScattering(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, float3& single_mie_scattering)
{
	float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
	float tex_x = floorf(tex_coord_x);
	float lerp = tex_coord_x - tex_x;
	float3 uvw0 = make_float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
	float3 uvw1 = make_float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);


#ifdef COMBINED_SCATTERING_TEXTURES
	float4 combined_scattering = tex3D<float4>(atmosphere.scattering_texture, uvw0.x, uvw0.y, uvw0.z) * (1.0 - lerp) + tex3D<float4>(atmosphere.scattering_texture, uvw1.x, uvw1.y, uvw1.z) * lerp;
	float3 scattering = make_float3(combined_scattering);
	single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
#else
	float3 scattering = make_float3(tex3D<float4>(atmosphere.scattering_texture, uvw0.x, uvw0.y, uvw0.z) * (1.0 - lerp) + tex3D<float4>(atmosphere.scattering_texture, uvw1.x, uvw1.y, uvw1.z) * lerp);
	single_mie_scattering = make_float3(tex3D<float4>(atmosphere.single_mie_scattering_texture, uvw0.x, uvw0.y, uvw0.z) * (1.0 - lerp) + tex3D<float4>(atmosphere.single_mie_scattering_texture, uvw1.x, uvw1.y, uvw1.z) * lerp);
#endif

	return scattering;
}

__device__  float3 GetSkyRadiance(const AtmosphereParameters atmosphere, float3 camera, float3 view_ray, float shadow_length, float3 sun_direction, float3& transmittance)
{
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	float r = length(camera);
	float rmu = dot(camera, view_ray);
	float distance_to_top_atmosphere_boundary = -rmu -
		sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}
	else if (r > atmosphere.top_radius) {
		// If the view ray does not intersect the atmosphere, simply return 0.
		transmittance = make_float3(1.0f);
		return make_float3(0.0f * watt_per_square_meter_per_sr_per_nm());
	}
	// Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
	float mu = rmu / r;
	float mu_s = dot(camera, sun_direction) / r;
	float nu = dot(view_ray, sun_direction);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = ray_r_mu_intersects_ground ? make_float3(0.0f) : GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
	float3 single_mie_scattering;
	float3 scattering;
	if (shadow_length == 0.0 * m) {
		scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);
	}
	else {
		// Case of light shafts (shadow_length is the total float noted l in our
		// paper): we omit the scattering between the camera and the point at
		// distance l, by implementing Eq. (18) of the paper (shadow_transmittance
		// is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
		float d = shadow_length;
		float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
		float mu_p = (r * mu + d) / r_p;
		float mu_s_p = (r * mu_s + d * nu) / r_p;

		scattering = GetCombinedScattering(atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering);
		float3 shadow_transmittance = GetTransmittance(atmosphere, r, mu, shadow_length, ray_r_mu_intersects_ground);
		scattering = scattering * shadow_transmittance;
		single_mie_scattering = single_mie_scattering * shadow_transmittance;
	}

	float3 sky_radiance = scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);

	if (atmosphere.use_luminance != 0) sky_radiance *= atmosphere.sky_spectral_radiance_to_luminance;
	return sky_radiance;
}

__device__  float3 GetSkyRadianceToPoint(const AtmosphereParameters atmosphere, float3 camera, float3 point, float shadow_length, float3 sun_direction, float3& transmittance)
{
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	float3 view_ray = normalize(point - camera);
	float r = length(camera);
	float rmu = dot(camera, view_ray);
	float distance_to_top_atmosphere_boundary = -rmu - sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}

	// Compute the r, mu, mu_s and nu parameters for the first texture lookup.
	float mu = rmu / r;
	float mu_s = dot(camera, sun_direction) / r;
	float nu = dot(view_ray, sun_direction);
	float d = length(point - camera);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = GetTransmittance(atmosphere, r, mu, d, ray_r_mu_intersects_ground);

	float3 single_mie_scattering;
	float3 scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);

	// Compute the r, mu, mu_s and nu parameters for the second texture lookup.
	// If shadow_length is not 0 (case of light shafts), we want to ignore the
	// scattering along the last shadow_length meters of the view ray, which we
	// do by subtracting shadow_length from d (this way scattering_p is equal to
	// the S|x_s=x_0-lv term in Eq. (17) of our paper).
	d = max(d - shadow_length, 0.0 * m);
	float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	float mu_p = (r * mu + d) / r_p;
	float mu_s_p = (r * mu_s + d * nu) / r_p;

	float3 single_mie_scattering_p;
	float3 scattering_p = GetCombinedScattering(atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering_p);

	// Combine the lookup results to get the scattering between camera and point.
	float3 shadow_transmittance = transmittance;
	if (shadow_length > 0.0 * m) {
		// This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
		shadow_transmittance = GetTransmittance(atmosphere, r, mu, d, ray_r_mu_intersects_ground);
	}
	scattering = scattering - shadow_transmittance * scattering_p;
	single_mie_scattering =
		single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
	single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, make_float4(scattering, single_mie_scattering.x));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering * smoothstep(float(0.0), float(0.01), mu_s);

	float3 sky_radiance = scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
	if (atmosphere.use_luminance != 0) sky_radiance *= atmosphere.sky_spectral_radiance_to_luminance;
	return sky_radiance;
}

__device__  float3 GetSunAndSkyIrradiance(const AtmosphereParameters atmosphere, float3 point, float3 normal, float3 sun_direction, float3 &sky_irradiance)
{
	float r = length(point);
	float mu_s = dot(point, sun_direction) / r;

	// Indirect irradiance (approximated if the surface is not horizontal).
	sky_irradiance = GetIrradiance(atmosphere, r, mu_s) * (1.0 + dot(normal, point) / r) * 0.5;
	float3 sun_irradiance = atmosphere.solar_irradiance * GetTransmittanceToSun(atmosphere, r, mu_s) * max(dot(normal, sun_direction), 0.0);

	if (atmosphere.use_luminance != 0) {

		sky_irradiance *= atmosphere.sky_spectral_radiance_to_luminance;
		sun_irradiance *= atmosphere.sun_spectral_radiance_to_luminance;
	}
	// Direct irradiance.
	return sun_irradiance;
}

__device__ float3 GetSolarRadiance(const AtmosphereParameters atmosphere) {

	float3 solar_radiance = atmosphere.solar_irradiance / (M_PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius);
	if (atmosphere.use_luminance != 0) solar_radiance *= atmosphere.sun_spectral_radiance_to_luminance;
	return solar_radiance;
}

// Light Samplers 

__device__ inline float3 sample_atmosphere(
	const Kernel_params &kernel_params,
	const AtmosphereParameters &atmosphere,
	const float3 ray_pos, const float3 ray_dir)
{

	float3 earth_center = make_float3(.0f, -atmosphere.bottom_radius, .0f);
	float3 sun_direction = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

	float3 p = ray_pos - earth_center;
	float p_dot_v = dot(p, ray_dir);
	float p_dot_p = dot(p, p);
	float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
	float distance_to_intersection = -p_dot_v - sqrt(earth_center.y * earth_center.y - ray_earth_center_squared_distance);

	float ground_alpha = 0.0;
	float3 ground_radiance = make_float3(0.0);

	if (distance_to_intersection > 0.0) {
		float3 point = ray_pos + ray_dir * distance_to_intersection;
		float3 normal = normalize(point - earth_center);

		// Compute the radiance reflected by the ground.
		float3 sky_irradiance;
		float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, point - earth_center, normal, sun_direction, sky_irradiance);
		ground_radiance = atmosphere.ground_albedo * (1.0 / M_PI) * (sun_irradiance + sky_irradiance);

		float3 transmittance;
		float3 in_scatter = GetSkyRadianceToPoint(atmosphere, ray_pos - earth_center, point - earth_center, .0f, sun_direction, transmittance);
		ground_radiance = ground_radiance * transmittance + in_scatter;
		ground_alpha = 1.0;
	}

	float3 transmittance_sky;
	float3 radiance_sky = GetSkyRadiance(atmosphere, ray_pos - earth_center, ray_dir, .0f, sun_direction, transmittance_sky);

	float2 sun_size = make_float2(tanf(atmosphere.sun_angular_radius), cosf(atmosphere.sun_angular_radius));

	if (dot(ray_dir, sun_direction) > sun_size.y) {
		radiance_sky = radiance_sky + transmittance_sky * GetSolarRadiance(atmosphere);
	}

	ground_radiance = lerp(radiance_sky, ground_radiance, ground_alpha);

	float3 exposure = atmosphere.use_luminance == 0 ? make_float3(atmosphere.exposure) : make_float3(atmosphere.exposure) * 1e-5;

	ground_radiance = powf(make_float3(1.0f) - expf(-ground_radiance / atmosphere.white_point * exposure), make_float3(1.0 / 2.2));

	return ground_radiance;

	/* old sky
	float azimuth = atan2f(-ray_dir.z, -ray_dir.x) * INV_2_PI + 0.5f;
	float elevation = acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * INV_PI;
	const float4 texval = tex2D<float4>( kernel_params.sky_tex, azimuth, elevation);
	return make_float3(texval.x, texval.y, texval.z);
	*/
}

__device__ inline float3 sample_env_tex(
	const Kernel_params kernel_params,
	const float3 wi)
{

	const float4 texval = tex2D<float4>(
		kernel_params.env_tex,
		atan2f(wi.z, wi.x) * (float)(0.5 / M_PI) + 0.5f,
		acosf(fmaxf(fminf(wi.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
	return make_float3(texval);
}


__device__ __inline__ float get_density(float3 pos, const GPU_VDB &gpu_vdb) {

	// world space to object space
	pos = gpu_vdb.get_xform().transpose().inverse().transform_point(pos);

	// object space position to index position
	pos -= gpu_vdb.vdb_info.bmin;

	// index position to [0-1] position
	pos.x /= float(gpu_vdb.vdb_info.dim.x);
	pos.y /= float(gpu_vdb.vdb_info.dim.y);
	pos.z /= float(gpu_vdb.vdb_info.dim.z);

	float density = tex3D<float>(gpu_vdb.vdb_info.density_texture, pos.x, pos.y, pos.z);
	return density;
}

__device__ inline float sum_density(float3 ray_pos, OCTNode *leaf_node, const GPU_VDB *volumes) {

	float density = 0.0f;

	for (int i = 0; i < leaf_node->num_volumes; ++i) {

		density += get_density(ray_pos, volumes[leaf_node->vol_indices[i]]);

	}

	return density;
}

__device__ inline bool traverse_octree(float3 ray_pos, float3 ray_dir, OCTNode *root, float &t_min, float&t_max) {

	/*
	// Recursive traversal (This doesn't work)
	if (root->depth == 1) {
		if (root->bbox.Intersect_no_t(ray_pos, ray_dir)) {
			if (root->num_volumes > 0) {
				return true;
			}
		}
	}
	else {
		for (int i = 0; i < 8; ++i) {
			traverse_octree(ray_pos, ray_dir, root->children[i]);
		}
	}

	return false;*/

	// Serial traversal
	if (root->bbox.Intersect(ray_pos, ray_dir, t_min, t_max)) {
		for (int i = 0; i < 8; ++i) {
			float temp_min;
			if (root->children[i]->bbox.Intersect(ray_pos, ray_dir, temp_min, t_max)) {
				if (root->children[i]->num_volumes > 0) {
					for (int x = 0; x < 8; ++x) {
						if (root->children[i]->children[x]->bbox.Intersect(ray_pos, ray_dir, temp_min, t_max)) {
							if (root->children[i]->children[x]->num_volumes > 0) {
								for (int y = 0; y < 8; ++y) {
									if (root->children[i]->children[x]->children[y]->bbox.Intersect(ray_pos, ray_dir, temp_min, t_max)) {
										if (root->children[i]->children[x]->children[y]->num_volumes > 0) {
											t_min = fminf(t_min, temp_min);
											return true;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return false;

}

__device__ inline OCTNode* get_closest_leaf_node(float3 ray_pos, float3 ray_dir, OCTNode *root, float &t_min, float &t_max) {

	OCTNode *node = NULL;

	// Lets first check if we intersect root
	if (root->bbox.Intersect(ray_pos, ray_dir, t_min, t_max)) {
		// intersected root. now check if we intersect depth 3 nodes 
		for (int i = 0; i < 8; ++i) {
			if (root->children[i]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max)) {
				if (root->children[i]->num_volumes > 0) {
					for (int x = 0; x < 8; ++x) {
						if (root->children[i]->children[x]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max)) {
							if (root->children[i]->children[x]->num_volumes > 0) {
								for (int y = 0; y < 8; ++y) {
									float temp_min = FLT_MAX;
									if (root->children[i]->children[x]->children[y]->bbox.Intersect(ray_pos, ray_dir, temp_min, t_max)) {
										if (root->children[i]->children[x]->children[y]->num_volumes > 0) {
											if (t_min < temp_min) {

												t_min = fminf(t_min, temp_min);
												node = root->children[i]->children[x]->children[y];

											}

										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return node;
}

__device__ inline int get_quadrant(OCTNode *root, float3 pos) {

	int child_idx = -1;

	for (int i = 0; i < 8; ++i) {

		if (Contains(root->children[i]->bbox, pos)) child_idx = i;

	}
	return child_idx;
}

__device__ inline float3 Tr(
	Rand_state &rand_state,
	float3 ray_pos,
	float3 ray_dir,
	const Kernel_params &kernel_params,
	const GPU_VDB *volumes,
	OCTNode *root)
{

	// Run ratio tracking to estimate transmittance


	float3 tr = WHITE;
	float3 p = ray_pos;
	float t_min, t_max, t = 0.0f;

	// Code path 1:
	// This is the old transmittance estimate algorithm that is agnostic of octree structure 
#if 1
	
	float inv_max_density = 1 / volumes[0].vdb_info.max_density;

	while (true) {
		t -= logf(1 - rand(&rand_state)) * inv_max_density * kernel_params.tr_depth / kernel_params.extinction.x;
		p += ray_dir * t;
		if (!Contains(root->bbox, p))	break;
		float density = sum_density(p, root, volumes);
		tr *= 1 - fmaxf(.0f, density*inv_max_density);
		tr = fmaxf(tr, make_float3(.0f));
	}
#endif

	// Code path 2:
	// This is a DDA stepping algorithm that checks the quadrant in Octree nodes and skips them if they contain no volumes 

#if 0  
	stop1:
	while (Contains(root->bbox, p)) {
		if (length(tr) < EPS) break;
		// First we figure out which quadrant on depth 3 our point sits in
		
		int depth3_node = get_quadrant(root, p);
		int depth2_node, leaf_node;
		if (depth3_node > -1) {
			if (!root->children[depth3_node]->num_volumes > 0) { //We are in the depth3 node but it is empty
				root->children[depth3_node]->bbox.Intersect(p, ray_dir, t_min, t_max);
				p += ray_dir * (t_max + EPS);
				goto stop1;
			}
			else { // We are in depth3 node and it's not empty
				stop2:
				while (Contains(root->children[depth3_node]->bbox, p)) {
					depth2_node = get_quadrant(root->children[depth3_node], p);
					if (depth2_node > -1) {
						if (!root->children[depth3_node]->children[depth2_node]->num_volumes > 0) { //We are in the depth2 node but it is empty
							root->children[depth3_node]->children[depth2_node]->bbox.Intersect(p, ray_dir, t_min, t_max);
							p += ray_dir * (t_max + EPS);
							goto stop1;
						}
						else {// We are in depth2 node and it's not empty
							while (Contains(root->children[depth3_node]->children[depth2_node]->bbox, p)) {

								leaf_node = get_quadrant(root->children[depth3_node]->children[depth2_node], p);
								if (leaf_node > -1) {
									if (!root->children[depth3_node]->children[depth2_node]->children[leaf_node]->num_volumes > 0) { //We are in the leaf node but it is empty
										root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox.Intersect(p, ray_dir, t_min, t_max);
										p += ray_dir * (t_max + EPS);
										goto stop2;
									}
									else {//We are in the leaf node and it is not empty. We can finally do the sampling

										float inv_max_density = 1.0f / root->children[depth3_node]->children[depth2_node]->children[leaf_node]->max_extinction;

										while (Contains(root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox, p)) {

											float voxel_size = root->children[depth3_node]->children[depth2_node]->children[leaf_node]->voxel_size;

											t -= logf(1 - rand(&rand_state)) * inv_max_density * kernel_params.tr_depth;
											p += ray_dir * t;
											float density = sum_density(p, root->children[depth3_node]->children[depth2_node]->children[leaf_node], volumes);
											tr *= 1 - fmaxf(.0f, density*inv_max_density);
											tr = fmaxf(tr, make_float3(.0f));
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
#endif

	return tr;
}


// Check if point projects onto top sky or ground and return point 
__device__ inline float3 project_to_earth(float3 point, float3 dir, AtmosphereParameters atmosphere) {

	float t0, t1;

	if (raySphereIntersect(point + atmosphere.bottom_radius, dir, atmosphere.bottom_radius, t0, t1)) {
		point += (dir * t0);
		return point;
	}
	if (raySphereIntersect(point + atmosphere.bottom_radius , dir, atmosphere.top_radius, t0, t1)) point += (dir * t0);

	return point;

}


//determines if ray intersects volume pyramid projected by octree bbox and returns the calculated transmission
__device__ inline bool get_shadow_box(float3 ray_pos, float3 ray_dir, OCTNode *root, AtmosphereParameters atmosphere, Kernel_params kernel_params, float &tr) {

	// construct planes created by edges of bbox projected on earth (or top atmosphere if no ground intersection)
	// we will need 12 planes for 12 edges of bbox 

	float3 p0;
	float3 p1;
	float3 pr_0;
	float3 pr_1;

	plane bbox_planes[12];
	
	float3 pmin = root->bbox.pmin;
	float3 pmax = root->bbox.pmax;

	//Find sun direction 
	float3 l_dir = -normalize(degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation));

	// First Plane 
	p0 = pmin;
	p1 = make_float3(pmin.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[0] = plane(p0, p1, pr_1, pr_0);

	// Second Plane 
	p0 = pmin;
	p1 = make_float3(pmax.x, pmin.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[1] = plane(p0, p1, pr_1, pr_0);

	// Third Plane 
	p0 = pmin;
	p1 = make_float3(pmin.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[2] = plane(p0, p1, pr_1, pr_0);

	// Fourth Plane 
	p0 = pmax;
	p1 = make_float3(pmax.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[3] = plane(p0, p1, pr_1, pr_0);

	// Fifth Plane 
	p0 = pmax;
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[4] = plane(p0, p1, pr_1, pr_0);

	// Sixth Plane 
	p0 = pmax;
	p1 = make_float3(pmin.x, pmax.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[5] = plane(p0, p1, pr_1, pr_0);




	// Seventh Plane 
	p0 = make_float3(pmin.x, pmax.y, pmin.z);
	p1 = make_float3(pmin.x, pmax.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[0] = plane(p0, p1, pr_1, pr_0);

	// Eighth Plane 
	p0 = make_float3(pmin.x, pmax.y, pmin.z);
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[1] = plane(p0, p1, pr_1, pr_0);

	// Nineth Plane 
	p0 = make_float3(pmin.x, pmax.y, pmax.z);
	p1 = make_float3(pmin.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[2] = plane(p0, p1, pr_1, pr_0);

	// Tenth Plane 
	p0 = make_float3(pmin.x, pmin.y, pmax.z);
	p1 = make_float3(pmax.x, pmin.y, pmax.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[3] = plane(p0, p1, pr_1, pr_0);

	// Eleventh Plane 
	p0 = make_float3(pmax.x, pmin.y, pmax.z);
	p1 = make_float3(pmax.x, pmin.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[4] = plane(p0, p1, pr_1, pr_0);

	// Twelfth Plane 
	p0 = make_float3(pmax.x, pmin.y, pmin.z);
	p1 = make_float3(pmax.x, pmax.y, pmin.z);
	pr_0 = project_to_earth(p0, l_dir, atmosphere);
	pr_1 = project_to_earth(p1, l_dir, atmosphere);
	bbox_planes[5] = plane(p0, p1, pr_1, pr_0);

	float t;

	for (int i = 0; i < 12; ++i) {
	
		if (bbox_planes[i].intersect(ray_pos, ray_dir, t)) tr = 0;
	}

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
	const GPU_VDB *gpu_vdb,
	OCTNode *root,
	const AtmosphereParameters atmosphere)
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
			Li = sample_atmosphere(kernel_params, atmosphere, ray_pos, wi);
		}
		else {
			light_pdf = sample_spherical(randstate, wi);
			Li = sample_env_tex(kernel_params, wi);
		}

		if (light_pdf > .0f && !isBlack(Li)) {

			float cos_theta = dot(ray_dir, wi);
			phase_pdf = henyey_greenstein(cos_theta, kernel_params.phase_g1);

			if (phase_pdf > .0f) {
				float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, root);
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

			float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, root);

			if (kernel_params.environment_type == 0)
			{
				Li = sample_atmosphere(kernel_params, atmosphere, ray_pos, wi);
			}
			else Li = sample_env_tex(kernel_params, wi);


			if (!isBlack(Li))
				Ld += Li * tr * weight;
		}


	}

	return Ld;

}

__device__ inline float3 estimate_point_light(
	Kernel_params kernel_params,
	const light_list lights,
	Rand_state &randstate,
	const float3 &ray_pos,
	float3 &ray_dir,
	const GPU_VDB *gpu_vdb,
	OCTNode *root)
{

	float3 Ld = make_float3(.0f);
	float max_density = gpu_vdb[0].vdb_info.max_density;

	for (int i = 0; i < lights.num_lights; i++) {

		float dist = length(lights.light_ptr[i].pos - ray_pos);
		float possible_tr = expf(-gpu_vdb[0].vdb_info.max_density * dist / (sqrtf(lights.light_ptr[i].power)*kernel_params.tr_depth));

		if (possible_tr > 0.01f) {
			float3 dir = normalize(lights.light_ptr[i].pos - ray_pos);
			float3 tr = Tr(randstate, ray_pos, dir, kernel_params, gpu_vdb, root);
			Ld += lights.light_ptr[i].Le(randstate, ray_pos, ray_dir, kernel_params.phase_g1, tr, max_density, kernel_params.density_mult, kernel_params.tr_depth);
		}

	}

	return Ld;

}


__device__ inline float3 estimate_sun(
	Kernel_params kernel_params,
	Rand_state &randstate,
	const float3 &ray_pos,
	float3 &ray_dir,
	const GPU_VDB *gpu_vdb,
	OCTNode *root,
	const AtmosphereParameters atmosphere)
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
	float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, root);

	float3 sky_irradiance;
	float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, ray_pos, ray_dir, wi, sky_irradiance);

	// Ld = Li * visibility.Tr * scattering_pdf / light_pdf  
	//Ld = (length(sky_irradiance)*sun_irradiance) * tr  * phase_pdf;
	Ld = tr  * phase_pdf;

	// No need for sampling BSDF with importance sampling
	// please see: http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Direct_Lighting.html#fragment-SampleBSDFwithmultipleimportancesampling-0

	return Ld;

}


__device__ inline float3 uniform_sample_one_light(
	Kernel_params kernel_params,
	const light_list lights,
	const float3 &ray_pos,
	float3 &ray_dir,
	Rand_state &randstate,
	const GPU_VDB *gpu_vdb,
	OCTNode *root,
	const AtmosphereParameters atmosphere)
{

	int nLights = 3; // number of lights
	float light_num = rand(&randstate) * nLights;

	float3 L = BLACK;

	if (light_num < 1) {

		if (kernel_params.sun_mult > .0f)
			L += estimate_sun(kernel_params, randstate, ray_pos, ray_dir, gpu_vdb, root, atmosphere) * kernel_params.sun_mult * kernel_params.sun_color;
	}
	else if (light_num >= 1 && light_num < 2) {

		if (lights.num_lights > 0)
			L += estimate_point_light(kernel_params, lights, randstate, ray_pos, ray_dir, gpu_vdb, root);

	}
	else {
		if (kernel_params.sky_mult > .0f)
			L += estimate_sky(kernel_params, randstate, ray_pos, ray_dir, gpu_vdb, root, atmosphere) * kernel_params.sky_mult;
	}

	return L * (float)nLights;

}

__device__ inline float3 sample(
	Rand_state &rand_state,
	float3 &ray_pos,
	const float3 &ray_dir,
	bool &interaction,
	float &tr,
	const Kernel_params &kernel_params,
	const GPU_VDB *volumes,
	OCTNode *root)
{
	// Run delta tracking with octree traversal
	// We assume that the ray_pos is inside the root bbox at the beginning 

	float t_min, t_max, t = 0.0f;


#if 1

	float inv_max_density = 1 / volumes[0].vdb_info.max_density;
	float inv_density_mult = 1.0f / kernel_params.density_mult;

	while (true) {
		t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult;
		ray_pos += ray_dir * t;
		if (!Contains(root->bbox, ray_pos))	break;
		float density = sum_density(ray_pos, root, volumes);
		tr *= 1 - fmaxf(.0f, density*inv_max_density);
		if (tr < 1.0f) tr += density;

		if (density * inv_max_density > rand(&rand_state)) {
			interaction = true;
			return (kernel_params.albedo / kernel_params.extinction) * float(kernel_params.energy_inject);
		}
	}
#endif





	// Code path 1:
	// This is a DDA stepping algorithm that checks the quadrant in Octree nodes and skips them if they contain no volumes 
	// this is faster but requires more code and a little bit more complex
#if 0   

	while (Contains(root->bbox, ray_pos)) {
		// First we figure out which quadrant on depth 3 our point sits in

		int depth3_node = get_quadrant(root, ray_pos);
		if (depth3_node > -1 ) {
			if (!(root->children[depth3_node]->num_volumes > 0)) { //We are in the depth3 node but it is empty
				root->children[depth3_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
				ray_pos += ray_dir * (t_max + EPS);
				goto stop;
			}
			else { // We are in depth3 node and it's not empty
				while (Contains(root->children[depth3_node]->bbox, ray_pos)) {
					int depth2_node = get_quadrant(root->children[depth3_node], ray_pos);
					if (depth2_node > -1) {
						if (!(root->children[depth3_node]->children[depth2_node]->num_volumes > 0)) { //We are in the depth2 node but it is empty
							root->children[depth3_node]->children[depth2_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
							ray_pos += ray_dir * (t_max + EPS);
							goto stop;
						}
						else {// We are in depth2 node and it's not empty

							while (Contains(root->children[depth3_node]->children[depth2_node]->bbox, ray_pos)) {
								int leaf_node = get_quadrant(root->children[depth3_node]->children[depth2_node], ray_pos);
								if (leaf_node > -1) {
									if (!(root->children[depth3_node]->children[depth2_node]->children[leaf_node]->num_volumes > 0)) { //We are in the leaf node but it is empty
										root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
										ray_pos += ray_dir * (t_max + EPS);
										goto stop;
									}
									else {//We are in the leaf node and it is not empty. We can finally do the sampling

										while (Contains(root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox, ray_pos)) {
											float inv_max_density = 1.0f / root->children[depth3_node]->children[depth2_node]->children[leaf_node]->max_extinction;
											float inv_density_mult = 1.0f / kernel_params.density_mult;

											float voxel_size = root->children[depth3_node]->children[depth2_node]->children[leaf_node]->voxel_size;

											t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult;
											ray_pos += ray_dir * t;
											float density = sum_density(ray_pos, root->children[depth3_node]->children[depth2_node]->children[leaf_node], volumes);
											// Accumulate opacity
											if (tr < 1.0f) tr += density;
											
											if (density * inv_max_density > rand(&rand_state)) {
												interaction = true;
												return (kernel_params.albedo / kernel_params.extinction) * float(kernel_params.energy_inject);
											}
											
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

stop:
#endif





	// Code path 2:
	// This is a DDA stepping algorithm that checks the quadrant in Octree nodes and skips them if they contain no volumes 
	// this path is slower but easier to implement 

#if 0
	while (Contains(root->bbox, ray_pos)) {

		// First we figure out which quadrant on various depths our point sits in

		int depth3_node = get_quadrant(root, ray_pos);
		int depth2_node = get_quadrant(root->children[depth3_node], ray_pos);
		int leaf_node = get_quadrant(root->children[depth3_node]->children[depth2_node], ray_pos);

		if (depth3_node < 0) break;
		if (!root->children[depth3_node]->num_volumes > 0) { //We are in the depth3 node but it is empty
			root->children[depth3_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
			ray_pos += ray_dir * (t_max + EPS);
		}
		else {
			if (depth2_node < 0) break;
			if (!root->children[depth3_node]->children[depth2_node]->num_volumes > 0) { //We are in the depth2 node but it is empty
				root->children[depth3_node]->children[depth2_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
				ray_pos += ray_dir * (t_max + EPS);
			}
			else {
				if (leaf_node < 0) break;
				if (!root->children[depth3_node]->children[depth2_node]->children[leaf_node]->num_volumes > 0) { //We are in the leaf node but it is empty
					root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
					ray_pos += ray_dir * (t_max + EPS);
				}
				else {//We are in the leaf node and it is not empty. We can finally do the sampling 

					float inv_max_density = 1.0f / root->children[depth3_node]->children[depth2_node]->children[leaf_node]->max_extinction;
					float inv_density_mult = 1.0f / kernel_params.density_mult;

					while (Contains(root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox, ray_pos)) {

						t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult;
						ray_pos += ray_dir * t;
						float density = sum_density(ray_pos, root->children[depth3_node]->children[depth2_node]->children[leaf_node], volumes);
						// Accumulate opacity
						if (tr < 1.0f) tr += density;

						if (density * inv_max_density > rand(&rand_state)) {
							interaction = true;
							return (kernel_params.albedo / kernel_params.extinction) * float(kernel_params.energy_inject);
						}
					}
				}

			}
		}

	}
#endif







#if 0
	// Code path 3:
	// This is the old algorithm with extra position awareness

	while (true) {

		int depth3_node = get_quadrant(root, ray_pos);
		if (depth3_node > -1) {
			if (root->children[depth3_node]->num_volumes < 1) { //We are in the depth3 node but it is empty
				root->children[depth3_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
				ray_pos += ray_dir * (t_max + EPS);
				continue;
			}
		}
		else break;

		int depth2_node = get_quadrant(root->children[depth3_node], ray_pos);
		if (depth2_node > -1) {
			if (root->children[depth3_node]->children[depth2_node]->num_volumes < 1) { //We are in the depth2 node but it is empty
				root->children[depth3_node]->children[depth2_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
				ray_pos += ray_dir * (t_max + EPS);
				continue;
			}
		}
		else break;


		int leaf_node = get_quadrant(root->children[depth3_node]->children[depth2_node], ray_pos);
		if (leaf_node > -1) {
			if (root->children[depth3_node]->children[depth2_node]->children[leaf_node]->num_volumes < 1) { //We are in the leaf node but it is empty
				root->children[depth3_node]->children[depth2_node]->children[leaf_node]->bbox.Intersect(ray_pos, ray_dir, t_min, t_max);
				ray_pos += ray_dir * (t_max + EPS);
				continue;
			}
		}
		else break;

		float inv_max_density = 1 / root->children[depth3_node]->children[depth2_node]->children[leaf_node]->max_extinction;
		float inv_density_mult = 1.0f / kernel_params.density_mult;

		t -= logf(1 - rand(&rand_state)) * inv_max_density * inv_density_mult * root->children[depth3_node]->children[depth2_node]->children[leaf_node]->voxel_size;
		ray_pos += ray_dir * t;
		if (!Contains(root->bbox, ray_pos))	break;
		float density = sum_density(ray_pos, root, volumes);
		tr *= 1 - fmaxf(.0f, density*inv_max_density);
		if (tr < 1.0f) tr += density;

		if (density * inv_max_density > rand(&rand_state)) {
			interaction = true;
			return (kernel_params.albedo / kernel_params.extinction) * float(kernel_params.energy_inject);
		}
	}

#endif

	return WHITE;

}

__device__ void traverse_bvh(float3 ray_pos, float3 ray_dir, const GPU_VDB *volumes, BVHNode *node, float3 &t, int &volume_idx) {

	float tmin, tmax;
	if (node->boundingBox.Intersect(ray_pos, ray_dir, tmin, tmax)) {


		if (node->IsLeaf()) {

			t = volumes[node->volIndex].rayBoxIntersect(ray_pos, ray_dir);
			if (t.z != NOHIT) volume_idx = node->volIndex;
		}
		else {

			BVHNode *leftChild = node->leftChild;
			BVHNode *rightChild = node->rightChild;

			traverse_bvh(ray_pos, ray_dir, volumes, leftChild, t, volume_idx);
			traverse_bvh(ray_pos, ray_dir, volumes, rightChild, t, volume_idx);

		}
	}
}


//////////////////////////////////////////////////////////////////////////
// Rendering Integrators 
//////////////////////////////////////////////////////////////////////////

// PBRT Volume Integrator
__device__ inline float3 vol_integrator(
	Rand_state rand_state,
	const light_list lights,
	float3 ray_pos,
	float3 ray_dir,
	float &tr,
	const Kernel_params kernel_params,
	const GPU_VDB *gpu_vdb,
	OCTNode *root,
	const AtmosphereParameters atmosphere)
{

	float3 L = BLACK;
	float3 beta = WHITE;
	float3 env_pos = ray_pos;
	bool mi;
	float t, tmax;

	if (root->bbox.Intersect(ray_pos, ray_dir, t, tmax)) { // found an intersection
		ray_pos += ray_dir * (t + EPS);
		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, tr, kernel_params, gpu_vdb, root);
			if (isBlack(beta)) break;

			if (mi) { // medium interaction 
				L += beta * uniform_sample_one_light(kernel_params, lights, ray_pos, ray_dir, rand_state, gpu_vdb, root, atmosphere);
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
			}

		}

		ray_dir = normalize(ray_dir);
	}

	if (length(beta) > 0.9999f) ray_pos = env_pos;

	L += beta * sample_atmosphere(kernel_params, atmosphere, ray_pos, ray_dir);

	tr = fminf(tr, 1.0f);
	return L;
}


// From Ray Tracing Gems Vol-28
__device__ inline float3 direct_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	float &tr,
	const Kernel_params kernel_params,
	const GPU_VDB *gpu_vdb,
	OCTNode *root,
	const AtmosphereParameters atmosphere)
{
	float3 L = BLACK;
	float3 beta = WHITE;
	bool mi = false;
	float3 env_pos = ray_pos;

	float t_min, t_max;

	if (root->bbox.Intersect(ray_pos, ray_dir, t_min, t_max)) {
		ray_pos += ray_dir * (t_min + EPS);

		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, tr, kernel_params, gpu_vdb, root);
			if (isBlack(beta)) break;

			if (mi) { // medium interaction 
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
			}
		}
	}

	ray_dir = normalize(ray_dir);

	float shadow_tr = 1.0f;
	get_shadow_box(ray_pos, ray_dir, root, atmosphere, kernel_params, shadow_tr);


	if (kernel_params.environment_type == 0) {

		if (mi) L += estimate_sun(kernel_params, rand_state, ray_pos, ray_dir, gpu_vdb, root, atmosphere) * beta  * kernel_params.sun_color * kernel_params.sun_mult;
		L += sample_atmosphere(kernel_params, atmosphere, env_pos, ray_dir) * beta * kernel_params.sky_mult * kernel_params.sky_color * shadow_tr;

	}
	else {

		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
		L += make_float3(texval.x, texval.y, texval.z) * kernel_params.sky_color * beta * isotropic();
	}


	tr = fminf(tr, 1.0f);
	return L;

}


//////////////////////////////////////////////////////////////////////////
// Test Kernels 
//////////////////////////////////////////////////////////////////////////

__device__ inline float3 octree_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	float &tr,
	const Kernel_params kernel_params,
	const GPU_VDB *gpu_vdb,
	OCTNode *root_node,
	const AtmosphereParameters atmosphere)
{

	float3 L = BLACK;
	float3 beta = WHITE;
	bool mi = false;
	float3 env_pos = ray_pos;

	float t, tmax;

	if (root_node->bbox.Intersect(ray_pos, ray_dir, t, tmax)) {
		ray_pos += ray_dir * t;

		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, tr, kernel_params, gpu_vdb, root_node);
			if (isBlack(beta)) break;

			if (mi) { // medium interaction 
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
			}
		}
	}

	ray_dir = normalize(ray_dir);

	if (kernel_params.environment_type == 0) {

		if (mi) L += estimate_sun(kernel_params, rand_state, ray_pos, ray_dir, gpu_vdb, root_node, atmosphere) * beta  * kernel_params.sun_color * kernel_params.sun_mult;
		L += sample_atmosphere(kernel_params, atmosphere, env_pos, ray_dir) * beta;

	}
	else {

		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
		L += make_float3(texval.x, texval.y, texval.z) * kernel_params.sky_color * beta * isotropic();
	}


	tr = fminf(tr, 1.0f);
	return L;

}

__device__ inline float3 visualize_BVH(float3 ray_pos, float3 ray_dir, const GPU_VDB *volumes, BVHNode *root_node)
{
	float3 L = BLACK;
	float3 t = make_float3(NOHIT);
	int vol_index;

	traverse_bvh(ray_pos, ray_dir, volumes, root_node, t, vol_index);

	if (t.z != NOHIT) {
		ray_pos += ray_dir * t.x;

		L = make_float3(float(vol_index) / 10.0f);


	}

	return L;
}

__device__ inline float3 visualize_OCTree(float3 ray_pos, float3 ray_dir, const GPU_VDB *volumes, OCTNode *root)
{
	float3 L = BLACK;
	float3 t = make_float3(NOHIT);
	float t_min, t_max;

	if (traverse_octree(ray_pos, ray_dir, root, t_min, t_max)) return RED;

	return L;
}

__device__ inline float3 render_earth(float3 ray_pos, float3 ray_dir, const Kernel_params kernel_params, const AtmosphereParameters atmosphere) {

	float3 earth_center = make_float3(.0f, -atmosphere.bottom_radius, .0f);
	float3 sun_direction = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

	float3 p = ray_pos - earth_center;
	float p_dot_v = dot(p, ray_dir);
	float p_dot_p = dot(p, p);
	float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
	float distance_to_intersection = -p_dot_v - sqrt(earth_center.y * earth_center.y - ray_earth_center_squared_distance);

	float ground_alpha = 0.0;
	float3 ground_radiance = make_float3(0.0);

	if (distance_to_intersection > 0.0) {
		float3 point = ray_pos + ray_dir * distance_to_intersection;
		float3 normal = normalize(point - earth_center);

		// Compute the radiance reflected by the ground.
		float3 sky_irradiance;
		float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, point - earth_center, normal, sun_direction, sky_irradiance);
		ground_radiance = atmosphere.ground_albedo * (1.0 / M_PI) * (sun_irradiance + sky_irradiance);

		float3 transmittance;
		float3 in_scatter = GetSkyRadianceToPoint(atmosphere, ray_pos - earth_center, point - earth_center, .0f, sun_direction, transmittance);
		ground_radiance = ground_radiance * transmittance + in_scatter;
		ground_alpha = 1.0;
	}

	float3 transmittance_sky;
	float3 radiance_sky = GetSkyRadiance(atmosphere, ray_pos - earth_center, ray_dir, .0f, sun_direction, transmittance_sky);

	float2 sun_size = make_float2(tanf(atmosphere.sun_angular_radius), cosf(atmosphere.sun_angular_radius));

	if (dot(ray_dir, sun_direction) > sun_size.y) {
		radiance_sky = radiance_sky + transmittance_sky * GetSolarRadiance(atmosphere);
	}

	ground_radiance = lerp(radiance_sky, ground_radiance, ground_alpha);

	return ground_radiance;

}


//////////////////////////////////////////////////////////////////////////
// Main kernel accessors
//////////////////////////////////////////////////////////////////////////


// Tone mapping functions 
__device__ inline float3 rtt_and_odt_fit(float3 v)
{
	float3 a = v * (v + 0.0245786f) - 0.000090537f;
	float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
	return a / b;
}


extern "C" __global__ void volume_rt_kernel(
	const camera cam,
	const light_list lights,
	const GPU_VDB *gpu_vdb,
	BVHNode *root_node,
	OCTNode *oct_root,
	const AtmosphereParameters atmosphere,
	const Kernel_params kernel_params) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= kernel_params.resolution.x || y >= kernel_params.resolution.y)
		return;

	// Initialize pseudorandom number generator (PRNG); assume we need no more than 4096 random numbers.
	const unsigned int idx = y * kernel_params.resolution.x + x;
	Rand_state rand_state;
	curand_init(idx, 0, kernel_params.iteration * 4096, &rand_state);

	// Get a blue noise sample from buffer
	int x_new = x % 256;
	int y_new = y % 256;
	int bn_index = y_new * 256 + x_new;
	float3 bn = kernel_params.blue_noise_buffer[bn_index];

	float u = float(x + bn.x) / float(kernel_params.resolution.x);
	float v = float(y + bn.y) / float(kernel_params.resolution.y);
	ray camera_ray = cam.get_ray(u, v, &rand_state);
	float3 ray_dir = normalize(camera_ray.B);
	float3 ray_pos = camera_ray.A;
	float3 value = WHITE;
	float tr = .0f;


	if (kernel_params.iteration < kernel_params.max_interactions && kernel_params.render)
	{
		if (kernel_params.integrator) value = vol_integrator(rand_state, lights, ray_pos, ray_dir, tr, kernel_params, gpu_vdb, oct_root, atmosphere);
		else value = direct_integrator(rand_state, ray_pos, ray_dir, tr, kernel_params, gpu_vdb, oct_root, atmosphere);

	}


	// Check if values contains nan or infinite values
	if (isNan(value) || isInf(value)) value = kernel_params.accum_buffer[idx];
	if (isnan(tr) || isinf(tr)) tr = 1.0f;

	// Accumulate.
	if (kernel_params.iteration == 0)kernel_params.accum_buffer[idx] = value;
	else if (kernel_params.iteration < kernel_params.max_interactions) {
		kernel_params.accum_buffer[idx] = kernel_params.accum_buffer[idx] +
			(value - kernel_params.accum_buffer[idx]) / (float)(kernel_params.iteration + 1);
	}

	// Update display buffer (ACES Tonemapping).
	// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

	mat3 aces_input_matrix(
		0.59719f, 0.35458f, 0.04823f,
		0.07600f, 0.90834f, 0.01566f,
		0.02840f, 0.13383f, 0.83777f);

	mat3 aces_output_matrix(
		1.60475f, -0.53108f, -0.07367f,
		-0.10208f, 1.10813f, -0.00605f,
		-0.00327f, -0.07276f, 1.07602f);


	float3 val = aces_input_matrix * kernel_params.accum_buffer[idx];
	val = rtt_and_odt_fit(val);
	val = aces_output_matrix * val * kernel_params.exposure_scale;

	// gamma correction
	const unsigned int r = (unsigned int)(255.0f * fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int g = (unsigned int)(255.0f * fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f));
	const unsigned int b = (unsigned int)(255.0f * fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f));

	kernel_params.display_buffer[idx] = 0xff000000 | (r << 16) | (g << 8) | b;


	// fill in raw buffer
	float4 raw = make_float4(val.x, val.y, val.z, tr);
	kernel_params.raw_buffer[idx] = raw;


	// Update blue_noise texture with golden ratio
	if (idx < (256 * 256)) {
		float3 val = kernel_params.blue_noise_buffer[idx];
		val += (1.0f + sqrtf(5.0f)) / 2.0f;
		val = fmodf(val, make_float3(1.0f));
		kernel_params.blue_noise_buffer[idx] = val;
	}
}
