

#include <device_launch_parameters.h>
#include <cuda_runtime.h> 
#include <curand_kernel.h>

#include <stdio.h>

#include "helper_math.h"
#include "matrix_math.h"
#include "atmosphere/constants.h"
#include "atmosphere/definitions.h"
//#include <//assert.h>

#define COMBINED_SCATTERING_TEXTURES

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

__device__  float DistanceToBottomAtmosphereBoundary(const AtmosphereParameters atmosphere,	float r, float mu) 
{
	float discriminant = r * r * (mu * mu - 1.0) + atmosphere.bottom_radius * atmosphere.bottom_radius;
	return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

__device__  bool RayIntersectsGround(const AtmosphereParameters atmosphere,	float r, float mu) 
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
	return altitude < profile.layers[0].width ?	GetLayerDensity(profile.layers[0], altitude) : GetLayerDensity(profile.layers[1], altitude);
}

__device__  float ComputeOpticalLengthToTopAtmosphereBoundary( const AtmosphereParameters atmosphere, const DensityProfile profile, float r, float mu) 
{
	// float of intervals for the numerical integration.
	const int SAMPLE_COUNT = 500;
	// The integration step, i.e. the float of each integration interval.
	float dx =	DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / float(SAMPLE_COUNT);
	// Integration loop.
	float result = 0.0 * m;
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		float d_i = float(i) * dx;
		// Distance between the current sample point and the planet center.
		float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
		// float density at the current sample point (divided by the float density
		// at the bottom of the atmosphere, yielding a dimensionless float).
		float y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);
		// Sample weight (from the trapezoidal rule).
		float weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5 : 1.0;
		result += y_i * weight_i * dx;
	}
	return result;
}

__device__  float3 ComputeTransmittanceToTopAtmosphereBoundary(	const AtmosphereParameters atmosphere, float r, float mu) 
{
	return expf(-(
		atmosphere.rayleigh_scattering *
		ComputeOpticalLengthToTopAtmosphereBoundary(
			atmosphere, atmosphere.rayleigh_density, r, mu) +
		atmosphere.mie_extinction *
		ComputeOpticalLengthToTopAtmosphereBoundary(
			atmosphere, atmosphere.mie_density, r, mu) +
		atmosphere.absorption_extinction *
		ComputeOpticalLengthToTopAtmosphereBoundary(
			atmosphere, atmosphere.absorption_density, r, mu)));
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
	float H = sqrtf(atmosphere.top_radius * atmosphere.top_radius -	atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	float rho =	SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
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
	float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -	atmosphere.bottom_radius * atmosphere.bottom_radius);
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

__device__  float3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(const AtmosphereParameters atmosphere, float2 frag_coord) 
{
	const float2 TRANSMITTANCE_TEXTURE_SIZE = make_float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
	float r;
	float mu;
	GetRMuFromTransmittanceTextureUv(atmosphere, (frag_coord / TRANSMITTANCE_TEXTURE_SIZE), r, mu);
	return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

__device__  float3 GetTransmittanceToTopAtmosphereBoundary(const AtmosphereParameters atmosphere, float r, float mu) 
{

	float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
	int x = int(floor(uv.x * TRANSMITTANCE_TEXTURE_WIDTH));
	int y = int(floor(uv.y * TRANSMITTANCE_TEXTURE_HEIGHT));
	int idx = (y * TRANSMITTANCE_TEXTURE_WIDTH) + x;
	idx = clamp(idx, 0, TRANSMITTANCE_TEXTURE_WIDTH*TRANSMITTANCE_TEXTURE_HEIGHT);
	
	const float3 texval = atmosphere.transmittance_buffer[idx];	
	return texval;
}

__device__  float3 GetTransmittance(const AtmosphereParameters atmosphere, float r, float mu, float d, bool ray_r_mu_intersects_ground) 
{

	float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	float mu_d = ClampCosine((r * mu + d) / r_d);

	if (ray_r_mu_intersects_ground) {
		return fminf(GetTransmittanceToTopAtmosphereBoundary( atmosphere,  r_d, -mu_d) / GetTransmittanceToTopAtmosphereBoundary( atmosphere,  r, -mu),	make_float3(1.0f));
	}
	else {
		return fminf(GetTransmittanceToTopAtmosphereBoundary(atmosphere,  r, mu) / GetTransmittanceToTopAtmosphereBoundary( atmosphere,  r_d, mu_d), make_float3(1.0));
	}
}

__device__  float3 GetTransmittanceToSun(const AtmosphereParameters atmosphere, float r, float mu_s) 
{

	float sin_theta_h = atmosphere.bottom_radius / r;
	float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
	return GetTransmittanceToTopAtmosphereBoundary(
		atmosphere,  r, mu_s) *
		smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
			sin_theta_h * atmosphere.sun_angular_radius / rad,
			mu_s - cos_theta_h);
}

__device__  void ComputeSingleScatteringIntegrand(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, float d, bool ray_r_mu_intersects_ground, float3 &rayleigh, float3 &mie) 
{
	float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	float mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
	float3 transmittance =	GetTransmittance( atmosphere, r, mu, d, ray_r_mu_intersects_ground) * GetTransmittanceToSun( atmosphere, r_d, mu_s_d);
	rayleigh = transmittance * GetProfileDensity(atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
	mie = transmittance * GetProfileDensity(atmosphere.mie_density, r_d - atmosphere.bottom_radius);
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

__device__  void ComputeSingleScattering(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, float3 &rayleigh, float3 &mie) 
{

	// float of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the float of each integration interval.
	float dx =
		DistanceToNearestAtmosphereBoundary(atmosphere, r, mu, ray_r_mu_intersects_ground) / float(SAMPLE_COUNT);
	// Integration loop.
	float3 rayleigh_sum = make_float3(0.0f);
	float3 mie_sum = make_float3(0.0f);
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		float d_i = float(i) * dx;
		// The Rayleigh and Mie single scattering at the current sample point.
		float3 rayleigh_i;
		float3 mie_i;
		ComputeSingleScatteringIntegrand(atmosphere, r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
		// Sample weight (from the trapezoidal rule).
		float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_sum += rayleigh_i * weight_i;
		mie_sum += mie_i * weight_i;
	}
	rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance * atmosphere.rayleigh_scattering;
	mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
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

__device__  void GetRMuMuSNuFromScatteringTextureUvwz(const AtmosphereParameters atmosphere, float4 uvwz, float &r, float &mu, float &mu_s,	float &nu, bool &ray_r_mu_intersects_ground) 
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
	float frag_coord_mu_s =	fmodf(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
	float4 uvwz = make_float4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /	SCATTERING_TEXTURE_SIZE;
	GetRMuMuSNuFromScatteringTextureUvwz(atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	// Clamp nu to its valid range of values, given mu and mu_s.
	nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)), mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

__device__  void ComputeSingleScatteringTexture(const AtmosphereParameters atmosphere, float3 frag_coord, float3& rayleigh, float3& mie) {
	float r;
	float mu;
	float mu_s;
	float nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	ComputeSingleScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
}

__device__  float3 GetScattering(const AtmosphereParameters atmosphere,	float4 *scattering_buffer, float r, float mu, float mu_s, float nu,	bool ray_r_mu_intersects_ground) 
{
	
	float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
	float tex_x = floor(tex_coord_x);
	float lerp = tex_coord_x - tex_x;
	float3 uvw0 = make_float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),	uvwz.z, uvwz.w);
	float3 uvw1 = make_float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

	int3 uvw0_i = make_int3(uvw0.x * SCATTERING_TEXTURE_WIDTH, uvw0.y * SCATTERING_TEXTURE_HEIGHT, uvw0.z * SCATTERING_TEXTURE_DEPTH);
	int3 uvw1_i = make_int3(uvw1.x * SCATTERING_TEXTURE_WIDTH, uvw1.y * SCATTERING_TEXTURE_HEIGHT, uvw1.z * SCATTERING_TEXTURE_DEPTH);

	int index0 = uvw0_i.x + SCATTERING_TEXTURE_WIDTH * (uvw0_i.y + SCATTERING_TEXTURE_HEIGHT * uvw0_i.z);
	int index1 = uvw1_i.x + SCATTERING_TEXTURE_WIDTH * (uvw1_i.y + SCATTERING_TEXTURE_HEIGHT * uvw1_i.z);

	const float4 val1 = scattering_buffer[index0];
	const float4 val2 = scattering_buffer[index1];

	return float3(make_float3(val1) * (1.0 - lerp) + make_float3(val2) * lerp);
}

__device__  float3 GetScattering(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu,	bool ray_r_mu_intersects_ground, int scattering_order) 
{
	if (scattering_order == 1) {
		float3 rayleigh = GetScattering(atmosphere, atmosphere.delta_rayleigh_scattering_buffer, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
		float3 mie = GetScattering(atmosphere, atmosphere.delta_mie_scattering_buffer, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
		return rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
	}
	else {
		return GetScattering(atmosphere, atmosphere.scattering_buffer, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	}
}

__device__  float3 GetIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s);

__device__  float3 ComputeScatteringDensity(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, int scattering_order) 
{
	// Compute unit float3 vectors for the zenith, the view float3 omega and
	// and the sun float3 omega_s, such that the cosine of the view-zenith
	// float is mu, the cosine of the sun-zenith float is mu_s, and the cosine of
	// the view-sun float is nu. The goal is to simplify computations below.
	float3 zenith_direction = make_float3(0.0, 0.0, 1.0);
	float3 omega = make_float3(sqrt(1.0f - mu * mu), 0.0, mu);
	float sun_dir_x = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
	float sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
	float3 omega_s = make_float3(sun_dir_x, sun_dir_y, mu_s);

	const int SAMPLE_COUNT = 16;
	const float dphi = pi() / float(SAMPLE_COUNT);
	const float dtheta = pi() / float(SAMPLE_COUNT);
	float3 rayleigh_mie =
		make_float3(0.0f * watt_per_cubic_meter_per_sr_per_nm());

	// Nested loops for the integral over all the incident directions omega_i.
	for (int l = 0; l < SAMPLE_COUNT; ++l) {
		float theta = (float(l) + 0.5) * dtheta;
		float cos_theta = cos(theta);
		float sin_theta = sin(theta);
		bool ray_r_theta_intersects_ground =
			RayIntersectsGround(atmosphere, r, cos_theta);

		// The distance and transmittance to the ground only depend on theta, so we
		// can compute them in the outer loop for efficiency.
		float distance_to_ground = 0.0 * m;
		float3 transmittance_to_ground = make_float3(0.0f);
		float3 ground_albedo = make_float3(0.0f);
		if (ray_r_theta_intersects_ground) {
			distance_to_ground = DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);
			transmittance_to_ground = GetTransmittance(atmosphere, r, cos_theta, distance_to_ground, true);
			ground_albedo = atmosphere.ground_albedo;
		}

		for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {
			float phi = (float(m) + 0.5) * dphi;
			float3 omega_i = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
			float domega_i = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			// The radiance L_i arriving from float3 omega_i after n-1 bounces is
			// the sum of a term given by the precomputed scattering texture for the
			// (n-1)-th order:
			float nu1 = dot(omega_s, omega_i);
			float3 incident_radiance = GetScattering(atmosphere, r, omega_i.z, mu_s, nu1, ray_r_theta_intersects_ground, scattering_order - 1);

			// and of the contribution from the light paths with n-1 bounces and whose
			// last bounce is on the ground. This contribution is the product of the
			// transmittance to the ground, the ground albedo, the ground BRDF, and
			// the irradiance received on the ground after n-2 bounces.
			float3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);
			float3 ground_irradiance = GetIrradiance( atmosphere, atmosphere.bottom_radius, dot(ground_normal, omega_s));
			incident_radiance += transmittance_to_ground * ground_albedo * (1.0 / (PI * sr)) * ground_irradiance;

			// The radiance finally scattered from float3 omega_i towards float3
			// -omega is the product of the incident radiance, the scattering
			// coefficient, and the phase function for directions omega and omega_i
			// (all this summed over all particle types, i.e. Rayleigh and Mie).
			float nu2 = dot(omega, omega_i);
			float rayleigh_density = GetProfileDensity(	atmosphere.rayleigh_density, r - atmosphere.bottom_radius);
			float mie_density = GetProfileDensity( atmosphere.mie_density, r - atmosphere.bottom_radius);
			rayleigh_mie += incident_radiance * ( 
				atmosphere.rayleigh_scattering * rayleigh_density *
				RayleighPhaseFunction(nu2) +
				atmosphere.mie_scattering * mie_density *
				MiePhaseFunction(atmosphere.mie_phase_function_g, nu2)) *
				domega_i;
		}
	}
	return rayleigh_mie;
}

__device__  float3 ComputeMultipleScattering(const AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground) 
{
	// float of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the float of each integration interval.
	float dx =
		DistanceToNearestAtmosphereBoundary(
			atmosphere, r, mu, ray_r_mu_intersects_ground) /
		float(SAMPLE_COUNT);
	// Integration loop.
	float3 rayleigh_mie_sum =
		make_float3(0.0f * watt_per_square_meter_per_sr_per_nm());
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		float d_i = float(i) * dx;

		// The r, mu and mu_s parameters at the current integration point (see the
		// single scattering section for a detailed explanation).
		float r_i =
			ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
		float mu_i = ClampCosine((r * mu + d_i) / r_i);
		float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

		// The Rayleigh and Mie multiple scattering at the current sample point.
		float3 rayleigh_mie_i =	
			GetScattering( atmosphere, atmosphere.delta_scattering_density_buffer, r_i, mu_i, mu_s_i, nu, ray_r_mu_intersects_ground) *	
			GetTransmittance( atmosphere, r, mu, d_i, ray_r_mu_intersects_ground) *	dx;
		// Sample weight (from the trapezoidal rule).
		float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_mie_sum += rayleigh_mie_i * weight_i;
	}
	return rayleigh_mie_sum;
}

__device__  float3 ComputeScatteringDensityTexture(const AtmosphereParameters atmosphere, float3 frag_coord, int scattering_order) {
	float r;
	float mu;
	float mu_s;
	float nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeScatteringDensity(atmosphere, r, mu, mu_s, nu, scattering_order);
}

__device__  float3 ComputeMultipleScatteringTexture(const AtmosphereParameters atmosphere, float3 frag_coord, float& nu) {
	float r;
	float mu;
	float mu_s;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeMultipleScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
}

__device__  float3 ComputeDirectIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s) {

	float alpha_s = atmosphere.sun_angular_radius / rad;
	// Approximate average of the cosine factor mu_s over the visible fraction of
	// the Sun disc.
	
	float average_cosine_factor =	mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));
	return atmosphere.solar_irradiance * GetTransmittanceToTopAtmosphereBoundary( atmosphere, r, mu_s) * average_cosine_factor;
}

__device__  float3 ComputeIndirectIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s, int scattering_order) 
{

	const int SAMPLE_COUNT = 32;
	const float dphi = pi() / float(SAMPLE_COUNT);
	const float dtheta = pi() / float(SAMPLE_COUNT);

	float3 result =	make_float3(0.0f * watt_per_square_meter_per_nm());
	float3 omega_s = make_float3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
	for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
		float theta = (float(j) + 0.5) * dtheta;
		for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
			float phi = (float(i) + 0.5) * dphi;
			float3 omega =
				make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
			float domega = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			float nu = dot(omega, omega_s);
			result += GetScattering(atmosphere, r, omega.z, mu_s, nu, false, scattering_order) * omega.z * domega;
		}
	}
	return result;
}

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

__device__  float3 ComputeDirectIrradianceTexture(const AtmosphereParameters atmosphere, float2 frag_coord) {
	
	float r;
	float mu_s;
	
	GetRMuSFromIrradianceTextureUv(	atmosphere, (frag_coord / make_float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT)), r, mu_s);
	return ComputeDirectIrradiance(atmosphere, r, mu_s);
}

__device__  float3 ComputeIndirectIrradianceTexture(const AtmosphereParameters atmosphere, float2 frag_coord, int scattering_order) {
	float r;
	float mu_s;
	GetRMuSFromIrradianceTextureUv(	atmosphere, frag_coord / make_float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT), r, mu_s);
	return ComputeIndirectIrradiance(atmosphere, r, mu_s, scattering_order);
}

__device__  float3 GetIrradiance(const AtmosphereParameters atmosphere, float r, float mu_s) {
	float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);

	int x = int(floor(uv.x * IRRADIANCE_TEXTURE_WIDTH));
	int y = int(floor(uv.y * IRRADIANCE_TEXTURE_HEIGHT));
	int idx = (y * IRRADIANCE_TEXTURE_WIDTH) + x;
	idx = clamp(idx, 0, IRRADIANCE_TEXTURE_WIDTH*IRRADIANCE_TEXTURE_HEIGHT);

	const float3 val = atmosphere.irradiance_buffer[idx];
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
	float tex_x = floor(tex_coord_x);
	float lerp = tex_coord_x - tex_x;
	float3 uvw0 = make_float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),	uvwz.z, uvwz.w);
	float3 uvw1 = make_float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

	int3 uvw0_i = make_int3(uvw0.x * SCATTERING_TEXTURE_WIDTH, uvw0.y * SCATTERING_TEXTURE_HEIGHT, uvw0.z * SCATTERING_TEXTURE_DEPTH);
	int3 uvw1_i = make_int3(uvw1.x * SCATTERING_TEXTURE_WIDTH, uvw1.y * SCATTERING_TEXTURE_HEIGHT, uvw1.z * SCATTERING_TEXTURE_DEPTH);

	int index0 = uvw0_i.x + SCATTERING_TEXTURE_WIDTH * (uvw0_i.y + SCATTERING_TEXTURE_HEIGHT * uvw0_i.z);
	int index1 = uvw1_i.x + SCATTERING_TEXTURE_WIDTH * (uvw1_i.y + SCATTERING_TEXTURE_HEIGHT * uvw1_i.z);

#ifdef COMBINED_SCATTERING_TEXTURES
	float4 combined_scattering = atmosphere.scattering_buffer[index0] * (1.0 - lerp) + atmosphere.scattering_buffer[index1] * lerp;
	float3 scattering = make_float3(combined_scattering);
	single_mie_scattering =	GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
#else
	float3 scattering = make_float3(atmosphere.scattering_buffer[index0] * (1.0 - lerp) + atmosphere.scattering_buffer[index1] * lerp);
	single_mie_scattering = make_float3(atmosphere.optional_mie_single_scattering_buffer[index0] * (1.0 - lerp) + atmosphere.optional_mie_single_scattering_buffer[index1] * lerp);
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
		float r_p =	ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
		float mu_p = (r * mu + d) / r_p;
		float mu_s_p = (r * mu_s + d * nu) / r_p;

		scattering = GetCombinedScattering(	atmosphere,	r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering);
		float3 shadow_transmittance = GetTransmittance(atmosphere, r, mu, shadow_length, ray_r_mu_intersects_ground);
		scattering = scattering * shadow_transmittance;
		single_mie_scattering = single_mie_scattering * shadow_transmittance;
	}
	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

__device__  float3 GetSkyRadianceToPoint(const AtmosphereParameters atmosphere, float3 camera, float3 point, float shadow_length, float3 sun_direction, float3& transmittance) 
{
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	float3 view_ray = normalize(point - camera);
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

	// Compute the r, mu, mu_s and nu parameters for the first texture lookup.
	float mu = rmu / r;
	float mu_s = dot(camera, sun_direction) / r;
	float nu = dot(view_ray, sun_direction);
	float d = length(point - camera);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = GetTransmittance(atmosphere, r, mu, d, ray_r_mu_intersects_ground);

	float3 single_mie_scattering;
	float3 scattering = GetCombinedScattering(atmosphere,r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);

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
	float3 scattering_p = GetCombinedScattering(atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,single_mie_scattering_p);

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
	single_mie_scattering = GetExtrapolatedSingleMieScattering( atmosphere, make_float4(scattering, single_mie_scattering.x));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering *	smoothstep(float(0.0), float(0.01), mu_s);

	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *	MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

__device__  float3 GetSunAndSkyIrradiance(const AtmosphereParameters atmosphere,float3 point, float3 normal, float3 sun_direction, float3 &sky_irradiance) 
{
	float r = length(point);
	float mu_s = dot(point, sun_direction) / r;

	// Indirect irradiance (approximated if the surface is not horizontal).
	sky_irradiance = GetIrradiance(atmosphere, r, mu_s) * (1.0 + dot(normal, point) / r) * 0.5;

	// Direct irradiance.
	return atmosphere.solar_irradiance * GetTransmittanceToSun(atmosphere,  r, mu_s) * max(dot(normal, sun_direction), 0.0);
}


// KERNEL ACCESSORS
//**************************************************************************************************************************************

extern "C" __global__ void calculate_transmittance(const AtmosphereParameters atmosphere) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= TRANSMITTANCE_TEXTURE_WIDTH || y >= TRANSMITTANCE_TEXTURE_HEIGHT) return;
	const unsigned int idx = y * TRANSMITTANCE_TEXTURE_WIDTH + x;

	float2 frag_coord = make_float2(x, y);
	frag_coord += make_float2(0.5f, 0.5f);
	atmosphere.transmittance_buffer[idx] = ComputeTransmittanceToTopAtmosphereBoundaryTexture(atmosphere, frag_coord);

}

extern "C" __global__ void calculate_direct_irradiance(const AtmosphereParameters atmosphere, const int blend){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= IRRADIANCE_TEXTURE_WIDTH || y >= IRRADIANCE_TEXTURE_HEIGHT) return;
	const unsigned int idx = y * IRRADIANCE_TEXTURE_WIDTH + x;
	
	float2 frag_coord = make_float2(x, y);
	frag_coord += make_float2(0.5f, 0.5f);

	if(!blend) atmosphere.irradiance_buffer[idx] = make_float3(.0f);
	
	float3 temp_val = atmosphere.irradiance_buffer[idx];

	atmosphere.delta_irradience_buffer[idx] = ComputeDirectIrradianceTexture(atmosphere, frag_coord);
	
	if(blend) atmosphere.irradiance_buffer[idx] += temp_val;
	
}

extern "C" __global__ void calculate_indirect_irradiance(const AtmosphereParameters atmosphere, const int blend, mat4 luminance_from_radiance , const int scattering_order) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= IRRADIANCE_TEXTURE_WIDTH || y >= IRRADIANCE_TEXTURE_HEIGHT) return;
	const unsigned int idx = y * IRRADIANCE_TEXTURE_WIDTH + x;

	float2 frag_coord = make_float2(x, y);
	frag_coord += make_float2(0.5f, 0.5f);

	float3 delta_irradiance_value = ComputeIndirectIrradianceTexture(atmosphere, frag_coord, scattering_order-1);

	float3 temp_val = atmosphere.irradiance_buffer[idx];

	atmosphere.irradiance_buffer[idx] = luminance_from_radiance * delta_irradiance_value;

	atmosphere.delta_irradience_buffer[idx] = atmosphere.irradiance_buffer[idx];

	if (blend) atmosphere.irradiance_buffer[idx] += temp_val;

}

extern "C" __global__ void calculate_multiple_scattering(const AtmosphereParameters atmosphere, const int blend, mat4 luminance_from_radiance, const int scattering_order){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= SCATTERING_TEXTURE_WIDTH || y >= SCATTERING_TEXTURE_HEIGHT || z >= SCATTERING_TEXTURE_DEPTH) return;

	const unsigned int idx = x + SCATTERING_TEXTURE_WIDTH * (y + SCATTERING_TEXTURE_HEIGHT * z);

	float3 frag_coord = make_float3(x, y, z);
	frag_coord += make_float3(0.5f, 0.5f, 0.5f);

	float4 temp_val = atmosphere.scattering_buffer[idx];

	float nu;
	float3 delta_multiple_scattering_value = ComputeMultipleScatteringTexture(atmosphere, frag_coord, nu);

	atmosphere.delta_multiple_scattering_buffer[idx] = make_float4(delta_multiple_scattering_value, 1.0f);

	atmosphere.scattering_buffer[idx] = make_float4((luminance_from_radiance * delta_multiple_scattering_value) / RayleighPhaseFunction(nu), .0f);

	if (blend) atmosphere.scattering_buffer[idx] += temp_val;

}

extern "C" __global__ void calculate_scattering_density(const AtmosphereParameters atmosphere, const float4 blend, mat4 luminance_from_radiance, const int scattering_order){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= SCATTERING_TEXTURE_WIDTH || y >= SCATTERING_TEXTURE_HEIGHT || z >= SCATTERING_TEXTURE_DEPTH) return;

	const unsigned int idx = x + SCATTERING_TEXTURE_WIDTH * (y + SCATTERING_TEXTURE_HEIGHT * z);

	float3 frag_coord = make_float3(x, y, z);
	frag_coord += make_float3(0.5f, 0.5f, 0.5f);

	float3 scattering_density = ComputeScatteringDensityTexture(atmosphere, frag_coord, scattering_order);
	atmosphere.delta_scattering_density_buffer[idx] = make_float4(scattering_density, 1.0f);
}

extern "C" __global__ void calculate_single_scattering(const AtmosphereParameters atmosphere, const float4 blend, mat4 luminance_from_radiance){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= SCATTERING_TEXTURE_WIDTH || y >= SCATTERING_TEXTURE_HEIGHT || z>= SCATTERING_TEXTURE_DEPTH) return;
	
	const unsigned int idx = x + SCATTERING_TEXTURE_WIDTH * (y + SCATTERING_TEXTURE_HEIGHT * z);

	float3 frag_coord = make_float3(x, y, z);
	frag_coord += make_float3(0.5f, 0.5f, 0.5f);
	float3 delta_rayleigh, delta_mie;

	float4 temp_scatter, temp_single_scatter;
	temp_scatter = atmosphere.scattering_buffer[idx];
	temp_single_scatter = atmosphere.optional_mie_single_scattering_buffer[idx];

	ComputeSingleScatteringTexture(atmosphere, frag_coord, delta_rayleigh, delta_mie);

	atmosphere.delta_rayleigh_scattering_buffer[idx] = make_float4(delta_rayleigh, 1.0f);
	atmosphere.delta_mie_scattering_buffer[idx] = make_float4(delta_mie, 1.0f);

	atmosphere.scattering_buffer[idx] = make_float4(luminance_from_radiance * delta_rayleigh, (luminance_from_radiance * delta_mie).x);
	atmosphere.optional_mie_single_scattering_buffer[idx] = make_float4(delta_mie, 1.0f);

	if (blend.z) atmosphere.scattering_buffer[idx] += temp_scatter;
	if (blend.w) atmosphere.optional_mie_single_scattering_buffer[idx] += temp_single_scatter;

}

