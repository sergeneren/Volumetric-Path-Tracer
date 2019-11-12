


#ifndef __ATMOSPHERE_FUNCTIONS_H__
#define __ATMOSPHERE_FUNCTIONS_H__



#include "helper_math.h"
#include "constants.h"
#include "definitions.h"
#include "atmosphere.h"


__device__ __forceinline__ float atmosphere::DistanceToTopAtmosphereBoundary(
	const AtmosphereParameters atmosphere,
	float r, float mu) {

	float discriminant = r * r * (mu * mu - 1.0) + atmosphere.top_radius * atmosphere.top_radius;
	return fmaxf(-r * mu + fmaxf(0.0f, sqrtf(discriminant)), .0f);
}

__device__ __forceinline__ float atmosphere::GetTextureCoordFromUnitRange(float x, int texture_size) {
	return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

__device__ __forceinline__ float  atmosphere::GetUnitRangeFromTextureCoord(float u, int texture_size) {
	return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}


__device__ __forceinline__ float2 atmosphere::GetTransmittanceTextureUvFromRMu(
	const AtmosphereParameters atmosphere,
	float r, float mu_s) {

	float2 uv = make_float2(.0f, .0f);

	if (r < atmosphere.bottom_radius || r > atmosphere.top_radius) return uv;
	if (mu_s < -1.0f || mu_s > 1.0f) return uv;

	float H = sqrt(atmosphere.top_radius * atmosphere.top_radius - atmosphere.bottom_radius * atmosphere.bottom_radius);
	float rho = fmaxf(0.0f, sqrtf(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius));

	float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu_s);
	float d_min = atmosphere.top_radius - r;
	float d_max = rho + H;
	float x_mu = (d - d_min) / (d_max - d_min);
	float x_r = rho / H;
	uv = make_float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH), GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));

	return uv;

}

__device__ __forceinline__ float2 atmosphere::GetIrradianceTextureUvFromRMuS(
	const AtmosphereParameters atmosphere,
	float r, float mu_s) {

	float2 uv = make_float2(.0f, .0f);

	if (r < atmosphere.bottom_radius || r > atmosphere.top_radius) return uv;
	if (mu_s < -1.0f || mu_s > 1.0f) return uv;

	float x_r = (r - atmosphere.bottom_radius) /
		(atmosphere.top_radius - atmosphere.bottom_radius);
	float x_mu_s = mu_s * 0.5 + 0.5;
	uv = make_float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH), GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));

	return uv;
}

__device__ __forceinline__ float atmosphere::GetTransmittanceToTopAtmosphereBoundary(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	float r, float mu_s) {

	float tr = .0f;

	if (r >= atmosphere.bottom_radius && r <= atmosphere.top_radius) {

		float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu_s);

		const float texval = tex2D<float>(transmittance_texture, uv.x, uv.y);

		tr = texval;
	}

	return tr;

}

__device__ __forceinline__ float3 atmosphere::getIrradiance(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t irradiance_texture,
	float r, float mu_s) {


	float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);

	const float4 texval = tex2D<float4>(irradiance_texture, uv.x, uv.y);

	float3 val = make_float3(texval.x, texval.y, texval.z);

	return val;
}


__device__ __forceinline__ float atmosphere::getTransmittanceToSun(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	float r, float mu_s) {

	float sin_theta_h = atmosphere.bottom_radius / r;
	float cos_theta_h = -sqrtf(max(1.0 - sin_theta_h * sin_theta_h, 0.0));

	return GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittance_texture, r, mu_s) *
		smoothstep(-sin_theta_h * sun_angular_radius,
			sin_theta_h * sun_angular_radius,
			mu_s - cos_theta_h);

}

__device__ __forceinline__ bool atmosphere::RayIntersectsGround(const AtmosphereParameters atmosphere, float r, float mu) {
	if (r < atmosphere.bottom_radius) return false;
	if (mu < -1.0 || mu > 1.0) return false;

	return mu < 0.0 && r * r * (mu * mu - 1.0) +
		atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}

__device__ __forceinline__ float atmosphere::GetTransmittance(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	float r, float mu, float d, bool ray_r_mu_intersects_ground)
{
	if(r < atmosphere.bottom_radius || r > atmosphere.top_radius) return .0f;
	if(mu < -1.0 || mu > 1.0) return .0f;
	if(d < 0.0) return .0f;

	float r_d = clamp(sqrt(d * d + 2.0 * r * mu * d + r * r), atmosphere.bottom_radius, atmosphere.top_radius);
	float mu_d = ClampCosine((r * mu + d) / r_d);

	if (ray_r_mu_intersects_ground) {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, -mu_d) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, -mu),
			float(1.0));
	}
	else {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, mu) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, mu_d),
			float(1.0));
	}
}


__device__ __forceinline__ float3 atmosphere::GetSkyRadianceToPoint(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	const cudaTextureObject_t scattering_texture,
	const cudaTextureObject_t single_mie_scattering_texture,
	float3 camera, float3 point, float shadow_length,
	float3 sun_direction, float transmittance) {
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
	if (distance_to_top_atmosphere_boundary > 0.0f) {
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

	transmittance = GetTransmittance(atmosphere, transmittance_texture,
		r, mu, d, ray_r_mu_intersects_ground);

	float3 single_mie_scattering;
	float3 scattering = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground,
		single_mie_scattering);

	// Compute the r, mu, mu_s and nu parameters for the second texture lookup.
	// If shadow_length is not 0 (case of light shafts), we want to ignore the
	// scattering along the last shadow_length meters of the view ray, which we
	// do by subtracting shadow_length from d (this way scattering_p is equal to
	// the S|x_s=x_0-lv term in Eq. (17) of our paper).
	d = max(d - shadow_length, 0.0f);
	float r_p = clamp(sqrt(d * d + 2.0 * r * mu * d + r * r), atmosphere.bottom_radius, atmosphere.top_radius);
	float mu_p = (r * mu + d) / r_p;
	float mu_s_p = (r * mu_s + d * nu) / r_p;

	float3 single_mie_scattering_p;
	float3 scattering_p = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
		single_mie_scattering_p);

	// Combine the lookup results to get the scattering between camera and point.
	float shadow_transmittance = transmittance;
	if (shadow_length > 0.0f) {
		// This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
		shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
			r, mu, d, ray_r_mu_intersects_ground);
	}
	scattering = scattering - shadow_transmittance * scattering_p;
	single_mie_scattering =
		single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
	single_mie_scattering = GetExtrapolatedSingleMieScattering(
		atmosphere, vec4(scattering, single_mie_scattering.r));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering *
		smoothstep(float(0.0), float(0.01), mu_s);

	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

__device__ __forceinline__ float3 atmosphere::getSunAndSkyIrradiance(
	const AtmosphereParameters atmosphere,
	const cudaTextureObject_t transmittance_texture,
	const cudaTextureObject_t irradiance_texture,
	const float3 position, const float3 normal, const float3 sun_direction,
	float3 &sky_irradiance) {

	float r = length(position);
	float mu_s = dot(position, sun_direction) / r;

	sky_irradiance = getIrradiance(atmosphere, irradiance_texture, r, mu_s) * (1.0f + dot(normal, position) / r) * 0.5f;

	return atmosphere.solar_irradiance * getTransmittanceToSun(atmosphere, transmittance_texture, r, mu_s) * fmaxf(dot(normal, sun_direction), .0f);

}


#endif






__device__ __forceinline__ Number atmosphere::ClampCosine(Number mu) {
	return clamp(mu, Number(-1.0), Number(1.0));
}

__device__ __forceinline__ Length atmosphere::ClampDistance(Length d) {
	return max(d, 0.0 * m);
}

__device__ __forceinline__ Length atmosphere::ClampRadius(const AtmosphereParameters atmosphere, Length r) {
	return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

__device__ __forceinline__ Length atmosphere::SafeSqrt(Area a) {
	return sqrt(max(a, 0.0 * m2));
}

__device__ __forceinline__ Length atmosphere::DistanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu) {
	assert(r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	Area discriminant = r * r * (mu * mu - 1.0) +
		atmosphere.top_radius * atmosphere.top_radius;
	return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

__device__ __forceinline__ Length atmosphere::DistanceToBottomAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	Area discriminant = r * r * (mu * mu - 1.0) +
		atmosphere.bottom_radius * atmosphere.bottom_radius;
	return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

__device__ __forceinline__ bool atmosphere::RayIntersectsGround(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	return mu < 0.0 && r * r * (mu * mu - 1.0) +
		atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;
}

__device__ __forceinline__ Number atmosphere::GetLayerDensity(IN(DensityProfileLayer) layer, Length altitude) {
	Number density = layer.exp_term * exp(layer.exp_scale * altitude) +
		layer.linear_term * altitude + layer.constant_term;
	return clamp(density, Number(0.0), Number(1.0));
}

__device__ __forceinline__ Number atmosphere::GetProfileDensity(IN(DensityProfile) profile, Length altitude) {
	return altitude < profile.layers[0].width ?
		GetLayerDensity(profile.layers[0], altitude) :
		GetLayerDensity(profile.layers[1], altitude);
}

__device__ __forceinline__ Length atmosphere::ComputeOpticalLengthToTopAtmosphereBoundary(
	IN(AtmosphereParameters) atmosphere, IN(DensityProfile) profile,
	Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	// Number of intervals for the numerical integration.
	const int SAMPLE_COUNT = 500;
	// The integration step, i.e. the length of each integration interval.
	Length dx =
		DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);
	// Integration loop.
	Length result = 0.0 * m;
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		Length d_i = Number(i) * dx;
		// Distance between the current sample point and the planet center.
		Length r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
		// Number density at the current sample point (divided by the number density
		// at the bottom of the atmosphere, yielding a dimensionless number).
		Number y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);
		// Sample weight (from the trapezoidal rule).
		Number weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5 : 1.0;
		result += y_i * weight_i * dx;
	}
	return result;
}

__device__ __forceinline__ DimensionlessSpectrum atmosphere::ComputeTransmittanceToTopAtmosphereBoundary(
	IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	return exp(-(
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

__device__ __forceinline__ Number atmosphere::GetTextureCoordFromUnitRange(Number x, int texture_size) {
	return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
}

__device__ __forceinline__ Number atmosphere::GetUnitRangeFromTextureCoord(Number u, int texture_size) {
	return (u - 0.5 / Number(texture_size)) / (1.0 - 1.0 / Number(texture_size));
}

__device__ __forceinline__ vec2 atmosphere::GetTransmittanceTextureUvFromRMu(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	Length rho =
		SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
	// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
	Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
	Length d_min = atmosphere.top_radius - r;
	Length d_max = rho + H;
	Number x_mu = (d - d_min) / (d_max - d_min);
	Number x_r = rho / H;
	return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
		GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

__device__ __forceinline__ void atmosphere::GetRMuFromTransmittanceTextureUv(IN(AtmosphereParameters) atmosphere,
	IN(vec2) uv, OUT(Length) r, OUT(Number) mu) {
	assert(uv.x >= 0.0 && uv.x <= 1.0);
	assert(uv.y >= 0.0 && uv.y <= 1.0);
	Number x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
	Number x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon, from which we can compute r:
	Length rho = H * x_r;
	r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
	// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
	// from which we can recover mu:
	Length d_min = atmosphere.top_radius - r;
	Length d_max = rho + H;
	Length d = d_min + x_mu * (d_max - d_min);
	mu = d == 0.0 * m ? Number(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
	mu = ClampCosine(mu);
}

__device__ __forceinline__ DimensionlessSpectrum atmosphere::ComputeTransmittanceToTopAtmosphereBoundaryTexture(
	IN(AtmosphereParameters) atmosphere, IN(vec2) frag_coord) {
	const vec2 TRANSMITTANCE_TEXTURE_SIZE =
		vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
	Length r;
	Number mu;
	GetRMuFromTransmittanceTextureUv(
		atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
	return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}


__device__ __forceinline__ DimensionlessSpectrum atmosphere::GetTransmittanceToTopAtmosphereBoundary(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	vec2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
	return DimensionlessSpectrum(texture(transmittance_texture, uv));
}


__device__ __forceinline__ DimensionlessSpectrum atmosphere::GetTransmittance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	assert(d >= 0.0 * m);

	Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_d = ClampCosine((r * mu + d) / r_d);

	if (ray_r_mu_intersects_ground) {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, -mu_d) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, -mu),
			DimensionlessSpectrum(1.0));
	}
	else {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, mu) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, mu_d),
			DimensionlessSpectrum(1.0));
	}
}

__device__ __forceinline__ DimensionlessSpectrum atmosphere::GetTransmittanceToSun(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu_s) {
	Number sin_theta_h = atmosphere.bottom_radius / r;
	Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
	return GetTransmittanceToTopAtmosphereBoundary(
		atmosphere, transmittance_texture, r, mu_s) *
		smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
			sin_theta_h * atmosphere.sun_angular_radius / rad,
			mu_s - cos_theta_h);
}

__device__ __forceinline__ void atmosphere::ComputeSingleScatteringIntegrand(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu, Number mu_s, Number nu, Length d,
	bool ray_r_mu_intersects_ground,
	OUT(DimensionlessSpectrum) rayleigh, OUT(DimensionlessSpectrum) mie) {
	Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
	DimensionlessSpectrum transmittance =
		GetTransmittance(
			atmosphere, transmittance_texture, r, mu, d,
			ray_r_mu_intersects_ground) *
		GetTransmittanceToSun(
			atmosphere, transmittance_texture, r_d, mu_s_d);
	rayleigh = transmittance * GetProfileDensity(
		atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
	mie = transmittance * GetProfileDensity(
		atmosphere.mie_density, r_d - atmosphere.bottom_radius);
}

__device__ __forceinline__ Length atmosphere::DistanceToNearestAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu, bool ray_r_mu_intersects_ground) {
	if (ray_r_mu_intersects_ground) {
		return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
	}
	else {
		return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
	}
}

__device__ __forceinline__ void atmosphere::ComputeSingleScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	assert(nu >= -1.0 && nu <= 1.0);

	// Number of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the length of each integration interval.
	Length dx =
		DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,
			ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);
	// Integration loop.
	DimensionlessSpectrum rayleigh_sum = DimensionlessSpectrum(0.0);
	DimensionlessSpectrum mie_sum = DimensionlessSpectrum(0.0);
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		Length d_i = Number(i) * dx;
		// The Rayleigh and Mie single scattering at the current sample point.
		DimensionlessSpectrum rayleigh_i;
		DimensionlessSpectrum mie_i;
		ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture,
			r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
		// Sample weight (from the trapezoidal rule).
		Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_sum += rayleigh_i * weight_i;
		mie_sum += mie_i * weight_i;
	}
	rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *
		atmosphere.rayleigh_scattering;
	mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
}


__device__ __forceinline__ InverseSolidAngle atmosphere::RayleighPhaseFunction(Number nu) {
	InverseSolidAngle k = 3.0 / (16.0 * PI * sr);
	return k * (1.0 + nu * nu);
}

__device__ __forceinline__ InverseSolidAngle atmosphere::MiePhaseFunction(Number g, Number nu) {
	InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

__device__ __forceinline__ vec4 atmosphere::GetScatteringTextureUvwzFromRMuMuSNu(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	assert(nu >= -1.0 && nu <= 1.0);

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	Length rho =
		SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
	Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

	// Discriminant of the quadratic equation for the intersections of the ray
	// (r,mu) with the ground (see RayIntersectsGround).
	Length r_mu = r * mu;
	Area discriminant =
		r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
	Number u_mu;
	if (ray_r_mu_intersects_ground) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon).
		Length d = -r_mu - SafeSqrt(discriminant);
		Length d_min = r - atmosphere.bottom_radius;
		Length d_max = rho;
		u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon).
		Length d = -r_mu + SafeSqrt(discriminant + H * H);
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}

	Length d = DistanceToTopAtmosphereBoundary(
		atmosphere, atmosphere.bottom_radius, mu_s);
	Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	Length d_max = H;
	Number a = (d - d_min) / (d_max - d_min);
	Number A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	Number u_mu_s = GetTextureCoordFromUnitRange(
		max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

	Number u_nu = (nu + 1.0) / 2.0;
	return vec4(u_nu, u_mu_s, u_mu, u_r);
}

__device__ __forceinline__ void atmosphere::GetRMuMuSNuFromScatteringTextureUvwz(IN(AtmosphereParameters) atmosphere,
	IN(vec4) uvwz, OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s,
	OUT(Number) nu, OUT(bool) ray_r_mu_intersects_ground) {
	assert(uvwz.x >= 0.0 && uvwz.x <= 1.0);
	assert(uvwz.y >= 0.0 && uvwz.y <= 1.0);
	assert(uvwz.z >= 0.0 && uvwz.z <= 1.0);
	assert(uvwz.w >= 0.0 && uvwz.w <= 1.0);

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	Length rho =
		H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
	r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

	if (uvwz.z < 0.5) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
		// we can recover mu:
		Length d_min = r - atmosphere.bottom_radius;
		Length d_max = rho;
		Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? Number(-1.0) :
			ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = true;
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon) - from which we can recover mu:
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? Number(1.0) :
			ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = false;
	}

	Number x_mu_s =
		GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
	Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	Length d_max = H;
	Number A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
	Length d = d_min + min(a, A) * (d_max - d_min);
	mu_s = d == 0.0 * m ? Number(1.0) :
		ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

	nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}


__device__ __forceinline__ void atmosphere::GetRMuMuSNuFromScatteringTextureFragCoord(
	IN(AtmosphereParameters) atmosphere, IN(vec3) frag_coord,
	OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s, OUT(Number) nu,
	OUT(bool) ray_r_mu_intersects_ground) {
	const vec4 SCATTERING_TEXTURE_SIZE = vec4(
		SCATTERING_TEXTURE_NU_SIZE - 1,
		SCATTERING_TEXTURE_MU_S_SIZE,
		SCATTERING_TEXTURE_MU_SIZE,
		SCATTERING_TEXTURE_R_SIZE);
	Number frag_coord_nu =
		floor(frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));
	Number frag_coord_mu_s =
		mod(frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));
	vec4 uvwz =
		vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
		SCATTERING_TEXTURE_SIZE;
	GetRMuMuSNuFromScatteringTextureUvwz(
		atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	// Clamp nu to its valid range of values, given mu and mu_s.
	nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
		mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

__device__ __forceinline__ void atmosphere::ComputeSingleScatteringTexture(IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture, IN(vec3) frag_coord,
	OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {
	Length r;
	Number mu;
	Number mu_s;
	Number nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	ComputeSingleScattering(atmosphere, transmittance_texture,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
}


TEMPLATE(AbstractSpectrum)
__device__ __forceinline__ AbstractSpectrum atmosphere::GetScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(AbstractScatteringTexture TEMPLATE_ARGUMENT(AbstractSpectrum))
	scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {
	vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
		atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
	Number tex_x = floor(tex_coord_x);
	Number lerp = tex_coord_x - tex_x;
	vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	return AbstractSpectrum(texture(scattering_texture, uvw0) * (1.0 - lerp) +
		texture(scattering_texture, uvw1) * lerp);
}


__device__ __forceinline__ RadianceSpectrum atmosphere::GetScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	int scattering_order) {
	if (scattering_order == 1) {
		IrradianceSpectrum rayleigh = GetScattering(
			atmosphere, single_rayleigh_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
		IrradianceSpectrum mie = GetScattering(
			atmosphere, single_mie_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
		return rayleigh * RayleighPhaseFunction(nu) +
			mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
	}
	else {
		return GetScattering(
			atmosphere, multiple_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
	}
}

__device__ __forceinline__ IrradianceSpectrum atmosphere::GetIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu_s);


__device__ __forceinline__ RadianceDensitySpectrum atmosphere::ComputeScatteringDensity(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu, Number mu_s, Number nu, int scattering_order) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	assert(nu >= -1.0 && nu <= 1.0);
	assert(scattering_order >= 2);

	// Compute unit direction vectors for the zenith, the view direction omega and
	// and the sun direction omega_s, such that the cosine of the view-zenith
	// angle is mu, the cosine of the sun-zenith angle is mu_s, and the cosine of
	// the view-sun angle is nu. The goal is to simplify computations below.
	vec3 zenith_direction = vec3(0.0, 0.0, 1.0);
	vec3 omega = vec3(sqrt(1.0 - mu * mu), 0.0, mu);
	Number sun_dir_x = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
	Number sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
	vec3 omega_s = vec3(sun_dir_x, sun_dir_y, mu_s);

	const int SAMPLE_COUNT = 16;
	const Angle dphi = pi / Number(SAMPLE_COUNT);
	const Angle dtheta = pi / Number(SAMPLE_COUNT);
	RadianceDensitySpectrum rayleigh_mie =
		RadianceDensitySpectrum(0.0 * watt_per_cubic_meter_per_sr_per_nm);

	// Nested loops for the integral over all the incident directions omega_i.
	for (int l = 0; l < SAMPLE_COUNT; ++l) {
		Angle theta = (Number(l) + 0.5) * dtheta;
		Number cos_theta = cos(theta);
		Number sin_theta = sin(theta);
		bool ray_r_theta_intersects_ground =
			RayIntersectsGround(atmosphere, r, cos_theta);

		// The distance and transmittance to the ground only depend on theta, so we
		// can compute them in the outer loop for efficiency.
		Length distance_to_ground = 0.0 * m;
		DimensionlessSpectrum transmittance_to_ground = DimensionlessSpectrum(0.0);
		DimensionlessSpectrum ground_albedo = DimensionlessSpectrum(0.0);
		if (ray_r_theta_intersects_ground) {
			distance_to_ground =
				DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);
			transmittance_to_ground =
				GetTransmittance(atmosphere, transmittance_texture, r, cos_theta,
					distance_to_ground, true /* ray_intersects_ground */);
			ground_albedo = atmosphere.ground_albedo;
		}

		for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {
			Angle phi = (Number(m) + 0.5) * dphi;
			vec3 omega_i =
				vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
			SolidAngle domega_i = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			// The radiance L_i arriving from direction omega_i after n-1 bounces is
			// the sum of a term given by the precomputed scattering texture for the
			// (n-1)-th order:
			Number nu1 = dot(omega_s, omega_i);
			RadianceSpectrum incident_radiance = GetScattering(atmosphere,
				single_rayleigh_scattering_texture, single_mie_scattering_texture,
				multiple_scattering_texture, r, omega_i.z, mu_s, nu1,
				ray_r_theta_intersects_ground, scattering_order - 1);

			// and of the contribution from the light paths with n-1 bounces and whose
			// last bounce is on the ground. This contribution is the product of the
			// transmittance to the ground, the ground albedo, the ground BRDF, and
			// the irradiance received on the ground after n-2 bounces.
			vec3 ground_normal =
				normalize(zenith_direction * r + omega_i * distance_to_ground);
			IrradianceSpectrum ground_irradiance = GetIrradiance(
				atmosphere, irradiance_texture, atmosphere.bottom_radius,
				dot(ground_normal, omega_s));
			incident_radiance += transmittance_to_ground *
				ground_albedo * (1.0 / (PI * sr)) * ground_irradiance;

			// The radiance finally scattered from direction omega_i towards direction
			// -omega is the product of the incident radiance, the scattering
			// coefficient, and the phase function for directions omega and omega_i
			// (all this summed over all particle types, i.e. Rayleigh and Mie).
			Number nu2 = dot(omega, omega_i);
			Number rayleigh_density = GetProfileDensity(
				atmosphere.rayleigh_density, r - atmosphere.bottom_radius);
			Number mie_density = GetProfileDensity(
				atmosphere.mie_density, r - atmosphere.bottom_radius);
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


__device__ __forceinline__ RadianceSpectrum atmosphere::ComputeMultipleScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ScatteringDensityTexture) scattering_density_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu >= -1.0 && mu <= 1.0);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	assert(nu >= -1.0 && nu <= 1.0);

	// Number of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the length of each integration interval.
	Length dx =
		DistanceToNearestAtmosphereBoundary(
			atmosphere, r, mu, ray_r_mu_intersects_ground) /
		Number(SAMPLE_COUNT);
	// Integration loop.
	RadianceSpectrum rayleigh_mie_sum =
		RadianceSpectrum(0.0 * watt_per_square_meter_per_sr_per_nm);
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		Length d_i = Number(i) * dx;

		// The r, mu and mu_s parameters at the current integration point (see the
		// single scattering section for a detailed explanation).
		Length r_i =
			ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
		Number mu_i = ClampCosine((r * mu + d_i) / r_i);
		Number mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

		// The Rayleigh and Mie multiple scattering at the current sample point.
		RadianceSpectrum rayleigh_mie_i =
			GetScattering(
				atmosphere, scattering_density_texture, r_i, mu_i, mu_s_i, nu,
				ray_r_mu_intersects_ground) *
			GetTransmittance(
				atmosphere, transmittance_texture, r, mu, d_i,
				ray_r_mu_intersects_ground) *
			dx;
		// Sample weight (from the trapezoidal rule).
		Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_mie_sum += rayleigh_mie_i * weight_i;
	}
	return rayleigh_mie_sum;
}


__device__ __forceinline__ RadianceDensitySpectrum atmosphere::ComputeScatteringDensityTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(IrradianceTexture) irradiance_texture,
	IN(vec3) frag_coord, int scattering_order) {
	Length r;
	Number mu;
	Number mu_s;
	Number nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeScatteringDensity(atmosphere, transmittance_texture,
		single_rayleigh_scattering_texture, single_mie_scattering_texture,
		multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu,
		scattering_order);
}

__device__ __forceinline__ RadianceSpectrum atmosphere::ComputeMultipleScatteringTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ScatteringDensityTexture) scattering_density_texture,
	IN(vec3) frag_coord, OUT(Number) nu) {
	Length r;
	Number mu;
	Number mu_s;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeMultipleScattering(atmosphere, transmittance_texture,
		scattering_density_texture, r, mu, mu_s, nu,
		ray_r_mu_intersects_ground);
}

__device__ __forceinline__ IrradianceSpectrum atmosphere::ComputeDirectIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu_s) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu_s >= -1.0 && mu_s <= 1.0);

	Number alpha_s = atmosphere.sun_angular_radius / rad;
	// Approximate average of the cosine factor mu_s over the visible fraction of
	// the Sun disc.
	Number average_cosine_factor =
		mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :
		(mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

	return atmosphere.solar_irradiance *
		GetTransmittanceToTopAtmosphereBoundary(
			atmosphere, transmittance_texture, r, mu_s) * average_cosine_factor;

}

__device__ __forceinline__ IrradianceSpectrum atmosphere::ComputeIndirectIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	Length r, Number mu_s, int scattering_order) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	assert(scattering_order >= 1);

	const int SAMPLE_COUNT = 32;
	const Angle dphi = pi / Number(SAMPLE_COUNT);
	const Angle dtheta = pi / Number(SAMPLE_COUNT);

	IrradianceSpectrum result =
		IrradianceSpectrum(0.0 * watt_per_square_meter_per_nm);
	vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
	for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
		Angle theta = (Number(j) + 0.5) * dtheta;
		for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
			Angle phi = (Number(i) + 0.5) * dphi;
			vec3 omega =
				vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
			SolidAngle domega = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			Number nu = dot(omega, omega_s);
			result += GetScattering(atmosphere, single_rayleigh_scattering_texture,
				single_mie_scattering_texture, multiple_scattering_texture,
				r, omega.z, mu_s, nu, false /* ray_r_theta_intersects_ground */,
				scattering_order) *
				omega.z * domega;
		}
	}
	return result;
}


__device__ __forceinline__ vec2 atmosphere::GetIrradianceTextureUvFromRMuS(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu_s) {
	assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
	assert(mu_s >= -1.0 && mu_s <= 1.0);
	Number x_r = (r - atmosphere.bottom_radius) /
		(atmosphere.top_radius - atmosphere.bottom_radius);
	Number x_mu_s = mu_s * 0.5 + 0.5;
	return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
		GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

__device__ __forceinline__ void atmosphere::GetRMuSFromIrradianceTextureUv(IN(AtmosphereParameters) atmosphere,
	IN(vec2) uv, OUT(Length) r, OUT(Number) mu_s) {
	assert(uv.x >= 0.0 && uv.x <= 1.0);
	assert(uv.y >= 0.0 && uv.y <= 1.0);
	Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
	Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
	r = atmosphere.bottom_radius +
		x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
	mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

const vec2 IRRADIANCE_TEXTURE_SIZE = vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

__device__ __forceinline__ IrradianceSpectrum atmosphere::ComputeDirectIrradianceTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(vec2) frag_coord) {
	Length r;
	Number mu_s;
	GetRMuSFromIrradianceTextureUv(
		atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
	return ComputeDirectIrradiance(atmosphere, transmittance_texture, r, mu_s);
}

__device__ __forceinline__ IrradianceSpectrum atmosphere::ComputeIndirectIrradianceTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(vec2) frag_coord, int scattering_order) {
	Length r;
	Number mu_s;
	GetRMuSFromIrradianceTextureUv(
		atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
	return ComputeIndirectIrradiance(atmosphere,
		single_rayleigh_scattering_texture, single_mie_scattering_texture,
		multiple_scattering_texture, r, mu_s, scattering_order);
}


__device__ __forceinline__ IrradianceSpectrum atmosphere::GetIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu_s) {
	vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
	return IrradianceSpectrum(texture(irradiance_texture, uv));
}

#ifdef COMBINED_SCATTERING_TEXTURES
__device__ __forceinline__ vec3 atmosphere::GetExtrapolatedSingleMieScattering(
	IN(AtmosphereParameters) atmosphere, IN(vec4) scattering) {
	if (scattering.r == 0.0) {
		return vec3(0.0);
	}
	return scattering.rgb * scattering.a / scattering.r *
		(atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
		(atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}
#endif


__device__ __forceinline__ IrradianceSpectrum atmosphere::GetCombinedScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	OUT(IrradianceSpectrum) single_mie_scattering) {
	vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
		atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
	Number tex_x = floor(tex_coord_x);
	Number lerp = tex_coord_x - tex_x;
	vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
#ifdef COMBINED_SCATTERING_TEXTURES
	vec4 combined_scattering =
		texture(scattering_texture, uvw0) * (1.0 - lerp) +
		texture(scattering_texture, uvw1) * lerp;
	IrradianceSpectrum scattering = IrradianceSpectrum(combined_scattering);
	single_mie_scattering =
		GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
#else
	IrradianceSpectrum scattering = IrradianceSpectrum(
		texture(scattering_texture, uvw0) * (1.0 - lerp) +
		texture(scattering_texture, uvw1) * lerp);
	single_mie_scattering = IrradianceSpectrum(
		texture(single_mie_scattering_texture, uvw0) * (1.0 - lerp) +
		texture(single_mie_scattering_texture, uvw1) * lerp);
#endif
	return scattering;
}

__device__ __forceinline__ RadianceSpectrum atmosphere::GetSkyRadiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Position camera, IN(Direction) view_ray, Length shadow_length,
	IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	Length r = length(camera);
	Length rmu = dot(camera, view_ray);
	Length distance_to_top_atmosphere_boundary = -rmu -
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
		transmittance = DimensionlessSpectrum(1.0);
		return RadianceSpectrum(0.0 * watt_per_square_meter_per_sr_per_nm);
	}
	// Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
	Number mu = rmu / r;
	Number mu_s = dot(camera, sun_direction) / r;
	Number nu = dot(view_ray, sun_direction);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = ray_r_mu_intersects_ground ? DimensionlessSpectrum(0.0) :
		GetTransmittanceToTopAtmosphereBoundary(
			atmosphere, transmittance_texture, r, mu);
	IrradianceSpectrum single_mie_scattering;
	IrradianceSpectrum scattering;
	if (shadow_length == 0.0 * m) {
		scattering = GetCombinedScattering(
			atmosphere, scattering_texture, single_mie_scattering_texture,
			r, mu, mu_s, nu, ray_r_mu_intersects_ground,
			single_mie_scattering);
	}
	else {
		// Case of light shafts (shadow_length is the total length noted l in our
		// paper): we omit the scattering between the camera and the point at
		// distance l, by implementing Eq. (18) of the paper (shadow_transmittance
		// is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
		Length d = shadow_length;
		Length r_p =
			ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
		Number mu_p = (r * mu + d) / r_p;
		Number mu_s_p = (r * mu_s + d * nu) / r_p;

		scattering = GetCombinedScattering(
			atmosphere, scattering_texture, single_mie_scattering_texture,
			r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
			single_mie_scattering);
		DimensionlessSpectrum shadow_transmittance =
			GetTransmittance(atmosphere, transmittance_texture,
				r, mu, shadow_length, ray_r_mu_intersects_ground);
		scattering = scattering * shadow_transmittance;
		single_mie_scattering = single_mie_scattering * shadow_transmittance;
	}
	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}


__device__ __forceinline__ RadianceSpectrum atmosphere::GetSkyRadianceToPoint(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Position camera, IN(Position) point, Length shadow_length,
	IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	Direction view_ray = normalize(point - camera);
	Length r = length(camera);
	Length rmu = dot(camera, view_ray);
	Length distance_to_top_atmosphere_boundary = -rmu -
		sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}

	// Compute the r, mu, mu_s and nu parameters for the first texture lookup.
	Number mu = rmu / r;
	Number mu_s = dot(camera, sun_direction) / r;
	Number nu = dot(view_ray, sun_direction);
	Length d = length(point - camera);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = GetTransmittance(atmosphere, transmittance_texture,
		r, mu, d, ray_r_mu_intersects_ground);

	IrradianceSpectrum single_mie_scattering;
	IrradianceSpectrum scattering = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground,
		single_mie_scattering);

	// Compute the r, mu, mu_s and nu parameters for the second texture lookup.
	// If shadow_length is not 0 (case of light shafts), we want to ignore the
	// scattering along the last shadow_length meters of the view ray, which we
	// do by subtracting shadow_length from d (this way scattering_p is equal to
	// the S|x_s=x_0-lv term in Eq. (17) of our paper).
	d = max(d - shadow_length, 0.0 * m);
	Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_p = (r * mu + d) / r_p;
	Number mu_s_p = (r * mu_s + d * nu) / r_p;

	IrradianceSpectrum single_mie_scattering_p;
	IrradianceSpectrum scattering_p = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
		single_mie_scattering_p);

	// Combine the lookup results to get the scattering between camera and point.
	DimensionlessSpectrum shadow_transmittance = transmittance;
	if (shadow_length > 0.0 * m) {
		// This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
		shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
			r, mu, d, ray_r_mu_intersects_ground);
	}
	scattering = scattering - shadow_transmittance * scattering_p;
	single_mie_scattering =
		single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
	single_mie_scattering = GetExtrapolatedSingleMieScattering(
		atmosphere, vec4(scattering, single_mie_scattering.r));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering *
		smoothstep(Number(0.0), Number(0.01), mu_s);

	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}


__device__ __forceinline__ IrradianceSpectrum atmosphere::GetSunAndSkyIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(IrradianceTexture) irradiance_texture,
	IN(Position) point, IN(Direction) normal, IN(Direction) sun_direction,
	OUT(IrradianceSpectrum) sky_irradiance) {
	Length r = length(point);
	Number mu_s = dot(point, sun_direction) / r;

	// Indirect irradiance (approximated if the surface is not horizontal).
	sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *
		(1.0 + dot(normal, point) / r) * 0.5;

	// Direct irradiance.
	return atmosphere.solar_irradiance *
		GetTransmittanceToSun(
			atmosphere, transmittance_texture, r, mu_s) *
		max(dot(normal, sun_direction), 0.0);
}
