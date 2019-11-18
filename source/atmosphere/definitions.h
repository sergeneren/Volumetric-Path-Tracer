

#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__


#define ALIGN(x)	__align__(x)

__device__ const float m = 1.0;
__device__ const float nm = 1.0;
__device__ const float rad = 1.0;
__device__ const float sr = 1.0;
__device__ const float watt = 1.0;
__device__ const float lm = 1.0;

__device__ const float PI = 3.14159265358979323846f;

__device__ constexpr float km() { return 1000.0 * m; }
__device__ constexpr float m2() { return m * m; }
__device__ constexpr float m3() { return m * m * m; }
__device__ constexpr float pi() { return PI / rad; }
__device__ constexpr float deg() { return pi() / 180.0f; }
__device__ constexpr float watt_per_square_meter() { return watt / m2() ; }
__device__ constexpr float watt_per_square_meter_per_sr() { return watt / (m2() * sr); }
__device__ constexpr float watt_per_square_meter_per_nm() { return watt / (m2() * nm); }
__device__ constexpr float watt_per_square_meter_per_sr_per_nm() { return watt / (m2() * sr * nm); }
__device__ constexpr float watt_per_cubic_meter_per_sr_per_nm() { return watt / (m3() * sr * nm); }
__device__ constexpr float cd() { return lm / sr; }
__device__ constexpr float kcd() { return 1000.0f * cd(); }
__device__ constexpr float cd_per_square_meter() { return cd() / m2(); }
__device__ constexpr float kcd_per_square_meter() { return kcd() / m2(); }

struct ALIGN(16) DensityProfileLayer {

	__device__ __host__ DensityProfileLayer() : DensityProfileLayer(.0f, .0f, .0f, .0f, .0f) {}
	__device__ __host__ DensityProfileLayer(float width, float exp_term, float exp_scale,
		float linear_term, float const_term)
		: width(width), exp_term(exp_term), exp_scale(exp_scale), linear_term(linear_term), const_term(const_term) {}

	float width;
	float exp_term;
	float exp_scale;
	float linear_term;
	float const_term;
};

struct ALIGN(16) DensityProfile {

	DensityProfileLayer layers[2];

};

struct ALIGN(16) AtmosphereParameters {

	float3 sky_spectral_radiance_to_luminance;
	float3 sun_spectral_radiance_to_luminance;

	float3 solar_irradiance;
	float angle;
	float bottom_radius;
	float top_radius;

	DensityProfile rayleigh_density;
	float3 rayleigh_scattering;

	DensityProfile mie_density;
	float3 mie_scattering;
	float3 mie_extinction;
	float mie_phase_function_g;

	DensityProfile absorption_density;
	float3 absorption_extinction;

	float3 ground_albedo;
	float sun_angular_radius;
	float mu_s_min;

	// Buffers
	
	float4 *delta_irradience_buffer;
	float4 *delta_rayleigh_scattering_buffer;
	float4 *delta_mie_scattering_buffer;
	float4 *delta_scattering_density_buffer;
	float4 *delta_multiple_scattering_buffer;
	float4 *transmittance_buffer;
	float4 *irradiance_buffer;	
	float4 *scattering_buffer;
	float4 *optional_mie_single_scattering_buffer;
	
	// Textures 

	cudaTextureObject_t transmittance_texture;
	cudaTextureObject_t scattering_texture;
	cudaTextureObject_t irradiance_texture;
	cudaTextureObject_t single_mie_scattering_texture;



};

#endif