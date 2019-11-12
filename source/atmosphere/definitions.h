

#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__


#define ALIGN(x)	__align__(x)

#define Length float
#define Wavelength float
#define Angle float
#define SolidAngle float
#define Power float
#define LuminousPower float

/*
<p>From this we "derive" the irradiance, radiance, spectral irradiance,
spectral radiance, luminance, etc, as well pure numbers, area, volume, etc (the
actual derivation is done in the <a href="reference/definitions.h.html">C++
equivalent</a> of this file).
*/

#define Number float
#define InverseLength float
#define Area float
#define Volume float
#define NumberDensity float
#define Irradiance float
#define Radiance float
#define SpectralPower float
#define SpectralIrradiance float
#define SpectralRadiance float
#define SpectralRadianceDensity float
#define ScatteringCoefficient float
#define InverseSolidAngle float
#define LuminousIntensity float
#define Luminance float
#define Illuminance float

// A generic function from Wavelength to some other type.
#define AbstractSpectrum float3
// A function from Wavelength to Number.
#define DimensionlessSpectrum float3
// A function from Wavelength to SpectralPower.
#define PowerSpectrum float3
// A function from Wavelength to SpectralIrradiance.
#define IrradianceSpectrum float3
// A function from Wavelength to SpectralRadiance.
#define RadianceSpectrum float3
// A function from Wavelength to SpectralRadianceDensity.
#define RadianceDensitySpectrum float3
// A function from Wavelength to ScaterringCoefficient.
#define ScatteringSpectrum float3

// A position in 3D (3 length values).
#define Position float3
// A unit direction vector in 3D (3 unitless values).
#define Direction float3
// A vector of 3 luminance values.
#define Luminance3 float3
// A vector of 3 illuminance values.
#define Illuminance3 float3

/*
<h3>Physical units</h3>

<p>We can then define the units for our six base physical quantities:
meter (m), nanometer (nm), radian (rad), steradian (sr), watt (watt) and lumen
(lm):
*/

__device__ const Length m = 1.0;
__device__ const Wavelength nm = 1.0;
__device__ const Angle rad = 1.0;
__device__ const SolidAngle sr = 1.0;
__device__ const Power watt = 1.0;
__device__ const LuminousPower lm = 1.0;

/*
<p>From which we can derive the units for some derived physical quantities,
as well as some derived units (kilometer km, kilocandela kcd, degree deg):
*/

__device__ const float PI = 3.14159265358979323846;

__device__ constexpr Length km() { return 1000.0 * m; }
__device__ constexpr Area m2() { return m * m; }
__device__ constexpr Volume m3() { return m * m * m; }
__device__ constexpr Angle pi() { return PI / rad; }
__device__ constexpr Angle deg() { return pi() / 180.0; }
__device__ constexpr Irradiance watt_per_square_meter() { return watt / m2() ; }
__device__ constexpr Radiance watt_per_square_meter_per_sr() { return watt / (m2() * sr); }
__device__ constexpr SpectralIrradiance watt_per_square_meter_per_nm() { return watt / (m2() * nm); }
__device__ constexpr SpectralRadiance watt_per_square_meter_per_sr_per_nm() { return watt / (m2() * sr * nm); }
__device__ constexpr SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm() { return watt / (m3() * sr * nm); }
__device__ constexpr LuminousIntensity cd() { return lm / sr; }
__device__ constexpr LuminousIntensity kcd() { return 1000.0 * cd(); }
__device__ constexpr Luminance cd_per_square_meter() { return cd() / m2(); }
__device__ constexpr Luminance kcd_per_square_meter() { return kcd() / m2(); }

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

};



#endif