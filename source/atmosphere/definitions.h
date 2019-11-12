

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

const Length m = 1.0;
const Wavelength nm = 1.0;
const Angle rad = 1.0;
const SolidAngle sr = 1.0;
const Power watt = 1.0;
const LuminousPower lm = 1.0;

/*
<p>From which we can derive the units for some derived physical quantities,
as well as some derived units (kilometer km, kilocandela kcd, degree deg):
*/

const float PI = 3.14159265358979323846;

const Length km = 1000.0 * m;
const Area m2 = m * m;
const Volume m3 = m * m * m;
const Angle pi = PI * rad;
const Angle deg = pi / 180.0;
const Irradiance watt_per_square_meter = watt / m2;
const Radiance watt_per_square_meter_per_sr = watt / (m2 * sr);
const SpectralIrradiance watt_per_square_meter_per_nm = watt / (m2 * nm);
const SpectralRadiance watt_per_square_meter_per_sr_per_nm =
watt / (m2 * sr * nm);
const SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm =
watt / (m3 * sr * nm);
const LuminousIntensity cd = lm / sr;
const LuminousIntensity kcd = 1000.0 * cd;
const Luminance cd_per_square_meter = cd / m2;
const Luminance kcd_per_square_meter = kcd / m2;

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

	float mu_s_min;

};



#endif