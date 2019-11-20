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
//	Version 1.0: Sergen Eren, 11/11/2019
//
// File: This is the header file for atmosphere class that implements 
//		 bruneton model sky in cuda. This file contains device side function 
//		 definitions and host side function declerations
//
//-----------------------------------------------


#ifndef  __ATMOSPHERE_H__
#define __ATMOSPHERE_H__


#include <cuda.h>
#include "texture_types.h"
#include <vector>

#include "atmosphere/definitions.h"
#include "atmosphere/constants.h"

enum atmosphere_error_t {

	ATMO_INIT_ERR,
	ATMO_INIT_FUNC_ERR,
	ATMO_RECOMPUTE_ERR,
	ATMO_FILL_TEX_ERR,
	ATMO_LAUNCH_ERR,
	ATMO_NO_ERR

};

constexpr double kPi = 3.1415926;


constexpr double kSolarIrradiance[48] = {
  1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
  1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
  1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
  1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
  1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
  1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
};

constexpr double kOzoneCrossSection[48] = {
1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
};

constexpr double kDobsonUnit = 2.687e20;
constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
constexpr double kConstantSolarIrradiance = 1.5;
constexpr double kRayleigh = 1.24062e-6;
constexpr double kRayleighScaleHeight = 8000.0f;
constexpr double kMieScaleHeight = 1200.0f;
constexpr double kMieAngstromAlpha = 0.0;
constexpr double kMieAngstromBeta = 5.328e-3;
constexpr double kMieSingleScatteringAlbedo = 0.9;
constexpr double kGroundAlbedo = 0.01;

static double kDefaultLuminanceFromRadiance[] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
static double kDefaultLambdas[] = { 680.0, 550.0, 440.0 };

class atmosphere {


private:

	#define kLambdaR 680.0
	#define kLambdaG 550.0
	#define kLambdaB 440.0

	#define kLambdaMin 360
	#define kLambdaMax 830
	

public:
	
	atmosphere();
	~atmosphere();

	atmosphere_error_t init();
	atmosphere_error_t precompute(double* lambdas, double* luminance_from_radiance, bool blend, int num_scattering_orders);
	atmosphere_error_t recompute();
private:
	atmosphere_error_t clear_buffers();
	void update_model(const float3 lambdas);
	void copy_transmittance_texture();
	void copy_scattering_texture();
	void copy_irradiance_texture();
	void copy_single_scattering_texture();

	atmosphere_error_t init_functions(CUmodule &cuda_module);
	atmosphere_error_t compute_transmittance(double* lambdas, double* luminance_from_radiance, bool blend, int num_scattering_orders);
	DensityProfile adjust_units(DensityProfile density);
	void print_texture(float3 *buffer ,const char* filename, const int width, const int height);
	void print_texture(float4 * buffer, const char * filename, const int width, const int height);
	double coeff(double lambda, int component);
	void sky_sun_radiance_to_luminance(float3& sky_spectral_radiance_to_luminance, float3& sun_spectral_radiance_to_luminance);
	static double cie_color_matching_function_table_value(double wavelength, int column);
	static double interpolate(const std::vector<double>& wavelengths, const std::vector<double>& wavelength_function, double wavelength);
	static void compute_spectral_radiance_to_luminance_factors(const std::vector<double>& wavelengths, const std::vector<double>& solar_irradiance, double lambda_power, double& k_r, double& k_g, double& k_b);

private:

	std::vector<double> m_wave_lengths;
	std::vector<double> m_solar_irradiance;
	
	double m_sun_angular_radius;
	double m_bottom_radius;
	double m_top_radius;
	
	DensityProfileLayer* m_rayleigh_density;
	std::vector<double> m_rayleigh_scattering;
	
	DensityProfileLayer* m_mie_density;
	std::vector<double> m_mie_scattering;
	std::vector<double> m_mie_extinction;
	double m_mie_phase_function_g;
	
	std::vector<DensityProfileLayer*> m_absorption_density;
	std::vector<double> m_absorption_extinction;
	
	std::vector<double> m_ground_albedo;

	double sun_k_r, sun_k_g, sun_k_b;
	double sky_k_r, sky_k_g, sky_k_b;

	double m_max_sun_zenith_angle;
	double m_length_unit_in_meters;
	
	inline int num_precomputed_wavelengths() { return m_use_luminance == LUMINANCE::PRECOMPUTED ? 15 : 3; }
	bool m_combine_scattering_textures;
	bool m_half_precision = false;
	
public:
	bool m_use_constant_solar_spectrum = true;
	bool m_use_ozone = true;
	LUMINANCE m_use_luminance;
	AtmosphereParameters atmosphere_parameters;

private:

	CUmodule atmosphere_module;

	CUfunction transmittance_function;
	CUfunction direct_irradiance_function;
	CUfunction indirect_irradiance_function;
	CUfunction multiple_scattering_function;
	CUfunction scattering_density_function;
	CUfunction single_scattering_function;
	// Buffer cleaner functions 
	CUfunction clear_transmittance_buffers_function;
	CUfunction clear_irradiance_buffers_function;
	CUfunction clear_scattering_buffers_function;
};

#endif // ! __ATMOSPHERE_H__
