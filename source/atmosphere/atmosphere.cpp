
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
//	Version 1.0: Sergen Eren, 12/11/2019
//
// File: This is the implementation file for atmosphere class functions 
//
//-----------------------------------------------


#include <vector>
#include <string>

#include "atmosphere/atmosphere.h"
#include "atmosphere/constants.h"


// Functions that hold the texture calculation kernels from atmosphere_kernels.ptx file
atmosphere_error_t atmosphere::init_functions(CUmodule &cuda_module) {

	CUresult error;
	error = cuModuleGetFunction(transmittance_texture_function, cuda_module, "fill_transmittance_buffer");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;
	
	error = cuModuleGetFunction(scattering_texture_function, cuda_module, "fill_scattering_buffer");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;

	error = cuModuleGetFunction(irradiance_texture_function, cuda_module, "fill_irradiance_buffer");
	if (error != CUDA_SUCCESS) return ATMO_INIT_FUNC_ERR;
	
	return ATMO_NO_ERR;

}

// Initialization function that fills the atmosphere parameters 
atmosphere_error_t atmosphere::init(bool use_constant_solar_spectrum_, bool use_ozone_) {

	constexpr double kPi = 3.1415926;
	constexpr double kSunAngularRadius = 0.00935 / 2.0;
	constexpr double kSunSolidAngle = kPi * kSunAngularRadius * kSunAngularRadius;
	constexpr double kLengthUnitInMeters = 1000.0;
	
	constexpr int kLambdaMin = 360;
	constexpr int kLambdaMax = 830;
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
	constexpr double kBottomRadius = 6360000.0;
	constexpr double kTopRadius = 6420000.0;
	constexpr double kRayleigh = 1.24062e-6;
	constexpr double kRayleighScaleHeight = 8000.0;
	constexpr double kMieScaleHeight = 1200.0;
	constexpr double kMieAngstromAlpha = 0.0;
	constexpr double kMieAngstromBeta = 5.328e-3;
	constexpr double kMieSingleScatteringAlbedo = 0.9;
	constexpr double kMiePhaseFunctionG = 0.8;
	constexpr double kGroundAlbedo = 0.1;
	const double max_sun_zenith_angle =	 102.0 / 180.0 * kPi;

	DensityProfileLayer rayleigh_layer(0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
	DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);
	
	std::vector<DensityProfileLayer> ozone_density;
	ozone_density.push_back(
		DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
	ozone_density.push_back(
		DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));

	std::vector<double> wavelengths;
	std::vector<double> solar_irradiance;
	std::vector<double> rayleigh_scattering;
	std::vector<double> mie_scattering;
	std::vector<double> mie_extinction;
	std::vector<double> absorption_extinction;
	std::vector<double> ground_albedo;


	for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
		double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
		double mie =
			kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
		wavelengths.push_back(l);
		if (use_constant_solar_spectrum_) {
			solar_irradiance.push_back(kConstantSolarIrradiance);
		}
		else {
			solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
		}
		rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
		mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
		mie_extinction.push_back(mie);
		absorption_extinction.push_back(use_ozone_ ?
			kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] :
			0.0);
		ground_albedo.push_back(kGroundAlbedo);
	}





}





atmosphere::~atmosphere() {

	transmittance_texture_function = nullptr;
	scattering_texture_function = nullptr;
	irradiance_texture_function = nullptr;

}

atmosphere::atmosphere() {

	transmittance_texture_function = new CUfunction;
	scattering_texture_function = new CUfunction;
	irradiance_texture_function = new CUfunction;
}