
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

#define DEBUG_TEXTURES

#include <vector>
#include <string>
#include <fstream>


#ifdef DEBUG_TEXTURES
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // DEBUG_TEXTURES

#include "atmosphere/atmosphere.h"
#include "atmosphere/constants.h"

#include "matrix_math.h"
#include <driver_types.h>
#include <helper_cuda.h>
#include <helper_math.h>

// TO-DO convert all device pointers to thrust device pointers 
//#include "thrust/device_ptr.h"

// Functions that hold the texture calculation kernels from atmosphere_kernels.ptx file
atmosphere_error_t atmosphere::init_functions(CUmodule &cuda_module) {

	CUresult error;
	error = cuModuleGetFunction(&transmittance_function, cuda_module, "calculate_transmittance");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind transmittance function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&direct_irradiance_function, cuda_module, "calculate_direct_irradiance");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind direct irradiance function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&indirect_irradiance_function, cuda_module, "calculate_indirect_irradiance");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind indirect irradiance function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&multiple_scattering_function, cuda_module, "calculate_multiple_scattering");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind multiple scattering function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&scattering_density_function, cuda_module, "calculate_scattering_density");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind scattering density function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&single_scattering_function, cuda_module, "calculate_single_scattering");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind single scattering function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	// Buffer cleaner functions 

	error = cuModuleGetFunction(&clear_transmittance_buffers_function, cuda_module, "clear_transmittance_buffers");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind clear_transmittance_buffers_function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&clear_irradiance_buffers_function, cuda_module, "clear_irradiance_buffers");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind clear_irradiance_buffers_function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&clear_scattering_buffers_function, cuda_module, "clear_scattering_buffers");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind clear_scattering_buffers_function!\n");
		return ATMO_INIT_FUNC_ERR;
	}

	return ATMO_NO_ERR;

}

double atmosphere::coeff(double lambda, int component) {

	double x = cie_color_matching_function_table_value(lambda, 1);
	double y = cie_color_matching_function_table_value(lambda, 2);
	double z = cie_color_matching_function_table_value(lambda, 3);
	double sRGB = XYZ_TO_SRGB[component * 3 + 0] * x + XYZ_TO_SRGB[component * 3 + 1] * y + XYZ_TO_SRGB[component * 3 + 2] * z;

	return sRGB;

}

void atmosphere::sky_sun_radiance_to_luminance(float3& sky_spectral_radiance_to_luminance, float3& sun_spectral_radiance_to_luminance) {

	bool precompute_illuminance = num_precomputed_wavelengths() > 3;
	double sky_k_r, sky_k_g, sky_k_b;

	if (precompute_illuminance)
		sky_k_r = sky_k_g = sky_k_b = static_cast<double>(MAX_LUMINOUS_EFFICACY);
	else
		compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, -3, sky_k_r, sky_k_g, sky_k_b);

	// Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant.
	double sun_k_r, sun_k_g, sun_k_b;
	compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, 0, sun_k_r, sun_k_g, sun_k_b);

	sky_spectral_radiance_to_luminance = make_float3((float)sky_k_r, (float)sky_k_g, (float)sky_k_b);
	sun_spectral_radiance_to_luminance = make_float3((float)sun_k_r, (float)sun_k_g, (float)sun_k_b);

}

double atmosphere::cie_color_matching_function_table_value(double wavelength, int column) {

	if (wavelength <= kLambdaMin || wavelength >= kLambdaMax)
		return 0.0;

	double u = (wavelength - kLambdaMin) / 5.0;
	int row = (int)floor(u);

	u -= row;
	return CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row + column] * (1.0 - u) + CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1) + column] * u;

}

double atmosphere::interpolate(const std::vector<double>& wavelengths, const std::vector<double>& wavelength_function, double wavelength)
{
	if (wavelength < wavelengths[0])
		return wavelength_function[0];

	for (int i = 0; i < wavelengths.size() - 1; ++i)
	{
		if (wavelength < wavelengths[i + 1])
		{
			double u = (wavelength - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i]);
			return wavelength_function[i] * (1.0 - u) + wavelength_function[i + 1] * u;
		}
	}

	return wavelength_function[wavelength_function.size() - 1];
}

void atmosphere::compute_spectral_radiance_to_luminance_factors(const std::vector<double>& wavelengths, const std::vector<double>& solar_irradiance, double lambda_power, double & k_r, double & k_g, double & k_b)
{
	k_r = 0.0;
	k_g = 0.0;
	k_b = 0.0;
	double solar_r = interpolate(wavelengths, solar_irradiance, kLambdaR);
	double solar_g = interpolate(wavelengths, solar_irradiance, kLambdaG);
	double solar_b = interpolate(wavelengths, solar_irradiance, kLambdaB);
	int dlambda = 1;

	for (int lambda = kLambdaMin; lambda < kLambdaMax; lambda += dlambda)
	{
		double x_bar = cie_color_matching_function_table_value(lambda, 1);
		double y_bar = cie_color_matching_function_table_value(lambda, 2);
		double z_bar = cie_color_matching_function_table_value(lambda, 3);

		const double* xyz2srgb = &XYZ_TO_SRGB[0];
		double r_bar = xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
		double g_bar = xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
		double b_bar = xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
		double irradiance = interpolate(wavelengths, solar_irradiance, lambda);

		k_r += r_bar * irradiance / solar_r * pow(lambda / kLambdaR, lambda_power);
		k_g += g_bar * irradiance / solar_g * pow(lambda / kLambdaG, lambda_power);
		k_b += b_bar * irradiance / solar_b * pow(lambda / kLambdaB, lambda_power);
	}

	k_r *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;
	k_g *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;
	k_b *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;

}

DensityProfile atmosphere::adjust_units(DensityProfile density) {
	density.layers[0].width /= m_length_unit_in_meters;
	density.layers[0].exp_scale *= m_length_unit_in_meters;
	density.layers[0].linear_term *= m_length_unit_in_meters;
	density.layers[1].width /= m_length_unit_in_meters;
	density.layers[1].exp_scale *= m_length_unit_in_meters;
	density.layers[1].linear_term *= m_length_unit_in_meters;
	return density;
}

void atmosphere::print_texture(float3 * buffer, const char * filename, const int width, const int height)
{
#ifdef DEBUG_TEXTURES

	unsigned char *data = new unsigned char[width*height * 3];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = min(max(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = min(max(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = min(max(0, unsigned char(buffer[index].z * 255)), 255);

		}
	}
	stbi_flip_vertically_on_write(1);
	stbi_write_jpg(filename, width, height, 3, (void*)data, 100);
#endif
}
void atmosphere::print_texture(float4 * buffer, const char * filename, const int width, const int height)
{
#ifdef DEBUG_TEXTURES
	unsigned char *data = new unsigned char[width*height * 4];

	int idx = 0;
	for (int i = 0; i < width; ++i) {
		for (int y = 0; y < height; ++y) {

			int index = i * height + y;
			data[idx++] = min(max(0, unsigned char(buffer[index].x * 255)), 255);
			data[idx++] = min(max(0, unsigned char(buffer[index].y * 255)), 255);
			data[idx++] = min(max(0, unsigned char(buffer[index].z * 255)), 255);
			//data[idx++] = min(max(0, unsigned char(buffer[index].w * 255)), 255);
			data[idx++] = 255;
		}
	}
	stbi_flip_vertically_on_write(1);
	stbi_write_png(filename, width, height, 4, (void*)data, 0);
#endif
}

// Clear buffers for next round of precomputation
atmosphere_error_t atmosphere::clear_buffers() {

	CUresult result;

	// Clear transmittance buffers 
	dim3 block(8, 8, 1);
	dim3 grid_transmittance(int(TRANSMITTANCE_TEXTURE_WIDTH / block.x) + 1, int(TRANSMITTANCE_TEXTURE_HEIGHT / block.y) + 1, 1);

	void *transmittance_params[] = {&atmosphere_parameters};
	result = cuLaunchKernel(clear_transmittance_buffers_function, grid_transmittance.x, grid_transmittance.y, 1, block.x, block.y, 1, 0, NULL, transmittance_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch clear_transmittance_buffers_function! \n");
		return ATMO_LAUNCH_ERR;
	}
#ifdef DEBUG_TEXTURES // Print transmittance values

	int transmittance_size = TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT * sizeof(float4);
	float4 *host_transmittance_buffer = new float4[TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT];

	checkCudaErrors(cudaMemcpy(host_transmittance_buffer, atmosphere_parameters.transmittance_buffer, transmittance_size, cudaMemcpyDeviceToHost));

	print_texture(host_transmittance_buffer, "transmittance_cleared.png", TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
	delete[] host_transmittance_buffer;
#endif


	// Clear irradiance buffers 
	dim3 grid_irradiance(int(IRRADIANCE_TEXTURE_WIDTH / block.x) + 1, int(IRRADIANCE_TEXTURE_HEIGHT / block.y) + 1, 1);
	void *irradiance_params[] = {&atmosphere_parameters};

	result = cuLaunchKernel(clear_irradiance_buffers_function, grid_irradiance.x, grid_irradiance.y, 1, block.x, block.y, 1, 0, NULL, irradiance_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch clear_irradiance_buffers_function! \n");
		return ATMO_LAUNCH_ERR;
	}

	// Clear scattering buffers 
	dim3 block_sct(8, 8, 8);
	dim3 grid_scattering(int(SCATTERING_TEXTURE_WIDTH / block_sct.x) + 1, int(SCATTERING_TEXTURE_HEIGHT / block_sct.y) + 1, int(SCATTERING_TEXTURE_DEPTH / block_sct.z) + 1);
	
	void *single_scattering_params[] = { &atmosphere_parameters};

	result = cuLaunchKernel(clear_scattering_buffers_function, grid_scattering.x, grid_scattering.y, grid_scattering.z, block_sct.x, block_sct.y, block_sct.z, 0, NULL, single_scattering_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch clear_scattering_buffers_function! \n");
		return ATMO_LAUNCH_ERR;
	}

	return ATMO_NO_ERR;
}

// Copy transmitance buffer to transmittance texture
void atmosphere::copy_transmittance_texture() {

	// Destroy texture if it's been created before 
	if (atmosphere_parameters.transmittance_texture) checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.transmittance_texture));
	
	cudaArray_t data_array = 0;

	const int rx = TRANSMITTANCE_TEXTURE_WIDTH;
	const int ry = TRANSMITTANCE_TEXTURE_HEIGHT;

	float4 *host_transmittance_buffer = new float4[rx*ry];
	checkCudaErrors(cudaMemcpy(host_transmittance_buffer, atmosphere_parameters.transmittance_buffer, rx*ry * sizeof(float4), cudaMemcpyDeviceToHost));
	
	const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMallocArray(&data_array, &channel_desc, rx, ry));
	checkCudaErrors(cudaMemcpy2DToArray(data_array, 0, 0, host_transmittance_buffer, rx * sizeof(float4), rx * sizeof(float4), ry, cudaMemcpyHostToDevice));
	
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data_array;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeWrap;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	checkCudaErrors(cudaCreateTextureObject(&atmosphere_parameters.transmittance_texture, &res_desc, &tex_desc, NULL));
	delete[] host_transmittance_buffer;
}

// Copy irradiance buffer to irradiance texture
void atmosphere::copy_irradiance_texture() {

	// Destroy texture if it's been created before 
	if (atmosphere_parameters.irradiance_texture)  checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.irradiance_texture));
	
	cudaArray_t data_array = 0;

	const int rx = IRRADIANCE_TEXTURE_WIDTH;
	const int ry = IRRADIANCE_TEXTURE_HEIGHT;

	float4 *host_buffer = new float4[rx*ry];
	checkCudaErrors(cudaMemcpy(host_buffer, atmosphere_parameters.irradiance_buffer, rx*ry * sizeof(float4), cudaMemcpyDeviceToHost));
	
	const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMallocArray(&data_array, &channel_desc, rx, ry));

	checkCudaErrors(cudaMemcpy2DToArray(data_array, 0, 0, host_buffer, rx * sizeof(float4), rx * sizeof(float4), ry, cudaMemcpyHostToDevice));
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data_array;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeWrap;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	checkCudaErrors(cudaCreateTextureObject(&atmosphere_parameters.irradiance_texture, &res_desc, &tex_desc, NULL));
	delete[] host_buffer;
}

// Copy scattering buffer to scattering texture
void atmosphere::copy_scattering_texture() {

	if (atmosphere_parameters.scattering_texture) checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.scattering_texture));

	cudaArray_t data_array = 0;

	const int rx = SCATTERING_TEXTURE_WIDTH;
	const int ry = SCATTERING_TEXTURE_HEIGHT;
	const int rz = SCATTERING_TEXTURE_DEPTH;

	cudaExtent extent;
	extent.width = rx;
	extent.height = ry;
	extent.depth = rz;

	float4 *volume_data_host = (float4 *)malloc(rx * ry * rz * sizeof(float4));
	checkCudaErrors(cudaMemcpy(volume_data_host, atmosphere_parameters.scattering_buffer, rx*ry*rz * sizeof(float4), cudaMemcpyDeviceToHost));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&data_array, &channelDesc, extent));

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(volume_data_host, extent.width * sizeof(float4), extent.width, extent.height);
	copyParams.dstArray = data_array;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = data_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	texDescr.normalizedCoords = 1;

	checkCudaErrors(cudaCreateTextureObject(&atmosphere_parameters.scattering_texture, &texRes, &texDescr, NULL));

	delete[] volume_data_host;
}

// Copy single scattering buffer to single scattering texture
void atmosphere::copy_single_scattering_texture() {

	if (atmosphere_parameters.single_mie_scattering_texture) checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.single_mie_scattering_texture));

	cudaArray_t data_array = 0;

	const int rx = SCATTERING_TEXTURE_WIDTH;
	const int ry = SCATTERING_TEXTURE_HEIGHT;
	const int rz = SCATTERING_TEXTURE_DEPTH;

	cudaExtent extent;
	extent.width = rx;
	extent.height = ry;
	extent.depth = rz;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&data_array, &channelDesc, extent));

	float4 *volume_data_host = (float4 *)malloc(rx * ry * rz * sizeof(float4));
	checkCudaErrors(cudaMemcpy(volume_data_host, atmosphere_parameters.optional_mie_single_scattering_buffer, rx*ry*rz * sizeof(float4), cudaMemcpyDeviceToHost));

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(volume_data_host, extent.width * sizeof(float4), extent.width, extent.height);
	copyParams.dstArray = data_array;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = data_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	texDescr.normalizedCoords = 1;

	checkCudaErrors(cudaCreateTextureObject(&atmosphere_parameters.single_mie_scattering_texture, &texRes, &texDescr, NULL));

	delete[] volume_data_host;
}

// Updates atmosphere_parameters by internal parameters 
void atmosphere::update_model(const float3 lambdas) {

	atmosphere_parameters.sky_spectral_radiance_to_luminance = make_float3(sky_k_r, sky_k_g, sky_k_b);
	atmosphere_parameters.sun_spectral_radiance_to_luminance = make_float3(sun_k_r, sun_k_g, sun_k_b);

	atmosphere_parameters.solar_irradiance.x = interpolate(m_wave_lengths, m_solar_irradiance, lambdas.x);
	atmosphere_parameters.solar_irradiance.y = interpolate(m_wave_lengths, m_solar_irradiance, lambdas.y);
	atmosphere_parameters.solar_irradiance.z = interpolate(m_wave_lengths, m_solar_irradiance, lambdas.z);


	atmosphere_parameters.sun_angular_radius = m_sun_angular_radius;
	atmosphere_parameters.bottom_radius = m_bottom_radius / m_length_unit_in_meters;
	atmosphere_parameters.top_radius = m_top_radius / m_length_unit_in_meters;

	DensityProfile rayleigh_density;
	rayleigh_density.layers[0] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
	rayleigh_density.layers[1] = { 0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0 };
	atmosphere_parameters.rayleigh_density = adjust_units(rayleigh_density);
	atmosphere_parameters.rayleigh_scattering.x = interpolate(m_wave_lengths, m_rayleigh_scattering, lambdas.x) * m_length_unit_in_meters;
	atmosphere_parameters.rayleigh_scattering.y = interpolate(m_wave_lengths, m_rayleigh_scattering, lambdas.y) * m_length_unit_in_meters;
	atmosphere_parameters.rayleigh_scattering.z = interpolate(m_wave_lengths, m_rayleigh_scattering, lambdas.z) * m_length_unit_in_meters;

	DensityProfile mie_density;
	mie_density.layers[0] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
	mie_density.layers[1] = { 0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0 };
	atmosphere_parameters.mie_density = adjust_units(mie_density);
	atmosphere_parameters.mie_scattering.x = interpolate(m_wave_lengths, m_mie_scattering, lambdas.x) * m_length_unit_in_meters;
	atmosphere_parameters.mie_scattering.y = interpolate(m_wave_lengths, m_mie_scattering, lambdas.y) * m_length_unit_in_meters;
	atmosphere_parameters.mie_scattering.z = interpolate(m_wave_lengths, m_mie_scattering, lambdas.z) * m_length_unit_in_meters;
	atmosphere_parameters.mie_extinction.x = interpolate(m_wave_lengths, m_mie_scattering, lambdas.x) * m_length_unit_in_meters;
	atmosphere_parameters.mie_extinction.y = interpolate(m_wave_lengths, m_mie_scattering, lambdas.y) * m_length_unit_in_meters;
	atmosphere_parameters.mie_extinction.z = interpolate(m_wave_lengths, m_mie_scattering, lambdas.z) * m_length_unit_in_meters;
	atmosphere_parameters.mie_phase_function_g = m_mie_phase_function_g;


	DensityProfile ozone_density;
	ozone_density.layers[0] = { 25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0 };
	ozone_density.layers[1] = { 0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0 };
	atmosphere_parameters.absorption_density = adjust_units(ozone_density);

	atmosphere_parameters.absorption_extinction.x = interpolate(m_wave_lengths, m_absorption_extinction, lambdas.x) * m_length_unit_in_meters;
	atmosphere_parameters.absorption_extinction.y = interpolate(m_wave_lengths, m_absorption_extinction, lambdas.y) * m_length_unit_in_meters;
	atmosphere_parameters.absorption_extinction.z = interpolate(m_wave_lengths, m_absorption_extinction, lambdas.z) * m_length_unit_in_meters;

	atmosphere_parameters.ground_albedo.x = interpolate(m_wave_lengths, m_ground_albedo, lambdas.x);
	atmosphere_parameters.ground_albedo.y = interpolate(m_wave_lengths, m_ground_albedo, lambdas.y);
	atmosphere_parameters.ground_albedo.z = interpolate(m_wave_lengths, m_ground_albedo, lambdas.z);

	const double max_sun_zenith_angle = (m_half_precision ? 102.0 : 120.0) / 180.0 * kPi;
	atmosphere_parameters.mu_s_min = cos(max_sun_zenith_angle);

}

// Recomputes the textures if there is a change
atmosphere_error_t atmosphere::recompute() {

	// clear vectors 
	
	m_wave_lengths.clear();
	m_solar_irradiance.clear();
	m_rayleigh_density = nullptr;
	m_rayleigh_scattering.clear();
	m_mie_density = nullptr;
	m_mie_scattering.clear();
	m_mie_extinction.clear();

	m_absorption_density.clear();
	m_absorption_extinction.clear();
	m_ground_albedo.clear();

	m_absorption_density.push_back(new DensityProfileLayer(25000.0f, 0.0f, 0.0f, 1.0f / 15000.0f, -2.0f / 3.0f));
	m_absorption_density.push_back(new DensityProfileLayer(0.0f, 0.0f, 0.0f, -1.0f / 15000.0f, 8.0f / 3.0f));

	for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
		double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
		double mie = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
		m_wave_lengths.push_back(l);
		if (m_use_constant_solar_spectrum) {
			m_solar_irradiance.push_back(kConstantSolarIrradiance);
		}
		else {
			m_solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
		}
		m_rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
		m_mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
		m_mie_extinction.push_back(mie);
		m_absorption_extinction.push_back(m_use_ozone ? kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
		m_ground_albedo.push_back(kGroundAlbedo);
	}

	m_half_precision = false;
	m_combine_scattering_textures = true;
	m_sun_angular_radius = 0.00935 / 2.0;
	m_bottom_radius = 6360000.0f;
	m_top_radius = 6420000.0f;
	m_rayleigh_density = new DensityProfileLayer(0.0f, 1.0f, -1.0f / float(kRayleighScaleHeight), 0.0f, 0.0f);
	m_mie_density = new DensityProfileLayer(0.0f, 1.0f, -1.0f / float(kMieScaleHeight), 0.0f, 0.0f);
	m_mie_phase_function_g = 0.8;
	m_max_sun_zenith_angle = 102.0 / 180.0 * kPi;
	m_length_unit_in_meters = 1000.0f;

	int num_scattering_orders = 4;
	// Start precomputation

	if (num_precomputed_wavelengths() <= 3) {
		atmosphere_error_t error = precompute(nullptr, nullptr, false, num_scattering_orders);
		if (error != ATMO_NO_ERR) {
			printf("Unable to precompute!");
			return ATMO_INIT_ERR;
		}
	}
	else {
		int num_iterations = (num_precomputed_wavelengths() + 2) / 3;
		double dlambda = (kLambdaMax - kLambdaMin) / (3.0 * num_iterations);

		for (int i = 0; i < num_iterations; ++i)
		{
			double lambdas[] =
			{
					kLambdaMin + (3 * i + 0.5) * dlambda,
					kLambdaMin + (3 * i + 1.5) * dlambda,
					kLambdaMin + (3 * i + 2.5) * dlambda
			};

			double luminance_from_radiance[] =
			{
					coeff(lambdas[0], 0) * dlambda, coeff(lambdas[1], 0) * dlambda, coeff(lambdas[2], 0) * dlambda,
					coeff(lambdas[0], 1) * dlambda, coeff(lambdas[1], 1) * dlambda, coeff(lambdas[2], 1) * dlambda,
					coeff(lambdas[0], 2) * dlambda, coeff(lambdas[1], 2) * dlambda, coeff(lambdas[2], 2) * dlambda
			};

			bool blend = i > 0;
			atmosphere_error_t error = precompute(lambdas, luminance_from_radiance, blend, num_scattering_orders);
			if (error != ATMO_NO_ERR) {
				printf("Unable to precompute!");
				return ATMO_INIT_ERR;
			}
		}

		// After the above iterations, the transmittance texture contains the
		// transmittance for the 3 wavelengths used at the last iteration. But we
		// want the transmittance at kLambdaR, kLambdaG, kLambdaB instead, so we
		// must recompute it here for these 3 wavelengths:
		atmosphere_error_t error = compute_transmittance(nullptr, nullptr, false, num_scattering_orders);
		if (error != ATMO_NO_ERR) {
			printf("Unable to precompute!");
			return ATMO_INIT_ERR;
		}
	}
	
	// copy textures and free buffers

	copy_transmittance_texture();
	copy_irradiance_texture();
	copy_scattering_texture();
	copy_single_scattering_texture();

	clear_buffers();

	return ATMO_NO_ERR;

}

// Precomputes the textures that will be sent to the render kernel
atmosphere_error_t atmosphere::precompute(double* lambda_ptr, double* luminance_from_radiance, bool blend, int num_scattering_orders) {

	float3 lambdas;
	int BLEND = blend ? 1 : 0;

	mat4 lfrm; // luminance_from_radiance_matrix

	if (lambda_ptr == nullptr)	lambdas = make_float3(kDefaultLambdas[0], kDefaultLambdas[1], kDefaultLambdas[2]);
	else lambdas = make_float3(lambda_ptr[0], lambda_ptr[1], lambda_ptr[2]);
	if (luminance_from_radiance == nullptr) luminance_from_radiance = kDefaultLuminanceFromRadiance;

	lfrm = lfrm.toMatrix(luminance_from_radiance);

	if (m_use_luminance == PRECOMPUTED) {
		sky_k_r = sky_k_g = sky_k_b = MAX_LUMINOUS_EFFICACY;
	}
	else {
		compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, -3 /* lambda_power */, sky_k_r, sky_k_g, sky_k_b);
	}

	compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, 0, sun_k_r, sun_k_g, sun_k_b);

	update_model(lambdas);

	// STARTING PRECOMPUTE

	// Precompute transmittance 
	//***************************************************************************************************************************

	CUresult result;

	dim3 block(8, 8, 1);
	dim3 grid_transmittance(int(TRANSMITTANCE_TEXTURE_WIDTH / block.x) + 1, int(TRANSMITTANCE_TEXTURE_HEIGHT / block.y) + 1, 1);

	void *transmittance_params[] = { &atmosphere_parameters };
	result = cuLaunchKernel(transmittance_function, grid_transmittance.x, grid_transmittance.y, 1, block.x, block.y, 1, 0, NULL, transmittance_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch direct transmittance function! \n");
		return ATMO_LAUNCH_ERR;
	}

#ifdef DEBUG_TEXTURES // Print transmittance values

	int transmittance_size = TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT * sizeof(float4);
	float4 *host_transmittance_buffer = new float4[TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT];

	checkCudaErrors(cudaMemcpy(host_transmittance_buffer, atmosphere_parameters.transmittance_buffer, transmittance_size, cudaMemcpyDeviceToHost));

	print_texture(host_transmittance_buffer, "transmittance.png", TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
	delete[] host_transmittance_buffer;
#endif

	// Compute direct irradiance 
	//***************************************************************************************************************************
	dim3 grid_irradiance(int(IRRADIANCE_TEXTURE_WIDTH / block.x) + 1, int(IRRADIANCE_TEXTURE_HEIGHT / block.y) + 1, 1);
	void *irradiance_params[] = { &atmosphere_parameters, (void*)&BLEND };

	result = cuLaunchKernel(direct_irradiance_function, grid_irradiance.x, grid_irradiance.y, 1, block.x, block.y, 1, 0, NULL, irradiance_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch direct irradiance function! \n");
		return ATMO_LAUNCH_ERR;
	}
#ifdef DEBUG_TEXTURES // Print irradiance values

	int irradiance_size = IRRADIANCE_TEXTURE_WIDTH * IRRADIANCE_TEXTURE_HEIGHT * sizeof(float4);
	float4 *host_irradiance_buffer = new float4[IRRADIANCE_TEXTURE_WIDTH * IRRADIANCE_TEXTURE_HEIGHT];

	checkCudaErrors(cudaMemcpy(host_irradiance_buffer, atmosphere_parameters.delta_irradience_buffer, irradiance_size, cudaMemcpyDeviceToHost));
	print_texture(host_irradiance_buffer, "irradiance.png", IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
	
#endif

	//***************************************************************************************************************************

	// Compute single scattering
	//***************************************************************************************************************************
	dim3 block_sct(8, 8, 8);
	dim3 grid_scattering(int(SCATTERING_TEXTURE_WIDTH / block_sct.x) + 1, int(SCATTERING_TEXTURE_HEIGHT / block_sct.y) + 1, int(SCATTERING_TEXTURE_DEPTH / block_sct.z) + 1);
	

	float4 blend_vec = make_float4(.0f, .0f, BLEND, BLEND);

	void *single_scattering_params[] = { &atmosphere_parameters, &blend_vec, &lfrm };

	result = cuLaunchKernel(single_scattering_function, grid_scattering.x, grid_scattering.y, grid_scattering.z, block_sct.x, block_sct.y, block_sct.z, 0, NULL, single_scattering_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch direct single scattering function! \n");
		return ATMO_LAUNCH_ERR;
	}
#ifdef DEBUG_TEXTURES // Print single scattering values

	int scattering_size = SCATTERING_TEXTURE_WIDTH * SCATTERING_TEXTURE_HEIGHT * SCATTERING_TEXTURE_DEPTH * sizeof(float4);
	float4 *host_scattering_buffer = new float4[SCATTERING_TEXTURE_WIDTH * SCATTERING_TEXTURE_HEIGHT * SCATTERING_TEXTURE_DEPTH];

	checkCudaErrors(cudaMemcpy(host_scattering_buffer, atmosphere_parameters.scattering_buffer, scattering_size, cudaMemcpyDeviceToHost));
	print_texture(host_scattering_buffer, "scattering.png", SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

	checkCudaErrors(cudaMemcpy(host_scattering_buffer, atmosphere_parameters.delta_rayleigh_scattering_buffer, scattering_size, cudaMemcpyDeviceToHost));
	print_texture(host_scattering_buffer, "delta_rayleigh_scattering.png", SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

	checkCudaErrors(cudaMemcpy(host_scattering_buffer, atmosphere_parameters.delta_mie_scattering_buffer, scattering_size, cudaMemcpyDeviceToHost));
	print_texture(host_scattering_buffer, "delta_mie_scattering.png", SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

	

#endif

	//***************************************************************************************************************************

	// Compute the 2nd, 3rd and 4th order of scattering, in sequence.
	for (int scattering_order = 2; scattering_order <= num_scattering_orders; ++scattering_order)
	{

		// Compute scattering density
		//***************************************************************************************************************************

		blend_vec = make_float4(.0f);

		void *scattering_density_params[] = { &atmosphere_parameters, &blend_vec, &lfrm, &scattering_order };
		result = cuLaunchKernel(scattering_density_function, grid_scattering.x, grid_scattering.y, grid_scattering.z, block_sct.x, block_sct.y, block_sct.z, 0, NULL, scattering_density_params, NULL);
		checkCudaErrors(cudaDeviceSynchronize());
		if (result != CUDA_SUCCESS) {
			printf("Unable to launch direct scattering density function! \n");
			return ATMO_LAUNCH_ERR;
		}

#ifdef DEBUG_TEXTURES // Print single scattering values

		checkCudaErrors(cudaMemcpy(host_scattering_buffer, atmosphere_parameters.delta_scattering_density_buffer, scattering_size, cudaMemcpyDeviceToHost));
		std::string name("scattering_density_");
		name.append(std::to_string(scattering_order));
		name.append(".png");

		print_texture(host_scattering_buffer, name.c_str(), SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

#endif

		// Compute indirect irradiance
		//***************************************************************************************************************************

		blend_vec = make_float4(.0f, 1.0f, .0f, .0f);

		void *indirect_irradiance_params[] = { &atmosphere_parameters, &blend_vec, &lfrm, &scattering_order };

		result = cuLaunchKernel(indirect_irradiance_function, grid_irradiance.x, grid_irradiance.y, 1, block.x, block.y, 1, 0, NULL, indirect_irradiance_params, NULL);
		checkCudaErrors(cudaDeviceSynchronize());
		if (result != CUDA_SUCCESS) {
			printf("Unable to launch direct irradiance function! \n");
			return ATMO_LAUNCH_ERR;
		}
#ifdef DEBUG_TEXTURES // Print indirect irradiance values

		checkCudaErrors(cudaMemcpy(host_irradiance_buffer, atmosphere_parameters.delta_irradience_buffer, irradiance_size, cudaMemcpyDeviceToHost));
		std::string name_indirect("delta_irradiance_");
		name_indirect.append(std::to_string(scattering_order));
		name_indirect.append(".png");
		print_texture(host_irradiance_buffer, name_indirect.c_str(), IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

#endif

		// Compute multiple scattering
		//***************************************************************************************************************************

		
		void *multiple_scattering_params[] = { &atmosphere_parameters, &blend_vec, &lfrm, &scattering_order };
		result = cuLaunchKernel(multiple_scattering_function, grid_scattering.x, grid_scattering.y, grid_scattering.z, block_sct.x, block_sct.y, block_sct.z, 0, NULL, multiple_scattering_params, NULL);
		checkCudaErrors(cudaDeviceSynchronize());
		if (result != CUDA_SUCCESS) {
			printf("Unable to launch direct scattering density function! \n");
			return ATMO_LAUNCH_ERR;
		}

#ifdef DEBUG_TEXTURES // Print multiple scattering values

		checkCudaErrors(cudaMemcpy(host_scattering_buffer, atmosphere_parameters.delta_multiple_scattering_buffer, scattering_size, cudaMemcpyDeviceToHost));
		std::string name_multi("multiple_scattering_");
		name_multi.append(std::to_string(scattering_order));
		name_multi.append(".png");

		print_texture(host_scattering_buffer, name_multi.c_str(), SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

#endif


	}
	delete[] host_irradiance_buffer;
	delete[] host_scattering_buffer;

	return ATMO_NO_ERR;

}

// Precomputes the textures that will be sent to the render kernel
atmosphere_error_t atmosphere::compute_transmittance(double* lambda_ptr, double* luminance_from_radiance, bool blend, int num_scattering_orders) {

	float3 lambdas;
	int BLEND = blend ? 1 : 0;

	mat4 lfrm; // luminance_from_radiance_matrix

	if (lambda_ptr == nullptr)	lambdas = make_float3(kDefaultLambdas[0], kDefaultLambdas[1], kDefaultLambdas[2]);
	else lambdas = make_float3(lambda_ptr[0], lambda_ptr[1], lambda_ptr[2]);
	if (luminance_from_radiance == nullptr) luminance_from_radiance = kDefaultLuminanceFromRadiance;

	lfrm = lfrm.toMatrix(luminance_from_radiance);

	if (m_use_luminance == PRECOMPUTED) {
		sky_k_r = sky_k_g = sky_k_b = MAX_LUMINOUS_EFFICACY;
	}
	else {
		compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, -3 /* lambda_power */, sky_k_r, sky_k_g, sky_k_b);
	}

	compute_spectral_radiance_to_luminance_factors(m_wave_lengths, m_solar_irradiance, 0, sun_k_r, sun_k_g, sun_k_b);

	update_model(lambdas);


	// Precompute transmittance 
	//***************************************************************************************************************************

	CUresult result;

	dim3 block(8, 8, 1);
	dim3 grid_transmittance(int(TRANSMITTANCE_TEXTURE_WIDTH / block.x) + 1, int(TRANSMITTANCE_TEXTURE_HEIGHT / block.y) + 1, 1);
	int transmittance_size = TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT * sizeof(float4);

	void *transmittance_params[] = { &atmosphere_parameters };
	result = cuLaunchKernel(transmittance_function, grid_transmittance.x, grid_transmittance.y, 1, block.x, block.y, 1, 0, NULL, transmittance_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch direct transmittance function! \n");
		return ATMO_LAUNCH_ERR;
	}

#ifdef DEBUG_TEXTURES // Print transmittance values
	float4 *host_transmittance_buffer = new float4[TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT];

	checkCudaErrors(cudaMemcpy(host_transmittance_buffer, atmosphere_parameters.transmittance_buffer, transmittance_size, cudaMemcpyDeviceToHost));

	print_texture(host_transmittance_buffer, "transmittance.jpg", TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
	delete[] host_transmittance_buffer;
#endif


	return ATMO_NO_ERR;

}

// Initialization function that fills the atmosphere parameters 
atmosphere_error_t atmosphere::init()
{
	   
	// Bind precomputation functions from ptx file 
	CUresult error = cuModuleLoad(&atmosphere_module, "atmosphere_kernels.ptx");
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleLoad, %i\n", error);

	init_functions(atmosphere_module);

	m_absorption_density.push_back(new DensityProfileLayer(25000.0f, 0.0f, 0.0f, 1.0f / 15000.0f, -2.0f / 3.0f));
	m_absorption_density.push_back(new DensityProfileLayer(0.0f, 0.0f, 0.0f, -1.0f / 15000.0f, 8.0f / 3.0f));

	for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
		double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
		double mie = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
		m_wave_lengths.push_back(l);
		if (m_use_constant_solar_spectrum) {
			m_solar_irradiance.push_back(kConstantSolarIrradiance);
		}
		else {
			m_solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
		}
		m_rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
		m_mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
		m_mie_extinction.push_back(mie);
		m_absorption_extinction.push_back(m_use_ozone ? kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
		m_ground_albedo.push_back(kGroundAlbedo);
	}

	m_half_precision = false;
	m_combine_scattering_textures = true;
	m_sun_angular_radius = 0.00935 / 2.0;
	m_bottom_radius = 6360000.0f;
	m_top_radius = 6420000.0f;
	m_rayleigh_density = new DensityProfileLayer(0.0f, 1.0f, -1.0f / float(kRayleighScaleHeight), 0.0f, 0.0f);
	m_mie_density = new DensityProfileLayer(0.0f, 1.0f, -1.0f / float(kMieScaleHeight), 0.0f, 0.0f);
	m_mie_phase_function_g = 0.8;
	m_max_sun_zenith_angle = 102.0 / 180.0 * kPi;
	m_length_unit_in_meters = 1000.0f;

	int num_scattering_orders = 4;
	// Start precomputation

	if (num_precomputed_wavelengths() <= 3) {
		atmosphere_error_t error = precompute(nullptr, nullptr, false, num_scattering_orders);
		if (error != ATMO_NO_ERR) {
			printf("Unable to precompute!");
			return ATMO_INIT_ERR;
		}
	}
	else {

		int num_iterations = (num_precomputed_wavelengths() + 2) / 3;
		double dlambda = (kLambdaMax - kLambdaMin) / (3.0 * num_iterations);

		for (int i = 0; i < num_iterations; ++i)
		{
			double lambdas[] =
			{
					kLambdaMin + (3 * i + 0.5) * dlambda,
					kLambdaMin + (3 * i + 1.5) * dlambda,
					kLambdaMin + (3 * i + 2.5) * dlambda
			};

			double luminance_from_radiance[] =
			{
					coeff(lambdas[0], 0) * dlambda, coeff(lambdas[1], 0) * dlambda, coeff(lambdas[2], 0) * dlambda,
					coeff(lambdas[0], 1) * dlambda, coeff(lambdas[1], 1) * dlambda, coeff(lambdas[2], 1) * dlambda,
					coeff(lambdas[0], 2) * dlambda, coeff(lambdas[1], 2) * dlambda, coeff(lambdas[2], 2) * dlambda
			};

			bool blend = i > 0;
			atmosphere_error_t error = precompute(lambdas, luminance_from_radiance, blend, num_scattering_orders);
			if (error != ATMO_NO_ERR) {
				printf("Unable to precompute!");
				return ATMO_INIT_ERR;
			}
		}

		// After the above iterations, the transmittance texture contains the
		// transmittance for the 3 wavelengths used at the last iteration. But we
		// want the transmittance at kLambdaR, kLambdaG, kLambdaB instead, so we
		// must recompute it here for these 3 wavelengths:
		atmosphere_error_t error = compute_transmittance(nullptr, nullptr, false, num_scattering_orders);
		if (error != ATMO_NO_ERR) {
			printf("Unable to precompute!");
			return ATMO_INIT_ERR;
		}
	}

	// copy textures and free buffers

	copy_transmittance_texture();
	copy_irradiance_texture();
	copy_scattering_texture();
	copy_single_scattering_texture();

	clear_buffers();

	return ATMO_NO_ERR;

}

atmosphere::~atmosphere() {


	// Free device memory

	checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.transmittance_texture));
	checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.irradiance_texture));
	checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.scattering_texture));
	checkCudaErrors(cudaDestroyTextureObject(atmosphere_parameters.single_mie_scattering_texture));

	checkCudaErrors(cudaFree(atmosphere_parameters.transmittance_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.delta_irradience_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.delta_mie_scattering_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.delta_multiple_scattering_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.delta_rayleigh_scattering_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.delta_scattering_density_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.irradiance_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.optional_mie_single_scattering_buffer));
	checkCudaErrors(cudaFree(atmosphere_parameters.scattering_buffer));


}

atmosphere::atmosphere() {

	// Allocate device memory for atmosphere_parameters buffers 
	int transmittance_size = TRANSMITTANCE_TEXTURE_WIDTH * TRANSMITTANCE_TEXTURE_HEIGHT * sizeof(float4);
	int irradiance_size = IRRADIANCE_TEXTURE_WIDTH * IRRADIANCE_TEXTURE_HEIGHT * sizeof(float4);
	int scattering_size = SCATTERING_TEXTURE_WIDTH * SCATTERING_TEXTURE_HEIGHT * SCATTERING_TEXTURE_DEPTH * sizeof(float4);

	checkCudaErrors(cudaMalloc(&atmosphere_parameters.transmittance_buffer, transmittance_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.delta_irradience_buffer, irradiance_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.irradiance_buffer, irradiance_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.delta_rayleigh_scattering_buffer, scattering_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.delta_mie_scattering_buffer, scattering_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.scattering_buffer, scattering_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.optional_mie_single_scattering_buffer, scattering_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.delta_scattering_density_buffer, scattering_size));
	checkCudaErrors(cudaMalloc(&atmosphere_parameters.delta_multiple_scattering_buffer, scattering_size));

	m_use_luminance = NONE;
}