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
// File: Custom path trace kernel header: 
//       Performs custom path tracing
//
//-----------------------------------------------

struct Kernel_params {

	//Debug
	bool render;
	bool debug;
	
	// Display
	uint2 resolution;
	float exposure_scale;
	unsigned int *display_buffer;
	float4 *raw_buffer;

	// Progressive rendering state
	unsigned int iteration;
	float3 *accum_buffer;

	// Limit on path length
	unsigned int max_interactions;		// Accumulation buffer count
	int ray_depth;						// Max number of bounces 

	//Volume parameters ( No absorbtion coefficient)

	float min_extinction;	// TO-DO: Extinction minorant (for residual ratio tracking)
	float phase_g1;			// Phase function anisotropy (.0f means isotropic phase function)
	float phase_g2;			// Phase function anisotropy for double lobe phase functions
	float phase_f;			// Anistropy factor for double henyey_greeinstein

	float3 albedo;			// sigma_s / sigma_t
	float3 extinction;		// sigma_t
	float3 transmittance;	// At which depth transmittance gets this value 
	float tr_depth;			// Multiply transmittance density factor (transmittance step size regulator)
	float density_mult;		// Tracking density multiplier

	// Environment
	unsigned int environment_type;
	float azimuth;
	float elevation;
	float3 sun_color;
	float3 sky_color;
	float sun_mult;
	float sky_mult;
	cudaTextureObject_t env_tex;

	int env_sample_tex_res;
	cudaTextureObject_t env_func_tex;
	cudaTextureObject_t env_cdf_tex;
	cudaTextureObject_t env_marginal_func_tex;
	cudaTextureObject_t env_marginal_cdf_tex;
	float env_marginal_int;

	// Debug parameters

	float3 *debug_buffer;
	
	// Integrators

	int integrator;

};
 