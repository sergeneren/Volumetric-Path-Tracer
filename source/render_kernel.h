//
// CUDA volume path tracing kernel interface
//

struct Kernel_params {

	//Debug
	bool render;

	// Display
	uint2 resolution;
	float exposure_scale;
	unsigned int *display_buffer;

	// Progressive rendering state
	unsigned int iteration;
	float3 *accum_buffer;

	// Limit on path length
	unsigned int max_interactions;		// Accumulation buffer count
	int ray_depth;						// Max number of bounces 

	//Volume parameters ( No absorbtion coefficient)

	float max_extinction;	// Extinction majorant
	float min_extinction;	// Extinction minorant (for residual ratio tracking)
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
	cudaTextureObject_t env_tex;
	cudaTextureObject_t env_func_tex;
	cudaTextureObject_t env_cdf_tex;
	cudaTextureObject_t env_marginal_func_tex;
	cudaTextureObject_t env_marginal_cdf_tex;

};
 
extern "C" __global__ void volume_rt_kernel(const VDBInfo gvdb, const Kernel_params kernel_params);
