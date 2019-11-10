const char* definitions_glsl = \
"#define Length float\r\n"\
"#define Wavelength float\r\n"\
"#define Angle float\r\n"\
"#define SolidAngle float\r\n"\
"#define Power float\r\n"\
"#define LuminousPower float\r\n"\
"#define Number float\r\n"\
"#define InverseLength float\r\n"\
"#define Area float\r\n"\
"#define Volume float\r\n"\
"#define NumberDensity float\r\n"\
"#define Irradiance float\r\n"\
"#define Radiance float\r\n"\
"#define SpectralPower float\r\n"\
"#define SpectralIrradiance float\r\n"\
"#define SpectralRadiance float\r\n"\
"#define SpectralRadianceDensity float\r\n"\
"#define ScatteringCoefficient float\r\n"\
"#define InverseSolidAngle float\r\n"\
"#define LuminousIntensity float\r\n"\
"#define Luminance float\r\n"\
"#define Illuminance float\r\n"\
"#define AbstractSpectrum vec3\r\n"\
"#define DimensionlessSpectrum vec3\r\n"\
"#define PowerSpectrum vec3\r\n"\
"#define IrradianceSpectrum vec3\r\n"\
"#define RadianceSpectrum vec3\r\n"\
"#define RadianceDensitySpectrum vec3\r\n"\
"#define ScatteringSpectrum vec3\r\n"\
"#define Position vec3\r\n"\
"#define Direction vec3\r\n"\
"#define Luminance3 vec3\r\n"\
"#define Illuminance3 vec3\r\n"\
"#define TransmittanceTexture sampler2D\r\n"\
"#define AbstractScatteringTexture sampler3D\r\n"\
"#define ReducedScatteringTexture sampler3D\r\n"\
"#define ScatteringTexture sampler3D\r\n"\
"#define ScatteringDensityTexture sampler3D\r\n"\
"#define IrradianceTexture sampler2D\r\n"\
"const Length m = 1.0;\r\n"\
"const Wavelength nm = 1.0;\r\n"\
"const Angle rad = 1.0;\r\n"\
"const SolidAngle sr = 1.0;\r\n"\
"const Power watt = 1.0;\r\n"\
"const LuminousPower lm = 1.0;\r\n"\
"const float PI = 3.14159265358979323846;\r\n"\
"const Length km = 1000.0 * m;\r\n"\
"const Area m2 = m * m;\r\n"\
"const Volume m3 = m * m * m;\r\n"\
"const Angle pi = PI * rad;\r\n"\
"const Angle deg = pi / 180.0;\r\n"\
"const Irradiance watt_per_square_meter = watt / m2;\r\n"\
"const Radiance watt_per_square_meter_per_sr = watt / (m2 * sr);\r\n"\
"const SpectralIrradiance watt_per_square_meter_per_nm = watt / (m2 * nm);\r\n"\
"const SpectralRadiance watt_per_square_meter_per_sr_per_nm =\r\n"\
"    watt / (m2 * sr * nm);\r\n"\
"const SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm =\r\n"\
"    watt / (m3 * sr * nm);\r\n"\
"const LuminousIntensity cd = lm / sr;\r\n"\
"const LuminousIntensity kcd = 1000.0 * cd;\r\n"\
"const Luminance cd_per_square_meter = cd / m2;\r\n"\
"const Luminance kcd_per_square_meter = kcd / m2;\r\n"\
"struct DensityProfileLayer {\r\n"\
"  Length width;\r\n"\
"  Number exp_term;\r\n"\
"  InverseLength exp_scale;\r\n"\
"  InverseLength linear_term;\r\n"\
"  Number constant_term;\r\n"\
"};\r\n"\
"struct DensityProfile {\r\n"\
"  DensityProfileLayer layers[2];\r\n"\
"};\r\n"\
"struct AtmosphereParameters {\r\n"\
"  IrradianceSpectrum solar_irradiance;\r\n"\
"  Angle sun_angular_radius;\r\n"\
"  Length bottom_radius;\r\n"\
"  Length top_radius;\r\n"\
"  DensityProfile rayleigh_density;\r\n"\
"  ScatteringSpectrum rayleigh_scattering;\r\n"\
"  DensityProfile mie_density;\r\n"\
"  ScatteringSpectrum mie_scattering;\r\n"\
"  ScatteringSpectrum mie_extinction;\r\n"\
"  Number mie_phase_function_g;\r\n"\
"  DensityProfile absorption_density;\r\n"\
"  ScatteringSpectrum absorption_extinction;\r\n"\
"  DimensionlessSpectrum ground_albedo;\r\n"\
"  Number mu_s_min;\r\n"\
"};\r\n"\
""; 
