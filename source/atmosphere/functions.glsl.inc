const char* functions_glsl = \
"Number ClampCosine(Number mu) {\r\n"\
"  return clamp(mu, Number(-1.0), Number(1.0));\r\n"\
"}\r\n"\
"Length ClampDistance(Length d) {\r\n"\
"  return max(d, 0.0 * m);\r\n"\
"}\r\n"\
"Length ClampRadius(IN(AtmosphereParameters) atmosphere, Length r) {\r\n"\
"  return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);\r\n"\
"}\r\n"\
"Length SafeSqrt(Area a) {\r\n"\
"  return sqrt(max(a, 0.0 * m2));\r\n"\
"}\r\n"\
"Length DistanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  Area discriminant = r * r * (mu * mu - 1.0) +\r\n"\
"      atmosphere.top_radius * atmosphere.top_radius;\r\n"\
"  return ClampDistance(-r * mu + SafeSqrt(discriminant));\r\n"\
"}\r\n"\
"Length DistanceToBottomAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  Area discriminant = r * r * (mu * mu - 1.0) +\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius;\r\n"\
"  return ClampDistance(-r * mu - SafeSqrt(discriminant));\r\n"\
"}\r\n"\
"bool RayIntersectsGround(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  return mu < 0.0 && r * r * (mu * mu - 1.0) +\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;\r\n"\
"}\r\n"\
"Number GetLayerDensity(IN(DensityProfileLayer) layer, Length altitude) {\r\n"\
"  Number density = layer.exp_term * exp(layer.exp_scale * altitude) +\r\n"\
"      layer.linear_term * altitude + layer.constant_term;\r\n"\
"  return clamp(density, Number(0.0), Number(1.0));\r\n"\
"}\r\n"\
"Number GetProfileDensity(IN(DensityProfile) profile, Length altitude) {\r\n"\
"  return altitude < profile.layers[0].width ?\r\n"\
"      GetLayerDensity(profile.layers[0], altitude) :\r\n"\
"      GetLayerDensity(profile.layers[1], altitude);\r\n"\
"}\r\n"\
"Length ComputeOpticalLengthToTopAtmosphereBoundary(\r\n"\
"    IN(AtmosphereParameters) atmosphere, IN(DensityProfile) profile,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  const int SAMPLE_COUNT = 500;\r\n"\
"  Length dx =\r\n"\
"      DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);\r\n"\
"  Length result = 0.0 * m;\r\n"\
"  for (int i = 0; i <= SAMPLE_COUNT; ++i) {\r\n"\
"    Length d_i = Number(i) * dx;\r\n"\
"    Length r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);\r\n"\
"    Number y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);\r\n"\
"    Number weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5 : 1.0;\r\n"\
"    result += y_i * weight_i * dx;\r\n"\
"  }\r\n"\
"  return result;\r\n"\
"}\r\n"\
"DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundary(\r\n"\
"    IN(AtmosphereParameters) atmosphere, Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  return exp(-(\r\n"\
"      atmosphere.rayleigh_scattering *\r\n"\
"          ComputeOpticalLengthToTopAtmosphereBoundary(\r\n"\
"              atmosphere, atmosphere.rayleigh_density, r, mu) +\r\n"\
"      atmosphere.mie_extinction *\r\n"\
"          ComputeOpticalLengthToTopAtmosphereBoundary(\r\n"\
"              atmosphere, atmosphere.mie_density, r, mu) +\r\n"\
"      atmosphere.absorption_extinction *\r\n"\
"          ComputeOpticalLengthToTopAtmosphereBoundary(\r\n"\
"              atmosphere, atmosphere.absorption_density, r, mu)));\r\n"\
"}\r\n"\
"Number GetTextureCoordFromUnitRange(Number x, int texture_size) {\r\n"\
"  return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));\r\n"\
"}\r\n"\
"Number GetUnitRangeFromTextureCoord(Number u, int texture_size) {\r\n"\
"  return (u - 0.5 / Number(texture_size)) / (1.0 - 1.0 / Number(texture_size));\r\n"\
"}\r\n"\
"vec2 GetTransmittanceTextureUvFromRMu(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length rho =\r\n"\
"      SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);\r\n"\
"  Length d_min = atmosphere.top_radius - r;\r\n"\
"  Length d_max = rho + H;\r\n"\
"  Number x_mu = (d - d_min) / (d_max - d_min);\r\n"\
"  Number x_r = rho / H;\r\n"\
"  return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),\r\n"\
"              GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));\r\n"\
"}\r\n"\
"void GetRMuFromTransmittanceTextureUv(IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(vec2) uv, OUT(Length) r, OUT(Number) mu) {\r\n"\
"  assert(uv.x >= 0.0 && uv.x <= 1.0);\r\n"\
"  assert(uv.y >= 0.0 && uv.y <= 1.0);\r\n"\
"  Number x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);\r\n"\
"  Number x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);\r\n"\
"  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length rho = H * x_r;\r\n"\
"  r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length d_min = atmosphere.top_radius - r;\r\n"\
"  Length d_max = rho + H;\r\n"\
"  Length d = d_min + x_mu * (d_max - d_min);\r\n"\
"  mu = d == 0.0 * m ? Number(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);\r\n"\
"  mu = ClampCosine(mu);\r\n"\
"}\r\n"\
"DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundaryTexture(\r\n"\
"    IN(AtmosphereParameters) atmosphere, IN(vec2) frag_coord) {\r\n"\
"  const vec2 TRANSMITTANCE_TEXTURE_SIZE =\r\n"\
"      vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);\r\n"\
"  Length r;\r\n"\
"  Number mu;\r\n"\
"  GetRMuFromTransmittanceTextureUv(\r\n"\
"      atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);\r\n"\
"  return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);\r\n"\
"}\r\n"\
"DimensionlessSpectrum GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  vec2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);\r\n"\
"  return DimensionlessSpectrum(texture(transmittance_texture, uv));\r\n"\
"}\r\n"\
"DimensionlessSpectrum GetTransmittance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  assert(d >= 0.0 * m);\r\n"\
"  Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));\r\n"\
"  Number mu_d = ClampCosine((r * mu + d) / r_d);\r\n"\
"  if (ray_r_mu_intersects_ground) {\r\n"\
"    return min(\r\n"\
"        GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"            atmosphere, transmittance_texture, r_d, -mu_d) /\r\n"\
"        GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"            atmosphere, transmittance_texture, r, -mu),\r\n"\
"        DimensionlessSpectrum(1.0));\r\n"\
"  } else {\r\n"\
"    return min(\r\n"\
"        GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"            atmosphere, transmittance_texture, r, mu) /\r\n"\
"        GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"            atmosphere, transmittance_texture, r_d, mu_d),\r\n"\
"        DimensionlessSpectrum(1.0));\r\n"\
"  }\r\n"\
"}\r\n"\
"DimensionlessSpectrum GetTransmittanceToSun(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu_s) {\r\n"\
"  Number sin_theta_h = atmosphere.bottom_radius / r;\r\n"\
"  Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));\r\n"\
"  return GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"          atmosphere, transmittance_texture, r, mu_s) *\r\n"\
"      smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,\r\n"\
"                 sin_theta_h * atmosphere.sun_angular_radius / rad,\r\n"\
"                 mu_s - cos_theta_h);\r\n"\
"}\r\n"\
"void ComputeSingleScatteringIntegrand(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu, Length d,\r\n"\
"    bool ray_r_mu_intersects_ground,\r\n"\
"    OUT(DimensionlessSpectrum) rayleigh, OUT(DimensionlessSpectrum) mie) {\r\n"\
"  Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));\r\n"\
"  Number mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);\r\n"\
"  DimensionlessSpectrum transmittance =\r\n"\
"      GetTransmittance(\r\n"\
"          atmosphere, transmittance_texture, r, mu, d,\r\n"\
"          ray_r_mu_intersects_ground) *\r\n"\
"      GetTransmittanceToSun(\r\n"\
"          atmosphere, transmittance_texture, r_d, mu_s_d);\r\n"\
"  rayleigh = transmittance * GetProfileDensity(\r\n"\
"      atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);\r\n"\
"  mie = transmittance * GetProfileDensity(\r\n"\
"      atmosphere.mie_density, r_d - atmosphere.bottom_radius);\r\n"\
"}\r\n"\
"Length DistanceToNearestAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu, bool ray_r_mu_intersects_ground) {\r\n"\
"  if (ray_r_mu_intersects_ground) {\r\n"\
"    return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);\r\n"\
"  } else {\r\n"\
"    return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);\r\n"\
"  }\r\n"\
"}\r\n"\
"void ComputeSingleScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground,\r\n"\
"    OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  assert(nu >= -1.0 && nu <= 1.0);\r\n"\
"  const int SAMPLE_COUNT = 50;\r\n"\
"  Length dx =\r\n"\
"      DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,\r\n"\
"          ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);\r\n"\
"  DimensionlessSpectrum rayleigh_sum = DimensionlessSpectrum(0.0);\r\n"\
"  DimensionlessSpectrum mie_sum = DimensionlessSpectrum(0.0);\r\n"\
"  for (int i = 0; i <= SAMPLE_COUNT; ++i) {\r\n"\
"    Length d_i = Number(i) * dx;\r\n"\
"    DimensionlessSpectrum rayleigh_i;\r\n"\
"    DimensionlessSpectrum mie_i;\r\n"\
"    ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture,\r\n"\
"        r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);\r\n"\
"    Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;\r\n"\
"    rayleigh_sum += rayleigh_i * weight_i;\r\n"\
"    mie_sum += mie_i * weight_i;\r\n"\
"  }\r\n"\
"  rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *\r\n"\
"      atmosphere.rayleigh_scattering;\r\n"\
"  mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;\r\n"\
"}\r\n"\
"InverseSolidAngle RayleighPhaseFunction(Number nu) {\r\n"\
"  InverseSolidAngle k = 3.0 / (16.0 * PI * sr);\r\n"\
"  return k * (1.0 + nu * nu);\r\n"\
"}\r\n"\
"InverseSolidAngle MiePhaseFunction(Number g, Number nu) {\r\n"\
"  InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);\r\n"\
"  return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);\r\n"\
"}\r\n"\
"vec4 GetScatteringTextureUvwzFromRMuMuSNu(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  assert(nu >= -1.0 && nu <= 1.0);\r\n"\
"  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length rho =\r\n"\
"      SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);\r\n"\
"  Length r_mu = r * mu;\r\n"\
"  Area discriminant =\r\n"\
"      r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;\r\n"\
"  Number u_mu;\r\n"\
"  if (ray_r_mu_intersects_ground) {\r\n"\
"    Length d = -r_mu - SafeSqrt(discriminant);\r\n"\
"    Length d_min = r - atmosphere.bottom_radius;\r\n"\
"    Length d_max = rho;\r\n"\
"    u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :\r\n"\
"        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);\r\n"\
"  } else {\r\n"\
"    Length d = -r_mu + SafeSqrt(discriminant + H * H);\r\n"\
"    Length d_min = atmosphere.top_radius - r;\r\n"\
"    Length d_max = rho + H;\r\n"\
"    u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(\r\n"\
"        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);\r\n"\
"  }\r\n"\
"  Length d = DistanceToTopAtmosphereBoundary(\r\n"\
"      atmosphere, atmosphere.bottom_radius, mu_s);\r\n"\
"  Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;\r\n"\
"  Length d_max = H;\r\n"\
"  Number a = (d - d_min) / (d_max - d_min);\r\n"\
"  Number A =\r\n"\
"      -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);\r\n"\
"  Number u_mu_s = GetTextureCoordFromUnitRange(\r\n"\
"      max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);\r\n"\
"  Number u_nu = (nu + 1.0) / 2.0;\r\n"\
"  return vec4(u_nu, u_mu_s, u_mu, u_r);\r\n"\
"}\r\n"\
"void GetRMuMuSNuFromScatteringTextureUvwz(IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(vec4) uvwz, OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s,\r\n"\
"    OUT(Number) nu, OUT(bool) ray_r_mu_intersects_ground) {\r\n"\
"  assert(uvwz.x >= 0.0 && uvwz.x <= 1.0);\r\n"\
"  assert(uvwz.y >= 0.0 && uvwz.y <= 1.0);\r\n"\
"  assert(uvwz.z >= 0.0 && uvwz.z <= 1.0);\r\n"\
"  assert(uvwz.w >= 0.0 && uvwz.w <= 1.0);\r\n"\
"  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -\r\n"\
"      atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  Length rho =\r\n"\
"      H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);\r\n"\
"  r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);\r\n"\
"  if (uvwz.z < 0.5) {\r\n"\
"    Length d_min = r - atmosphere.bottom_radius;\r\n"\
"    Length d_max = rho;\r\n"\
"    Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(\r\n"\
"        1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);\r\n"\
"    mu = d == 0.0 * m ? Number(-1.0) :\r\n"\
"        ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));\r\n"\
"    ray_r_mu_intersects_ground = true;\r\n"\
"  } else {\r\n"\
"    Length d_min = atmosphere.top_radius - r;\r\n"\
"    Length d_max = rho + H;\r\n"\
"    Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(\r\n"\
"        2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);\r\n"\
"    mu = d == 0.0 * m ? Number(1.0) :\r\n"\
"        ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));\r\n"\
"    ray_r_mu_intersects_ground = false;\r\n"\
"  }\r\n"\
"  Number x_mu_s =\r\n"\
"      GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);\r\n"\
"  Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;\r\n"\
"  Length d_max = H;\r\n"\
"  Number A =\r\n"\
"      -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);\r\n"\
"  Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);\r\n"\
"  Length d = d_min + min(a, A) * (d_max - d_min);\r\n"\
"  mu_s = d == 0.0 * m ? Number(1.0) :\r\n"\
"     ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));\r\n"\
"  nu = ClampCosine(uvwz.x * 2.0 - 1.0);\r\n"\
"}\r\n"\
"void GetRMuMuSNuFromScatteringTextureFragCoord(\r\n"\
"    IN(AtmosphereParameters) atmosphere, IN(vec3) frag_coord,\r\n"\
"    OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s, OUT(Number) nu,\r\n"\
"    OUT(bool) ray_r_mu_intersects_ground) {\r\n"\
"  const vec4 SCATTERING_TEXTURE_SIZE = vec4(\r\n"\
"      SCATTERING_TEXTURE_NU_SIZE - 1,\r\n"\
"      SCATTERING_TEXTURE_MU_S_SIZE,\r\n"\
"      SCATTERING_TEXTURE_MU_SIZE,\r\n"\
"      SCATTERING_TEXTURE_R_SIZE);\r\n"\
"  Number frag_coord_nu =\r\n"\
"      floor(frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));\r\n"\
"  Number frag_coord_mu_s =\r\n"\
"      mod(frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));\r\n"\
"  vec4 uvwz =\r\n"\
"      vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /\r\n"\
"          SCATTERING_TEXTURE_SIZE;\r\n"\
"  GetRMuMuSNuFromScatteringTextureUvwz(\r\n"\
"      atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),\r\n"\
"      mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));\r\n"\
"}\r\n"\
"void ComputeSingleScatteringTexture(IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture, IN(vec3) frag_coord,\r\n"\
"    OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {\r\n"\
"  Length r;\r\n"\
"  Number mu;\r\n"\
"  Number mu_s;\r\n"\
"  Number nu;\r\n"\
"  bool ray_r_mu_intersects_ground;\r\n"\
"  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,\r\n"\
"      r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  ComputeSingleScattering(atmosphere, transmittance_texture,\r\n"\
"      r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);\r\n"\
"}\r\n"\
"TEMPLATE(AbstractSpectrum)\r\n"\
"AbstractSpectrum GetScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(AbstractScatteringTexture TEMPLATE_ARGUMENT(AbstractSpectrum))\r\n"\
"        scattering_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground) {\r\n"\
"  vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(\r\n"\
"      atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);\r\n"\
"  Number tex_x = floor(tex_coord_x);\r\n"\
"  Number lerp = tex_coord_x - tex_x;\r\n"\
"  vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),\r\n"\
"      uvwz.z, uvwz.w);\r\n"\
"  vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),\r\n"\
"      uvwz.z, uvwz.w);\r\n"\
"  return AbstractSpectrum(texture(scattering_texture, uvw0) * (1.0 - lerp) +\r\n"\
"      texture(scattering_texture, uvw1) * lerp);\r\n"\
"}\r\n"\
"RadianceSpectrum GetScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    IN(ScatteringTexture) multiple_scattering_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground,\r\n"\
"    int scattering_order) {\r\n"\
"  if (scattering_order == 1) {\r\n"\
"    IrradianceSpectrum rayleigh = GetScattering(\r\n"\
"        atmosphere, single_rayleigh_scattering_texture, r, mu, mu_s, nu,\r\n"\
"        ray_r_mu_intersects_ground);\r\n"\
"    IrradianceSpectrum mie = GetScattering(\r\n"\
"        atmosphere, single_mie_scattering_texture, r, mu, mu_s, nu,\r\n"\
"        ray_r_mu_intersects_ground);\r\n"\
"    return rayleigh * RayleighPhaseFunction(nu) +\r\n"\
"        mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);\r\n"\
"  } else {\r\n"\
"    return GetScattering(\r\n"\
"        atmosphere, multiple_scattering_texture, r, mu, mu_s, nu,\r\n"\
"        ray_r_mu_intersects_ground);\r\n"\
"  }\r\n"\
"}\r\n"\
"IrradianceSpectrum GetIrradiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(IrradianceTexture) irradiance_texture,\r\n"\
"    Length r, Number mu_s);\r\n"\
"RadianceDensitySpectrum ComputeScatteringDensity(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    IN(ScatteringTexture) multiple_scattering_texture,\r\n"\
"    IN(IrradianceTexture) irradiance_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu, int scattering_order) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  assert(nu >= -1.0 && nu <= 1.0);\r\n"\
"  assert(scattering_order >= 2);\r\n"\
"  vec3 zenith_direction = vec3(0.0, 0.0, 1.0);\r\n"\
"  vec3 omega = vec3(sqrt(1.0 - mu * mu), 0.0, mu);\r\n"\
"  Number sun_dir_x = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;\r\n"\
"  Number sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));\r\n"\
"  vec3 omega_s = vec3(sun_dir_x, sun_dir_y, mu_s);\r\n"\
"  const int SAMPLE_COUNT = 16;\r\n"\
"  const Angle dphi = pi / Number(SAMPLE_COUNT);\r\n"\
"  const Angle dtheta = pi / Number(SAMPLE_COUNT);\r\n"\
"  RadianceDensitySpectrum rayleigh_mie =\r\n"\
"      RadianceDensitySpectrum(0.0 * watt_per_cubic_meter_per_sr_per_nm);\r\n"\
"  for (int l = 0; l < SAMPLE_COUNT; ++l) {\r\n"\
"    Angle theta = (Number(l) + 0.5) * dtheta;\r\n"\
"    Number cos_theta = cos(theta);\r\n"\
"    Number sin_theta = sin(theta);\r\n"\
"    bool ray_r_theta_intersects_ground =\r\n"\
"        RayIntersectsGround(atmosphere, r, cos_theta);\r\n"\
"    Length distance_to_ground = 0.0 * m;\r\n"\
"    DimensionlessSpectrum transmittance_to_ground = DimensionlessSpectrum(0.0);\r\n"\
"    DimensionlessSpectrum ground_albedo = DimensionlessSpectrum(0.0);\r\n"\
"    if (ray_r_theta_intersects_ground) {\r\n"\
"      distance_to_ground =\r\n"\
"          DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);\r\n"\
"      transmittance_to_ground =\r\n"\
"          GetTransmittance(atmosphere, transmittance_texture, r, cos_theta,\r\n"\
"              distance_to_ground, true /* ray_intersects_ground */);\r\n"\
"      ground_albedo = atmosphere.ground_albedo;\r\n"\
"    }\r\n"\
"    for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {\r\n"\
"      Angle phi = (Number(m) + 0.5) * dphi;\r\n"\
"      vec3 omega_i =\r\n"\
"          vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);\r\n"\
"      SolidAngle domega_i = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;\r\n"\
"      Number nu1 = dot(omega_s, omega_i);\r\n"\
"      RadianceSpectrum incident_radiance = GetScattering(atmosphere,\r\n"\
"          single_rayleigh_scattering_texture, single_mie_scattering_texture,\r\n"\
"          multiple_scattering_texture, r, omega_i.z, mu_s, nu1,\r\n"\
"          ray_r_theta_intersects_ground, scattering_order - 1);\r\n"\
"      vec3 ground_normal =\r\n"\
"          normalize(zenith_direction * r + omega_i * distance_to_ground);\r\n"\
"      IrradianceSpectrum ground_irradiance = GetIrradiance(\r\n"\
"          atmosphere, irradiance_texture, atmosphere.bottom_radius,\r\n"\
"          dot(ground_normal, omega_s));\r\n"\
"      incident_radiance += transmittance_to_ground *\r\n"\
"          ground_albedo * (1.0 / (PI * sr)) * ground_irradiance;\r\n"\
"      Number nu2 = dot(omega, omega_i);\r\n"\
"      Number rayleigh_density = GetProfileDensity(\r\n"\
"          atmosphere.rayleigh_density, r - atmosphere.bottom_radius);\r\n"\
"      Number mie_density = GetProfileDensity(\r\n"\
"          atmosphere.mie_density, r - atmosphere.bottom_radius);\r\n"\
"      rayleigh_mie += incident_radiance * (\r\n"\
"          atmosphere.rayleigh_scattering * rayleigh_density *\r\n"\
"              RayleighPhaseFunction(nu2) +\r\n"\
"          atmosphere.mie_scattering * mie_density *\r\n"\
"              MiePhaseFunction(atmosphere.mie_phase_function_g, nu2)) *\r\n"\
"          domega_i;\r\n"\
"    }\r\n"\
"  }\r\n"\
"  return rayleigh_mie;\r\n"\
"}\r\n"\
"RadianceSpectrum ComputeMultipleScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ScatteringDensityTexture) scattering_density_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu >= -1.0 && mu <= 1.0);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  assert(nu >= -1.0 && nu <= 1.0);\r\n"\
"  const int SAMPLE_COUNT = 50;\r\n"\
"  Length dx =\r\n"\
"      DistanceToNearestAtmosphereBoundary(\r\n"\
"          atmosphere, r, mu, ray_r_mu_intersects_ground) /\r\n"\
"              Number(SAMPLE_COUNT);\r\n"\
"  RadianceSpectrum rayleigh_mie_sum =\r\n"\
"      RadianceSpectrum(0.0 * watt_per_square_meter_per_sr_per_nm);\r\n"\
"  for (int i = 0; i <= SAMPLE_COUNT; ++i) {\r\n"\
"    Length d_i = Number(i) * dx;\r\n"\
"    Length r_i =\r\n"\
"        ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));\r\n"\
"    Number mu_i = ClampCosine((r * mu + d_i) / r_i);\r\n"\
"    Number mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);\r\n"\
"    RadianceSpectrum rayleigh_mie_i =\r\n"\
"        GetScattering(\r\n"\
"            atmosphere, scattering_density_texture, r_i, mu_i, mu_s_i, nu,\r\n"\
"            ray_r_mu_intersects_ground) *\r\n"\
"        GetTransmittance(\r\n"\
"            atmosphere, transmittance_texture, r, mu, d_i,\r\n"\
"            ray_r_mu_intersects_ground) *\r\n"\
"        dx;\r\n"\
"    Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;\r\n"\
"    rayleigh_mie_sum += rayleigh_mie_i * weight_i;\r\n"\
"  }\r\n"\
"  return rayleigh_mie_sum;\r\n"\
"}\r\n"\
"RadianceDensitySpectrum ComputeScatteringDensityTexture(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    IN(ScatteringTexture) multiple_scattering_texture,\r\n"\
"    IN(IrradianceTexture) irradiance_texture,\r\n"\
"    IN(vec3) frag_coord, int scattering_order) {\r\n"\
"  Length r;\r\n"\
"  Number mu;\r\n"\
"  Number mu_s;\r\n"\
"  Number nu;\r\n"\
"  bool ray_r_mu_intersects_ground;\r\n"\
"  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,\r\n"\
"      r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  return ComputeScatteringDensity(atmosphere, transmittance_texture,\r\n"\
"      single_rayleigh_scattering_texture, single_mie_scattering_texture,\r\n"\
"      multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu,\r\n"\
"      scattering_order);\r\n"\
"}\r\n"\
"RadianceSpectrum ComputeMultipleScatteringTexture(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ScatteringDensityTexture) scattering_density_texture,\r\n"\
"    IN(vec3) frag_coord, OUT(Number) nu) {\r\n"\
"  Length r;\r\n"\
"  Number mu;\r\n"\
"  Number mu_s;\r\n"\
"  bool ray_r_mu_intersects_ground;\r\n"\
"  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,\r\n"\
"      r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  return ComputeMultipleScattering(atmosphere, transmittance_texture,\r\n"\
"      scattering_density_texture, r, mu, mu_s, nu,\r\n"\
"      ray_r_mu_intersects_ground);\r\n"\
"}\r\n"\
"IrradianceSpectrum ComputeDirectIrradiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    Length r, Number mu_s) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  Number alpha_s = atmosphere.sun_angular_radius / rad;\r\n"\
"  Number average_cosine_factor =\r\n"\
"    mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :\r\n"\
"        (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));\r\n"\
"  return atmosphere.solar_irradiance *\r\n"\
"      GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"          atmosphere, transmittance_texture, r, mu_s) * average_cosine_factor;\r\n"\
"}\r\n"\
"IrradianceSpectrum ComputeIndirectIrradiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    IN(ScatteringTexture) multiple_scattering_texture,\r\n"\
"    Length r, Number mu_s, int scattering_order) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  assert(scattering_order >= 1);\r\n"\
"  const int SAMPLE_COUNT = 32;\r\n"\
"  const Angle dphi = pi / Number(SAMPLE_COUNT);\r\n"\
"  const Angle dtheta = pi / Number(SAMPLE_COUNT);\r\n"\
"  IrradianceSpectrum result =\r\n"\
"      IrradianceSpectrum(0.0 * watt_per_square_meter_per_nm);\r\n"\
"  vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);\r\n"\
"  for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {\r\n"\
"    Angle theta = (Number(j) + 0.5) * dtheta;\r\n"\
"    for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {\r\n"\
"      Angle phi = (Number(i) + 0.5) * dphi;\r\n"\
"      vec3 omega =\r\n"\
"          vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));\r\n"\
"      SolidAngle domega = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;\r\n"\
"      Number nu = dot(omega, omega_s);\r\n"\
"      result += GetScattering(atmosphere, single_rayleigh_scattering_texture,\r\n"\
"          single_mie_scattering_texture, multiple_scattering_texture,\r\n"\
"          r, omega.z, mu_s, nu, false /* ray_r_theta_intersects_ground */,\r\n"\
"          scattering_order) *\r\n"\
"              omega.z * domega;\r\n"\
"    }\r\n"\
"  }\r\n"\
"  return result;\r\n"\
"}\r\n"\
"vec2 GetIrradianceTextureUvFromRMuS(IN(AtmosphereParameters) atmosphere,\r\n"\
"    Length r, Number mu_s) {\r\n"\
"  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);\r\n"\
"  assert(mu_s >= -1.0 && mu_s <= 1.0);\r\n"\
"  Number x_r = (r - atmosphere.bottom_radius) /\r\n"\
"      (atmosphere.top_radius - atmosphere.bottom_radius);\r\n"\
"  Number x_mu_s = mu_s * 0.5 + 0.5;\r\n"\
"  return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),\r\n"\
"              GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));\r\n"\
"}\r\n"\
"void GetRMuSFromIrradianceTextureUv(IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(vec2) uv, OUT(Length) r, OUT(Number) mu_s) {\r\n"\
"  assert(uv.x >= 0.0 && uv.x <= 1.0);\r\n"\
"  assert(uv.y >= 0.0 && uv.y <= 1.0);\r\n"\
"  Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);\r\n"\
"  Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);\r\n"\
"  r = atmosphere.bottom_radius +\r\n"\
"      x_r * (atmosphere.top_radius - atmosphere.bottom_radius);\r\n"\
"  mu_s = ClampCosine(2.0 * x_mu_s - 1.0);\r\n"\
"}\r\n"\
"const vec2 IRRADIANCE_TEXTURE_SIZE =\r\n"\
"    vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);\r\n"\
"IrradianceSpectrum ComputeDirectIrradianceTexture(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(vec2) frag_coord) {\r\n"\
"  Length r;\r\n"\
"  Number mu_s;\r\n"\
"  GetRMuSFromIrradianceTextureUv(\r\n"\
"      atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);\r\n"\
"  return ComputeDirectIrradiance(atmosphere, transmittance_texture, r, mu_s);\r\n"\
"}\r\n"\
"IrradianceSpectrum ComputeIndirectIrradianceTexture(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    IN(ScatteringTexture) multiple_scattering_texture,\r\n"\
"    IN(vec2) frag_coord, int scattering_order) {\r\n"\
"  Length r;\r\n"\
"  Number mu_s;\r\n"\
"  GetRMuSFromIrradianceTextureUv(\r\n"\
"      atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);\r\n"\
"  return ComputeIndirectIrradiance(atmosphere,\r\n"\
"      single_rayleigh_scattering_texture, single_mie_scattering_texture,\r\n"\
"      multiple_scattering_texture, r, mu_s, scattering_order);\r\n"\
"}\r\n"\
"IrradianceSpectrum GetIrradiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(IrradianceTexture) irradiance_texture,\r\n"\
"    Length r, Number mu_s) {\r\n"\
"  vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);\r\n"\
"  return IrradianceSpectrum(texture(irradiance_texture, uv));\r\n"\
"}\r\n"\
"#ifdef COMBINED_SCATTERING_TEXTURES\r\n"\
"vec3 GetExtrapolatedSingleMieScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere, IN(vec4) scattering) {\r\n"\
"  if (scattering.r == 0.0) {\r\n"\
"    return vec3(0.0);\r\n"\
"  }\r\n"\
"  return scattering.rgb * scattering.a / scattering.r *\r\n"\
"	    (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *\r\n"\
"	    (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);\r\n"\
"}\r\n"\
"#endif\r\n"\
"IrradianceSpectrum GetCombinedScattering(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(ReducedScatteringTexture) scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    Length r, Number mu, Number mu_s, Number nu,\r\n"\
"    bool ray_r_mu_intersects_ground,\r\n"\
"    OUT(IrradianceSpectrum) single_mie_scattering) {\r\n"\
"  vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(\r\n"\
"      atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);\r\n"\
"  Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);\r\n"\
"  Number tex_x = floor(tex_coord_x);\r\n"\
"  Number lerp = tex_coord_x - tex_x;\r\n"\
"  vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),\r\n"\
"      uvwz.z, uvwz.w);\r\n"\
"  vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),\r\n"\
"      uvwz.z, uvwz.w);\r\n"\
"#ifdef COMBINED_SCATTERING_TEXTURES\r\n"\
"  vec4 combined_scattering =\r\n"\
"      texture(scattering_texture, uvw0) * (1.0 - lerp) +\r\n"\
"      texture(scattering_texture, uvw1) * lerp;\r\n"\
"  IrradianceSpectrum scattering = IrradianceSpectrum(combined_scattering);\r\n"\
"  single_mie_scattering =\r\n"\
"      GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);\r\n"\
"#else\r\n"\
"  IrradianceSpectrum scattering = IrradianceSpectrum(\r\n"\
"      texture(scattering_texture, uvw0) * (1.0 - lerp) +\r\n"\
"      texture(scattering_texture, uvw1) * lerp);\r\n"\
"  single_mie_scattering = IrradianceSpectrum(\r\n"\
"      texture(single_mie_scattering_texture, uvw0) * (1.0 - lerp) +\r\n"\
"      texture(single_mie_scattering_texture, uvw1) * lerp);\r\n"\
"#endif\r\n"\
"  return scattering;\r\n"\
"}\r\n"\
"RadianceSpectrum GetSkyRadiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ReducedScatteringTexture) scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    Position camera, IN(Direction) view_ray, Length shadow_length,\r\n"\
"    IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {\r\n"\
"  Length r = length(camera);\r\n"\
"  Length rmu = dot(camera, view_ray);\r\n"\
"  Length distance_to_top_atmosphere_boundary = -rmu -\r\n"\
"      sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);\r\n"\
"  if (distance_to_top_atmosphere_boundary > 0.0 * m) {\r\n"\
"    camera = camera + view_ray * distance_to_top_atmosphere_boundary;\r\n"\
"    r = atmosphere.top_radius;\r\n"\
"    rmu += distance_to_top_atmosphere_boundary;\r\n"\
"  } else if (r > atmosphere.top_radius) {\r\n"\
"    transmittance = DimensionlessSpectrum(1.0);\r\n"\
"    return RadianceSpectrum(0.0 * watt_per_square_meter_per_sr_per_nm);\r\n"\
"  }\r\n"\
"  Number mu = rmu / r;\r\n"\
"  Number mu_s = dot(camera, sun_direction) / r;\r\n"\
"  Number nu = dot(view_ray, sun_direction);\r\n"\
"  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);\r\n"\
"  transmittance = ray_r_mu_intersects_ground ? DimensionlessSpectrum(0.0) :\r\n"\
"      GetTransmittanceToTopAtmosphereBoundary(\r\n"\
"          atmosphere, transmittance_texture, r, mu);\r\n"\
"  IrradianceSpectrum single_mie_scattering;\r\n"\
"  IrradianceSpectrum scattering;\r\n"\
"  if (shadow_length == 0.0 * m) {\r\n"\
"    scattering = GetCombinedScattering(\r\n"\
"        atmosphere, scattering_texture, single_mie_scattering_texture,\r\n"\
"        r, mu, mu_s, nu, ray_r_mu_intersects_ground,\r\n"\
"        single_mie_scattering);\r\n"\
"  } else {\r\n"\
"    Length d = shadow_length;\r\n"\
"    Length r_p =\r\n"\
"        ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));\r\n"\
"    Number mu_p = (r * mu + d) / r_p;\r\n"\
"    Number mu_s_p = (r * mu_s + d * nu) / r_p;\r\n"\
"    scattering = GetCombinedScattering(\r\n"\
"        atmosphere, scattering_texture, single_mie_scattering_texture,\r\n"\
"        r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,\r\n"\
"        single_mie_scattering);\r\n"\
"    DimensionlessSpectrum shadow_transmittance =\r\n"\
"        GetTransmittance(atmosphere, transmittance_texture,\r\n"\
"            r, mu, shadow_length, ray_r_mu_intersects_ground);\r\n"\
"    scattering = scattering * shadow_transmittance;\r\n"\
"    single_mie_scattering = single_mie_scattering * shadow_transmittance;\r\n"\
"  }\r\n"\
"  return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *\r\n"\
"      MiePhaseFunction(atmosphere.mie_phase_function_g, nu);\r\n"\
"}\r\n"\
"RadianceSpectrum GetSkyRadianceToPoint(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(ReducedScatteringTexture) scattering_texture,\r\n"\
"    IN(ReducedScatteringTexture) single_mie_scattering_texture,\r\n"\
"    Position camera, IN(Position) point, Length shadow_length,\r\n"\
"    IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {\r\n"\
"  Direction view_ray = normalize(point - camera);\r\n"\
"  Length r = length(camera);\r\n"\
"  Length rmu = dot(camera, view_ray);\r\n"\
"  Length distance_to_top_atmosphere_boundary = -rmu -\r\n"\
"      sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);\r\n"\
"  if (distance_to_top_atmosphere_boundary > 0.0 * m) {\r\n"\
"    camera = camera + view_ray * distance_to_top_atmosphere_boundary;\r\n"\
"    r = atmosphere.top_radius;\r\n"\
"    rmu += distance_to_top_atmosphere_boundary;\r\n"\
"  }\r\n"\
"  Number mu = rmu / r;\r\n"\
"  Number mu_s = dot(camera, sun_direction) / r;\r\n"\
"  Number nu = dot(view_ray, sun_direction);\r\n"\
"  Length d = length(point - camera);\r\n"\
"  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);\r\n"\
"  transmittance = GetTransmittance(atmosphere, transmittance_texture,\r\n"\
"      r, mu, d, ray_r_mu_intersects_ground);\r\n"\
"  IrradianceSpectrum single_mie_scattering;\r\n"\
"  IrradianceSpectrum scattering = GetCombinedScattering(\r\n"\
"      atmosphere, scattering_texture, single_mie_scattering_texture,\r\n"\
"      r, mu, mu_s, nu, ray_r_mu_intersects_ground,\r\n"\
"      single_mie_scattering);\r\n"\
"  d = max(d - shadow_length, 0.0 * m);\r\n"\
"  Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));\r\n"\
"  Number mu_p = (r * mu + d) / r_p;\r\n"\
"  Number mu_s_p = (r * mu_s + d * nu) / r_p;\r\n"\
"  IrradianceSpectrum single_mie_scattering_p;\r\n"\
"  IrradianceSpectrum scattering_p = GetCombinedScattering(\r\n"\
"      atmosphere, scattering_texture, single_mie_scattering_texture,\r\n"\
"      r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,\r\n"\
"      single_mie_scattering_p);\r\n"\
"  DimensionlessSpectrum shadow_transmittance = transmittance;\r\n"\
"  if (shadow_length > 0.0 * m) {\r\n"\
"    shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,\r\n"\
"        r, mu, d, ray_r_mu_intersects_ground);\r\n"\
"  }\r\n"\
"  scattering = scattering - shadow_transmittance * scattering_p;\r\n"\
"  single_mie_scattering =\r\n"\
"      single_mie_scattering - shadow_transmittance * single_mie_scattering_p;\r\n"\
"#ifdef COMBINED_SCATTERING_TEXTURES\r\n"\
"  single_mie_scattering = GetExtrapolatedSingleMieScattering(\r\n"\
"      atmosphere, vec4(scattering, single_mie_scattering.r));\r\n"\
"#endif\r\n"\
"  single_mie_scattering = single_mie_scattering *\r\n"\
"      smoothstep(Number(0.0), Number(0.01), mu_s);\r\n"\
"  return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *\r\n"\
"      MiePhaseFunction(atmosphere.mie_phase_function_g, nu);\r\n"\
"}\r\n"\
"IrradianceSpectrum GetSunAndSkyIrradiance(\r\n"\
"    IN(AtmosphereParameters) atmosphere,\r\n"\
"    IN(TransmittanceTexture) transmittance_texture,\r\n"\
"    IN(IrradianceTexture) irradiance_texture,\r\n"\
"    IN(Position) point, IN(Direction) normal, IN(Direction) sun_direction,\r\n"\
"    OUT(IrradianceSpectrum) sky_irradiance) {\r\n"\
"  Length r = length(point);\r\n"\
"  Number mu_s = dot(point, sun_direction) / r;\r\n"\
"  sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *\r\n"\
"      (1.0 + dot(normal, point) / r) * 0.5;\r\n"\
"  return atmosphere.solar_irradiance *\r\n"\
"      GetTransmittanceToSun(\r\n"\
"          atmosphere, transmittance_texture, r, mu_s) *\r\n"\
"      max(dot(normal, sun_direction), 0.0);\r\n"\
"}\r\n"\
""; 
