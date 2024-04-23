import taichi as ti
from taichi.math import *
import lib.volume_rendering_models as volume
from lib.textures import TRANS_LUT_RES

# Transmittance LUT lookup mapping from
# https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html#transmittance_lookup
@ti.func
def tex_coord_to_unit_range(u: float, tex_size: float):
    return (u - .5 / tex_size) / (1. - 1. / tex_size)

@ti.func
def unit_range_to_tex_coord(x: float, tex_size: float):
    return .5 / tex_size + x * (1. - 1. / tex_size)
    
@ti.func
def uv_to_mu_r(uv: vec2):
  x_mu = tex_coord_to_unit_range(uv.x, TRANS_LUT_RES[0])
  x_r = tex_coord_to_unit_range(uv.y, TRANS_LUT_RES[1])
  H = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit -
      volume.planet_r * volume.planet_r)
  rho = H * x_r
  r = sqrt(rho * rho + volume.planet_r * volume.planet_r)
  d_min = volume.atmos_upper_limit - r
  d_max = rho + H
  d = d_min + x_mu * (d_max - d_min)
  mu = 1.0 if d == 0.0 else (H * H - rho * rho - d * d) / (2.0 * r * d)
  mu = clamp(mu, -1., 1.)
  return mu, r-volume.planet_r

@ti.func
def d_to_atm_top(mu: float, r: float):
    discriminant = r * r * (mu * mu - 1.0) + volume.atmos_upper_limit * volume.atmos_upper_limit
    return max(-r * mu + sqrt(max(discriminant, 0.)), 0.)

@ti.func
def mu_r_to_uv(mu: float, r: float):
    H     = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit - volume.planet_r * volume.planet_r)
    rho   = sqrt(r * r - volume.planet_r * volume.planet_r)
    d     = d_to_atm_top(mu, r)
    d_min = volume.atmos_upper_limit - r
    d_max = rho + H
    x_mu  = (d - d_min) / (d_max - d_min)
    x_r   = rho / H
        
    return vec2(
        unit_range_to_tex_coord(x_mu, TRANS_LUT_RES[0]),
        unit_range_to_tex_coord(x_r, TRANS_LUT_RES[1])
    )