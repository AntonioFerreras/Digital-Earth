import lib.volume_rendering_models as volume
import taichi as ti
from taichi.math import *
import numpy as np


#######################

# LUT Reparameterization

@ti.func
def ClampCosine(mu):
    return clamp(mu, -1.0, 1.0)

@ti.func
def ClampDistance(d):
    return max(d, 0.0)

@ti.func
def ClampRadius(r):
    return clamp(r, volume.planet_r, volume.atmos_upper_limit)

@ti.func
def SafeSqrt(a):
    return sqrt(max(a, 0.0))

@ti.func
def GetTextureCoordFromUnitRange(x, texture_size):
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size))

@ti.func
def GetUnitRangeFromTextureCoord(u, texture_size):
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size))

@ti.func
def DistanceToTopAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + volume.atmos_upper_limit * volume.atmos_upper_limit
    return ClampDistance(-r * mu + SafeSqrt(discriminant))

@ti.func
def DistanceToBottomAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + volume.planet_r * volume.planet_r
    return ClampDistance(-r * mu - SafeSqrt(discriminant))

@ti.func
def RayIntersectsGround(r, mu):
    return mu < 0.0 and r * r * (mu * mu - 1.0) + volume.planet_r * volume.planet_r >= 0.0

mu_s_min = -0.2

SCATTERING_TEXTURE_R_SIZE = 4096
SCATTERING_TEXTURE_MU_SIZE = 4096
SCATTERING_TEXTURE_MU_S_SIZE = 4096
SCATTERING_TEXTURE_NU_SIZE = 4096

@ti.func
def mu_s_mapping(x: float) -> float:
    return pow((x - mu_s_min) / (1.0 - mu_s_min), 0.65)

@ti.func
def inverse_mu_s_mapping(y: float, tol: float = 1e-12) -> float:
    return mu_s_min + (1.0 - mu_s_min) * pow(y, 1.0 / 0.65)


@ti.func
def GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground):
    H = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit -
        volume.planet_r * volume.planet_r)
    # Distance to the horizon.
    rho = SafeSqrt(r * r - volume.planet_r * volume.planet_r)
    u = rho / H
    v = mu * 0.5 + 0.5
    z = mu_s_mapping(mu_s)
    w = nu * 0.5 + 0.5
    return vec4(u, v, z, w)
    # # Assert statements

    # # Distance to top atmosphere boundary for a horizontal ray at ground level.
    # H = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit -
    #     volume.planet_r * volume.planet_r)
    # # Distance to the horizon.
    # rho = SafeSqrt(r * r - volume.planet_r * volume.planet_r)
    # u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE)

    # # Discriminant of the quadratic equation for the intersections of the ray
    # # (r,mu) with the ground (see RayIntersectsGround).
    # r_mu = r * mu
    # discriminant = r_mu * r_mu - r * r + volume.planet_r * volume.planet_r
    # u_mu = 0.0
    
    # if ray_r_mu_intersects_ground:
    #     # Distance to the ground for the ray (r,mu), and its minimum and maximum
    #     # values over all mu - obtained for (r,-1) and (r,mu_horizon).
    #     d = -r_mu - SafeSqrt(discriminant)
    #     d_min = r - volume.planet_r
    #     d_max = rho
    #     u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
    #         0.0 if d_max == d_min else (d - d_min) / (d_max - d_min), 
    #         SCATTERING_TEXTURE_MU_SIZE / 2)
    # else:
    #     # Distance to the top atmosphere boundary for the ray (r,mu), and its
    #     # minimum and maximum values over all mu - obtained for (r,1) and
    #     # (r,mu_horizon).
    #     d = -r_mu + SafeSqrt(discriminant + H * H)
    #     d_min = volume.atmos_upper_limit - r
    #     d_max = rho + H
    #     u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
    #         (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2)

    # d = DistanceToTopAtmosphereBoundary(volume.planet_r, mu_s)
    # d_min = volume.atmos_upper_limit - volume.planet_r
    # d_max = H
    # a = (d - d_min) / (d_max - d_min)
    # D = DistanceToTopAtmosphereBoundary(volume.planet_r, mu_s_min)
    # A = (D - d_min) / (d_max - d_min)
    
    # # An ad-hoc function equal to 0 for mu_s = mu_s_min (because then d = D and
    # # thus a = A), equal to 1 for mu_s = 1 (because then d = d_min and thus
    # # a = 0), and with a large slope around mu_s = 0, to get more texture 
    # # samples near the horizon.
    # u_mu_s = GetTextureCoordFromUnitRange(
    #     max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE)

    # u_nu = (nu + 1.0) / 2.0
    # return vec4(u_nu, u_mu_s, u_mu, u_r)



@ti.func
def GetRMuMuSNuFromScatteringTextureUvwz(uvwz):

    # Distance to top atmosphere boundary for a horizontal ray at ground level.
    # H = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit -
    #     volume.planet_r * volume.planet_r)
    # # Distance to the horizon.
    # rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE)
    # r = sqrt(rho * rho + volume.planet_r * volume.planet_r)

    # ray_r_mu_intersects_ground = False
    # mu = 0.0
    # if uvwz.z < 0.5:
    #     # Distance to the ground for the ray (r,mu), and its minimum and maximum
    #     # values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
    #     # we can recover mu:
    #     d_min = r - volume.planet_r
    #     d_max = rho
    #     d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
    #         1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2)
    #     mu = -1.0 if d == 0.0 else ClampCosine(-(rho * rho + d * d) / (2.0 * r * d))
    #     ray_r_mu_intersects_ground = True
    # else:
    #     # Distance to the top atmosphere boundary for the ray (r,mu), and its
    #     # minimum and maximum values over all mu - obtained for (r,1) and
    #     # (r,mu_horizon) - from which we can recover mu:
    #     d_min = volume.atmos_upper_limit - r
    #     d_max = rho + H
    #     d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
    #         2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2)
    #     mu = 1.0 if d == 0.0 else ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d))
    #     ray_r_mu_intersects_ground = False

    # x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE)
    # d_min = volume.atmos_upper_limit - volume.planet_r
    # d_max = H
    # D = DistanceToTopAtmosphereBoundary(volume.planet_r, mu_s_min)
    # A = (D - d_min) / (d_max - d_min)
    # a = (A - x_mu_s * A) / (1.0 + x_mu_s * A)
    # d = d_min + min(a, A) * (d_max - d_min)
    # mu_s = 1.0 if d == 0.0 else ClampCosine((H * H - d * d) / (2.0 * volume.planet_r * d))

    # nu = ClampCosine(uvwz.x * 2.0 - 1.0)

    # # Clamp nu to its valid range of values, given mu and mu_s.
    # nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
    #     mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)))

    # return r, mu, mu_s, nu, ray_r_mu_intersects_ground

    # Distance to top atmosphere boundary for a horizontal ray at ground level.
    H = sqrt(volume.atmos_upper_limit * volume.atmos_upper_limit -
        volume.planet_r * volume.planet_r)
    # Distance to the horizon.
    rho = H * uvwz.u
    r = sqrt(rho * rho + volume.planet_r * volume.planet_r)
    mu = uvwz.y * 2.0 - 1.0
    mu_s = inverse_mu_s_mapping(uvwz.z)
    nu = uvwz.w * 2.0 - 1.0
    return r, mu, mu_s, nu, RayIntersectsGround(r, mu)

# Ray Position/Direction to Bruneton Parameters Conversion Functions


@ti.func
def BrunetonToRayParams(r, mu, mu_s, nu):
    """Convert Bruneton parameters to ray position, ray direction, and sun direction.
    
    Args:
        r: Distance from planet center
        mu: Cosine of angle between ray direction and zenith
        mu_s: Cosine of angle between sun direction and zenith
        nu: Cosine of angle between ray and sun directions
    
    Returns:
        ray_pos: Position of ray origin (y is up)
        ray_dir: Direction of ray (normalized)
        sun_dir: Direction of sun (normalized)
    """
    # Ray position (on y-up sphere of radius r)
    ray_pos = vec3(0.0, r, 0.0)  # Start with point on y axis
    
    # Ray direction
    # First, start with direction based on mu (angle from y axis)
    sin_theta = SafeSqrt(1.0 - mu * mu)  # sin of angle with y axis
    ray_dir = vec3(sin_theta, mu, 0.0)  # Initial direction in xy plane
    
    # Sun direction
    # First, get sun direction in xy plane
    sin_theta_s = SafeSqrt(1.0 - mu_s * mu_s)
    sun_dir = vec3(sin_theta_s, mu_s, 0.0)
    
    # Now we need to rotate one of the vectors around y axis to achieve
    # the correct angle (nu) between ray_dir and sun_dir
    # We'll rotate ray_dir
    cos_phi = 0.0
    if abs(sin_theta) > 1e-8 and abs(sin_theta_s) > 1e-8:  # Avoid division by zero
        cos_phi = (nu - mu * mu_s) / (sin_theta * sin_theta_s)
        cos_phi = clamp(cos_phi, -1.0, 1.0)
    sin_phi = SafeSqrt(1.0 - cos_phi * cos_phi)
    
    # Apply the rotation to ray_dir
    ray_dir = vec3(
        ray_dir.x * cos_phi,
        ray_dir.y,
        ray_dir.x * sin_phi
    )
    
    return ray_pos, ray_dir.normalized(), sun_dir.normalized()

@ti.func
def RayParamsToBruneton(ray_pos, ray_dir, sun_dir):
    """Convert ray position and directions to Bruneton parameters.
    
    Args:
        ray_pos: Position of ray origin (y is up)
        ray_dir: Direction of ray (normalized)
        sun_dir: Direction of sun (normalized)
    
    Returns:
        r: Distance from planet center
        mu: Cosine of angle between ray direction and zenith
        mu_s: Cosine of angle between sun direction and zenith
        nu: Cosine of angle between ray and sun directions
    """
    # r is simply the distance from origin
    r = length(ray_pos)
    
    # mu is dot product of ray direction with zenith (y axis)
    mu = dot(ray_pos, ray_dir) / r
    
    # mu_s is dot product of sun direction with zenith (y axis)
    mu_s = dot(ray_pos, sun_dir) / r
    
    # nu is dot product of ray and sun directions
    nu = dot(ray_dir, sun_dir)
    
    return r, mu, mu_s, nu

@ti.func
def RayParamsToUvwz(ray_pos, ray_dir, sun_dir):
    """Convert ray parameters to texture coordinates uvwz.
    
    Args:
        ray_pos: Position of ray origin (y is up)
        ray_dir: Direction of ray (normalized)
        sun_dir: Direction of sun (normalized)
    
    Returns:
        uvwz: Vector4 of texture coordinates
        ray_r_mu_intersects_ground: Whether the ray intersects the ground
    """
    # First convert to Bruneton parameters
    r, mu, mu_s, nu = RayParamsToBruneton(ray_pos, ray_dir, sun_dir)
    
    # Check if ray intersects ground
    ray_r_mu_intersects_ground = RayIntersectsGround(r, mu)
    
    # Convert to texture coordinates
    uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground)
    
    return uvwz

@ti.func
def UvwzToRayParams(uvwz):
    """Convert texture coordinates uvwz to ray parameters.
    
    Args:
        uvwz: Vector4 of texture coordinates
    
    Returns:
        ray_pos: Position of ray origin (y is up)
        ray_dir: Direction of ray (normalized)
        sun_dir: Direction of sun (normalized)
    """
    # First convert to Bruneton parameters
    r, mu, mu_s, nu, ray_r_mu_intersects_ground = GetRMuMuSNuFromScatteringTextureUvwz(uvwz)
    
    # Convert to ray parameters
    ray_pos, ray_dir, sun_dir = BrunetonToRayParams(r, mu, mu_s, nu)
    
    return ray_pos, ray_dir, sun_dir