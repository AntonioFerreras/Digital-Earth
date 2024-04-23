import taichi as ti
import lib.volume_rendering_models as volume
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
import lib.surface_rendering_models as surface
from lib.textures import *

@ti.func
def land_sdf(heightmap: ti.template(), pos: vec3, scale: ti.f32):
    # bump-mapped sphere SDF for a sphere centered at origin
    return length(pos) - volume.planet_r - scale*sample_sphere_texture(heightmap, pos).x
    
@ti.func
def land_normal(heightmap: ti.template(), pos: vec3, scale: ti.f32):
    d = land_sdf(heightmap, pos, scale)

    e = ti.Vector([np.pi*volume.planet_r/TOPOGRAPHY_TEX_RES[0], 0.0])

    n = d - vec3(land_sdf(heightmap, pos - e.xyy, scale),
                 land_sdf(heightmap, pos - e.yxy, scale),
                 land_sdf(heightmap, pos - e.yyx, scale))
    return n.normalized()
    
@ti.func
def intersect_land(heightmap: ti.template(), pos: vec3, dir: vec3, height_scale: ti.f32):
    ray_dist = 0.
    max_ray_dist = volume.planet_r*10.0

    # do RSI on bounding sphere first to cut down starting distance
    rsi_dist = rsi(pos, dir, volume.atmos_upper_limit)
    if rsi_dist.x > 0.0: 
        ray_dist = rsi_dist.x
    
    for i in range(0, 250):
        ro = pos + dir * ray_dist

        dist = land_sdf(heightmap, ro, height_scale)
        ray_dist += dist
            
        if ray_dist > max_ray_dist or abs(dist) < ray_dist*0.0001:
            break
        
    return ray_dist if ray_dist < max_ray_dist else -1.0

@ti.func
def get_clouds_density(clouds_sampler: ti.template(), pos: vec3):
    r = length(pos)
    density = 0.0
    if r > volume.clouds_lower_limit and r < volume.clouds_upper_limit:
        h = (r - volume.clouds_lower_limit)/volume.clouds_thickness
        cloud_texture = sample_sphere_texture(clouds_sampler, pos).r
        # noise = sample_sphere_texture(noise_sampler, pos, 1300.0).r
        # if cloud_texture > 0.01: cloud_texture += (noise - 0.5)
        # noise = sample_sphere_texture(noise_sampler, pos, 2000.0).r
        # cloud_texture += (noise - 0.5)*2.0*cloud_texture
        # cloud_texture = max(cloud_texture, 0.0)
        column_height = cloud_texture
        
        split = 0.2
        density = max(cloud_texture, 0.4) if (h-split < column_height*(1.0 - split) and split-h < column_height*split) else 0.0
    
    return density  * volume.clouds_density

@ti.func
def get_atmos_density(pos: vec3, clouds_sampler: ti.template()):
    rmo = volume.get_density(volume.get_elevation(pos))
    c = get_clouds_density(clouds_sampler, pos)
    return vec4(rmo, c)

@ti.func
def speckle(p: vec2, density: ti.f32):
    m = 0.
    for y in range(-1, 2):
        for x in range(-1, 2):
            q = floor(p) + vec2(x, y) + hash22(floor(p) + vec2(x,y))
            a = 1.5 * -log(1e-4 + (1. - 2e-4) * hash12(q)) * pow(1.5 * clamp(density, 0., 0.67), 1.5)
            dist = distance(p,q)
            pdf = mix(normal_distribution(dist, 0.0, 0.1), normal_distribution(dist, 0.0, 1.0), 0.7)*0.6
            m += a * exp(-6.0 * dist / clamp(density, 0.67, 1.)) # 0.004 / (dist * dist + 0.002)# 
    return m

@ti.func
def get_land_material(albedo_sampler: ti.template(), 
                      ocean_sampler: ti.template(), 
                      bathymetry_sampler: ti.template(), 
                      emissive_sampler: ti.template(), 
                      pos: vec3):
    ocean = (sample_sphere_texture(ocean_sampler, pos).r)
    albedo_texture_srgb = (sample_sphere_texture(albedo_sampler, pos).rgb)

    # darken and desaturate greenery, boost saturation and orange of deserts
    land_albedo_srgb = mix(lum3(albedo_texture_srgb), albedo_texture_srgb, 6.5)
    land_greenery = pow(land_albedo_srgb.y / lum(land_albedo_srgb), 2.0)
    land_greenery = smoothstep(1.5, 1.9, land_greenery)
    land_albedo_srgb = 1.0*albedo_texture_srgb / (land_greenery*0.7 + 1.0)
    land_albedo_srgb = mix(lum3(land_albedo_srgb), land_albedo_srgb, 1.4 - land_greenery*0.45)
    land_albedo_srgb = mix(land_albedo_srgb, land_albedo_srgb * vec3(255.0, 128.0, 64.0)/255.0, 0.2*(1.0 - land_greenery))

    # desaturate ocean albedo
    ocean_albedo_srgb = mix(lum3(albedo_texture_srgb), albedo_texture_srgb, 0.75)*0.9

    # mix land and ocean
    albedo_srgb = mix(land_albedo_srgb, ocean_albedo_srgb, ocean)

    bathymetry = sample_sphere_texture(bathymetry_sampler, pos).r
    emissive_density = sample_sphere_texture(emissive_sampler, pos).r
    # emissive_density = pow(emissive_density, 2.0)
    # smooth_density = 0.5 + 0.5*smoothstep(0.5, 1.0, emissive_density)
    # emissive_factor = speckle(17000. * clamp(emissive_density, 0.4, 1.0) * sphere_UV_map(pos.normalized()), emissive_density)

    return albedo_srgb, ocean, bathymetry, emissive_density
