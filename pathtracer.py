import taichi as ti
import lib.volume_rendering_models as volume
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
import lib.surface_rendering_models as surface
from lib.textures import *
from lib.parameters import PathParameters, SceneParameters

@ti.func
def land_sdf(heightmap: ti.template(), pos, scale):
    # bump-mapped sphere SDF for a sphere centered at origin
    return length(pos) - volume.planet_r - scale*sample_sphere_texture(heightmap, pos).x
    
@ti.func
def land_normal(heightmap: ti.template(), pos, scale):
    d = land_sdf(heightmap, pos, scale)

    e = ti.Vector([0.5*2.0*np.pi*volume.planet_r/TOPOGRAPHY_TEX_RES[0], 0.0])

    n = d - vec3(land_sdf(heightmap, pos - e.xyy, scale),
                 land_sdf(heightmap, pos - e.yxy, scale),
                 land_sdf(heightmap, pos - e.yyx, scale))
    return n.normalized()
    
@ti.func
def intersect_land(heightmap: ti.template(), pos, dir, height_scale):
    ray_dist = 0.
    max_ray_dist = volume.planet_r*10.0
    
    for i in range(0, 150):
        ro = pos + dir * ray_dist

        dist = land_sdf(heightmap, ro, height_scale)
        ray_dist += dist
            
        if ray_dist > max_ray_dist or abs(dist) < ray_dist*0.0001:
            break
        
    return ray_dist if ray_dist < max_ray_dist else -1.0

@ti.func
def path_tracer(path: PathParameters,
                scene: SceneParameters,
                albedo_sampler: ti.template(),
                height_sampler: ti.template(),
                ocean_sampler: ti.template(),
                srgb_to_spectrum_buff: ti.template()):
    
    ray_pos = path.ray_pos
    ray_dir = path.ray_dir
    in_scattering = 0.0
    throughput = 1.0

            
    for scatter_count in range(0, 5):
        # Intersect ray with surfaces
        earth_intersection = intersect_land(height_sampler, ray_pos, ray_dir, scene.land_height_scale)


                
        if earth_intersection > 0.0:
            # Surface interaction

            land_pos = ray_pos + ray_dir*earth_intersection
            sphere_normal = land_pos.normalized()
            land_normal = land_normal(height_sampler, land_pos, scene.land_height_scale)
            ocean = (sample_sphere_texture(ocean_sampler, land_pos).r)
            land_albedo_srgb = (sample_sphere_texture(albedo_sampler, land_pos).rgb)
            ocean_albedo_srgb = mix(lum3(land_albedo_srgb), land_albedo_srgb, 0.3)
            albedo_srgb = mix(land_albedo_srgb, ocean_albedo_srgb, ocean)
            albedo = srgb_to_spectrum(srgb_to_spectrum_buff, albedo_srgb, path.wavelength)

            # Sample sun light
            offset_pos = land_pos * (1.0 + 0.0001*scene.land_height_scale/12000.0)
            light_dir = sample_cone_oriented(scene.sun_cos_angle, scene.light_direction)
            earth_land_shadow = intersect_land(height_sampler, offset_pos, light_dir, scene.land_height_scale) < 0.0

            # Ground lighting
            sun_irradiance = plancks(5778.0, path.wavelength) * cone_angle_to_solid_angle(scene.sun_angular_radius)
            brdf, n_dot_l = surface.earth_brdf(albedo, ocean, -ray_dir, land_normal, light_dir)
            in_scattering += earth_land_shadow * sun_irradiance * brdf * n_dot_l

            # Sample scattered ray direction
            view_dir = -ray_dir
            ray_dir = sample_hemisphere_cosine_weighted(land_normal)
            ray_pos = offset_pos
            brdf, n_dot_l = surface.earth_brdf(albedo, ocean, view_dir, land_normal, ray_dir)
            throughput *= brdf * np.pi

        else:
            break
                
                
        # Russian roulette path termination
        if scatter_count > 3:
            termination_p = max(0.05, 1.0 - throughput)
            if ti.random() < termination_p:
                break
                    
            throughput /= 1.0 - termination_p

    return in_scattering

