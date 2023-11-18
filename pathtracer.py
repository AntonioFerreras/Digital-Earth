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
def land_sdf(heightmap: ti.template(), pos: vec3, scale: ti.f32):
    # bump-mapped sphere SDF for a sphere centered at origin
    return length(pos) - volume.planet_r - scale*sample_sphere_texture(heightmap, pos).x
    
@ti.func
def land_normal(heightmap: ti.template(), pos: vec3, scale: ti.f32):
    d = land_sdf(heightmap, pos, scale)

    e = ti.Vector([0.5*2.0*np.pi*volume.planet_r/TOPOGRAPHY_TEX_RES[0], 0.0])

    n = d - vec3(land_sdf(heightmap, pos - e.xyy, scale),
                 land_sdf(heightmap, pos - e.yxy, scale),
                 land_sdf(heightmap, pos - e.yyx, scale))
    return n.normalized()
    
@ti.func
def intersect_land(heightmap: ti.template(), pos: vec3, dir: vec3, height_scale: ti.f32):
    ray_dist = 0.
    max_ray_dist = volume.planet_r*10.0
    
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
        column_height = cloud_texture
        density = cloud_texture if h < column_height else 0.0
    
    return density * volume.clouds_density

@ti.func
def get_atmos_density(pos: vec3, clouds_sampler: ti.template()):
    rmo = volume.get_density(volume.get_elevation(pos))
    c = get_clouds_density(clouds_sampler, pos)
    return vec4(rmo, c)

@ti.func
def sample_interaction_delta_tracking(ray_pos: vec3, 
                                      ray_dir: vec3,
                                      t_start: float,
                                      t_max: float,
                                      extinctions: vec4,
                                      max_extinction: float,
                                      clouds_sampler: ti.template()):
    t = t_start
    ray_pos += t*ray_dir

    interaction_id = 0
    interacted = False

    while t < t_max:
        t_step = -log(ti.random()) / max_extinction
        ray_pos += t_step*ray_dir
        t += t_step

        if (t >= t_max): break
        
        extinction_sample = extinctions * get_atmos_density(ray_pos, clouds_sampler)

        rand = ti.random()
        if rand < extinction_sample.sum() / max_extinction:
            cmf = 0.0
            while interaction_id < 3:
                cmf += extinction_sample[interaction_id]
                if rand < cmf / max_extinction: break
                interaction_id += 1
            interacted = True
            break

        
    return interacted, t, interaction_id

@ti.func
def transmittance_ratio_tracking(ray_pos: vec3, 
                                 ray_dir: vec3,
                                 t_start: float,
                                 t_max: float,
                                 extinctions: vec4,
                                 max_extinction: float,
                                 clouds_sampler: ti.template()):
    t = t_start
    ray_pos += t*ray_dir

    transmittance = 1.0

    while t < t_max:
        t_step = -log(ti.random()) / max_extinction
        ray_pos += t_step*ray_dir
        t += t_step

        if t >= t_max: break
        
        extinction_sample = extinctions * get_atmos_density(ray_pos, clouds_sampler)

        transmittance *= 1.0 - extinction_sample.sum() / max_extinction

        if transmittance < 1e-5: break
        
    return transmittance

@ti.func
def intersect_cloud_limits(ray_pos: vec3, ray_dir: vec3, land_isection: float):
    t_start = 0.0
    t_max = 0.0
    elevation =  length(ray_pos)

    cloud_lower_isection = rsi(ray_pos, ray_dir, volume.clouds_lower_limit)
    cloud_upper_isection = rsi(ray_pos, ray_dir, volume.clouds_upper_limit)
    if elevation >= volume.clouds_upper_limit:
        t_start = max(0.0, cloud_upper_isection.x)
        t_max = cloud_lower_isection.x if cloud_lower_isection.y >= 0.0 else cloud_upper_isection.y

        if cloud_upper_isection.y < 0.0: t_max = -1.0
    elif elevation >= volume.clouds_lower_limit:
        t_start = 0.0
        # t_max = cloud_upper_isection.y if cloud_lower_isection.y <= 0.0 else cloud_lower_isection.x
        t_max   = cloud_lower_isection.x if cloud_lower_isection.y >= 0.0 else cloud_upper_isection.y
    else:
        t_start = cloud_lower_isection.y
        t_max = cloud_upper_isection.y

        if land_isection > 0.0: t_max = -1.0
    
    
    return t_start, t_max


@ti.func
def sample_interaction(ray_pos: vec3, 
                       ray_dir: vec3,
                       land_isection: float,
                       extinctions: vec4,
                       max_extinction_rmo: float,
                       max_extinction_cloud: float,
                       clouds_sampler: ti.template()):
    atmos_isection = rsi(ray_pos, ray_dir, volume.atmos_upper_limit)
    t_start = max(0.0, atmos_isection.x)
    t_max = land_isection if land_isection >= 0.0 else atmos_isection.y
    if atmos_isection.y < 0.0: 
        t_max = -1.0 # ray doesnt cross atmosphere
    rmo_extinctions = vec4(extinctions.xyz, 0.0)
    rmo_interacted, rmo_t, rmo_id = sample_interaction_delta_tracking(ray_pos, ray_dir, t_start, t_max, rmo_extinctions, max_extinction_rmo, clouds_sampler)

    t_start, t_max = intersect_cloud_limits(ray_pos, ray_dir, land_isection)
    cloud_extinctions = vec4(0.0, 0.0, 0.0, extinctions.w)
    cloud_interacted, cloud_t, _ = sample_interaction_delta_tracking(ray_pos, ray_dir, t_start, t_max, cloud_extinctions, max_extinction_cloud, clouds_sampler)

    interacted = rmo_interacted or cloud_interacted
    t = 0.0
    interaction_id = 0
    if rmo_interacted: 
        t = rmo_t
        interaction_id = rmo_id
    if cloud_interacted and (cloud_t < rmo_t or not rmo_interacted): 
        t = cloud_t
        interaction_id = 3

    return interacted, t, interaction_id
    


@ti.func
def sample_transmittance(ray_pos: vec3, 
                         ray_dir: vec3,
                         land_isection: float,
                         extinctions: vec4,
                         max_extinction_rmo: float,
                         max_extinction_cloud: float,
                         clouds_sampler: ti.template()):
    atmos_isection = rsi(ray_pos, ray_dir, volume.atmos_upper_limit)

    
    t_start = max(0.0, atmos_isection.x)
    t_max = land_isection if land_isection >= 0.0 else atmos_isection.y
    if atmos_isection.y < 0.0: 
        t_max = -1.0 # ray doesnt cross atmosphere
    rmo_extinctions = vec4(extinctions.xyz, 0.0)
    transmittance  = transmittance_ratio_tracking(ray_pos, ray_dir, t_start, t_max, rmo_extinctions, max_extinction_rmo, clouds_sampler)
    
    t_start, t_max = intersect_cloud_limits(ray_pos, ray_dir, land_isection)
    cloud_extinctions = vec4(0.0, 0.0, 0.0, extinctions.w)
    transmittance *= transmittance_ratio_tracking(ray_pos, ray_dir, t_start, t_max, cloud_extinctions, max_extinction_cloud, clouds_sampler)
    return transmittance


@ti.func
def evaluate_phase(ray_dir: vec3, light_dir: vec3, interaction_id: ti.i32):
    phase = 0.0
    cos_theta = ray_dir.dot(light_dir)
    if interaction_id == 0:
        phase += volume.rayleigh_phase(cos_theta)
    elif interaction_id == 1:
        phase += volume.mie_phase(cos_theta)
    elif interaction_id == 3:
        phase += volume.cloud_phase(cos_theta)
    return phase
    
@ti.func
def sample_phase(ray_dir: vec3, interaction_id: ti.i32):
    sample_dir = vec3(0.0, 0.0, 0.0)
    phase_div_pdf = 1.0
    if interaction_id != 3:
        sample_dir = sample_sphere(vec2(ti.random(), ti.random()))
        phase_div_pdf = evaluate_phase(ray_dir, sample_dir, interaction_id) * (4.0 * np.pi)
    else:
        sample_dir = volume.sample_cloud_phase(ray_dir)
    return sample_dir, phase_div_pdf

@ti.func
def sample_scatter_event(interaction_id: ti.i32):
    scattering_albedos = ti.Vector([volume.rayleigh_albedo, 
                                    volume.aerosol_albedo, 
                                    volume.ozone_albedo, 
                                    volume.cloud_albedo])
    return ti.random() < scattering_albedos[interaction_id]

@ti.func
def path_tracer(path: PathParameters,
                scene: SceneParameters,
                albedo_sampler: ti.template(),
                height_sampler: ti.template(),
                ocean_sampler: ti.template(),
                clouds_sampler: ti.template(),
                srgb_to_spectrum_buff: ti.template()):
    
    ray_pos = path.ray_pos
    ray_dir = path.ray_dir
    in_scattering = 0.0
    throughput = 1.0
    sun_irradiance = plancks(5778.0, path.wavelength) * cone_angle_to_solid_angle(scene.sun_angular_radius)
    max_densities_rmo = vec3(volume.get_density(0.0).xy, volume.get_ozone_density(volume.ozone_peak_height))
    max_density_cloud = volume.clouds_density
    
    for scatter_count in range(0, 50):

        extinctions = vec4(0.0, 0.0, 0.0, 0.0)
        extinctions.x = volume.spectra_extinction_rayleigh(path.wavelength)
        extinctions.y = volume.spectra_extinction_mie(path.wavelength)
        extinctions.z = volume.spectra_extinction_ozone(path.wavelength)
        extinctions.w = volume.clouds_extinct
        

        max_extinction_rmo = (extinctions.xyz * max_densities_rmo).sum()
        max_extinction_cloud = extinctions.w * max_density_cloud

        # Intersect ray with surfaces
        earth_intersection = intersect_land(height_sampler, ray_pos, ray_dir, scene.land_height_scale)

        # Sample a particle interaction with volume
        interacted, interaction_dist, interaction_id = sample_interaction(ray_pos, 
                                                                          ray_dir,
                                                                          earth_intersection,
                                                                          extinctions,
                                                                          max_extinction_rmo,
                                                                          max_extinction_cloud,
                                                                          clouds_sampler)
        # Sample a direction to sun
        light_dir = sample_cone_oriented(scene.sun_cos_angle, scene.light_direction)

        if interacted:
            ### Volume scattering

            interaction_pos = ray_pos + interaction_dist*ray_dir

            # Direct illumination
            # compute sunlight visibility, phase and transmittance. 
            # no parallax heightmap shadow because its insignificant at atmosphere scale.
            direct_visibility = rsi(interaction_pos, light_dir, volume.planet_r).x < 0.0
            direct_transmittance = sample_transmittance(interaction_pos,
                                                                light_dir,
                                                                -1.0 if direct_visibility else 0.0,
                                                                extinctions,
                                                                max_extinction_rmo,
                                                                max_extinction_cloud,
                                                                clouds_sampler)
            direct_phase = evaluate_phase(ray_dir, light_dir, interaction_id)
            in_scattering += throughput * direct_transmittance * direct_visibility * sun_irradiance * direct_phase

            # Sample scattered ray direction (if scattering event)
            if sample_scatter_event(interaction_id):
                scatter_dir, phase_div_pdf = sample_phase(ray_dir, interaction_id)

                ray_dir = scatter_dir
                ray_pos = interaction_pos
                throughput *= phase_div_pdf
                
            else:
                break # absorbed



        elif earth_intersection > 0.0:
            #### Surface scattering
            land_pos = ray_pos + ray_dir*earth_intersection
            sphere_normal = land_pos.normalized()
            land_normal = land_normal(height_sampler, land_pos, scene.land_height_scale)
            ocean = (sample_sphere_texture(ocean_sampler, land_pos).r)
            albedo_texture_srgb = (sample_sphere_texture(albedo_sampler, land_pos).rgb)
            land_albedo_srgb = mix(lum3(albedo_texture_srgb), albedo_texture_srgb, 0.7)
            ocean_albedo_srgb = mix(lum3(albedo_texture_srgb), albedo_texture_srgb, 0.25)
            albedo_srgb = mix(land_albedo_srgb, ocean_albedo_srgb, ocean)
            albedo = srgb_to_spectrum(srgb_to_spectrum_buff, albedo_srgb, path.wavelength)

            # Direct illumination
            # compute sunlight visibility and transmittance
            offset_pos = land_pos * (1.0 + 0.0001*scene.land_height_scale/12000.0)
            direct_visibility = intersect_land(height_sampler, offset_pos, light_dir, scene.land_height_scale) < 0.0
            direct_transmittance = sample_transmittance(offset_pos,
                                                                light_dir,
                                                                -1.0 if direct_visibility else 0.0,
                                                                extinctions,
                                                                max_extinction_rmo,
                                                                max_extinction_cloud,
                                                                clouds_sampler)
            direct_brdf, direct_n_dot_l = surface.earth_brdf(albedo, ocean, -ray_dir, land_normal, light_dir)
            in_scattering += throughput * direct_transmittance * direct_visibility * sun_irradiance * direct_brdf * direct_n_dot_l

            # Sample scattered ray direction
            view_dir = -ray_dir
            ray_dir = sample_hemisphere_cosine_weighted(land_normal)
            ray_pos = offset_pos
            brdf, _ = surface.earth_brdf(albedo, ocean, view_dir, land_normal, ray_dir)
            throughput *= brdf * np.pi # NdotL and PI in denominator are cancelled due to cosine weighted PDF

        else:
            break
                
                
        # Russian roulette path termination
        if scatter_count > 3:
            termination_p =  max(0.05, 1.0 - throughput)
            if ti.random() < termination_p:
                break
                        
            throughput /= 1.0 - termination_p

    return in_scattering

