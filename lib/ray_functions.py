import taichi as ti
import lib.volume_rendering_models as volume
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
import lib.surface_rendering_models as surface
from lib.textures import *
from lib.parameters import PathParameters, SceneParameters
from lib.geometry import *
from lib.lut_sampling import *

NULL_EVENT = 0
ABSORB_EVENT = 1
SCATTER_EVENT = 2

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
    event = NULL_EVENT

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
            
            if volume.sample_scatter_event(interaction_id):
                event = SCATTER_EVENT
            else:
                event = ABSORB_EVENT
            break

        
    return event, t, interaction_id

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
    rmo_event, rmo_t, rmo_id = sample_interaction_delta_tracking(ray_pos, ray_dir, t_start, t_max, rmo_extinctions, max_extinction_rmo, clouds_sampler)


    t_start, t_max = intersect_cloud_limits(ray_pos, ray_dir, land_isection)

    event = rmo_event
    t = rmo_t
    interaction_id = rmo_id

    if rmo_event == NULL_EVENT or rmo_t > t_start:

        cloud_extinctions = vec4(0.0, 0.0, 0.0, extinctions.w)
        cloud_event, cloud_t, _ = sample_interaction_delta_tracking(ray_pos, ray_dir, t_start, t_max, cloud_extinctions, max_extinction_cloud, clouds_sampler)

        
        
        if cloud_event > 0 and (cloud_t < rmo_t or rmo_event == NULL_EVENT): 
            t = cloud_t
            interaction_id = volume.CLOUD_ID
            event = cloud_event

    return event, t, interaction_id
    


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
def ray_march_transmittance(ray_pos: vec3, ray_dir: vec3, rmo_extinction: vec3, steps: int = 16):
    r_steps = 1.0 / ti.cast(steps, ti.f32)
    transmittance = 0.0

    
    visibility = rsi(ray_pos, ray_dir, volume.planet_r).y > 0.0
    if not visibility:

        atmos_isection = rsi(ray_pos, ray_dir, volume.atmos_upper_limit)
        
        
        # t_start = max(0.0, atmos_isection.x)
        t_max = atmos_isection.y
        if atmos_isection.y < 0.0: 
            t_max = -1.0 # ray doesnt cross atmosphere

        dd = t_max * r_steps
        ray_step = ray_dir*dd

        od = vec3(0.0)
        for i in range(0, steps):
            density = volume.get_density(volume.get_elevation(ray_pos))
            od += density * dd

            ray_pos += ray_step

        transmittance = exp(-rmo_extinction.dot(od))
    return transmittance

@ti.func
def lut_transmittance(cos_theta: ti.f32, h: ti.f32, wavelength: ti.f32, transmittance_lut: ti.template()):
    # lookup_uvw = clamp(vec3(pow((cos_theta + 0.19)/1.19, 1.0/2.0), 
    #                                          h/volume.atmos_height, 
    #                                          (wavelength-390.0)/441.0), 0.0, 1.0)
    # lut_dims = vec3(TRANS_LUT_RES[0], TRANS_LUT_RES[1], TRANS_LUT_RES[2])
    # lookup_uvw *= ((lut_dims - 1.0) / lut_dims) + 0.5 / lut_dims
    lookup_uvw = vec3(0.0)
    lookup_uvw.xy = mu_r_to_uv(cos_theta, h + volume.planet_r)
    lookup_uvw.z = (wavelength-390.0)/441.0

    return 0.0 if cos_theta < -0.19 else transmittance_lut.sample_lod(lookup_uvw, 
                                             0.0).x