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
from lib.ray_functions import *


@ti.func
def path_tracer(path: PathParameters,
                scene: SceneParameters,
                albedo_sampler: ti.template(),
                height_sampler: ti.template(),
                ocean_sampler: ti.template(),
                clouds_sampler: ti.template(),
                bathymetry_sampler: ti.template(),
                emissive_sampler: ti.template(),
                stars_sampler: ti.template(),
                srgb_to_spectrum_buff: ti.template(),
                o3_crossec_buff: ti.template()):
    
    ray_pos = path.ray_pos
    ray_dir = path.ray_dir
    
    sun_power = plancks(5778.0, path.wavelength)
    nightlights_power = plancks(2700.0, path.wavelength) * 0.0001
    sun_irradiance = sun_power * cone_angle_to_solid_angle(scene.sun_angular_radius)

    max_densities_rmo = vec3(volume.get_density(0.0).xy, volume.get_ozone_density(volume.ozone_peak_height))
    max_density_cloud = volume.clouds_density

    extinctions = vec4(0.0, 0.0, 0.0, 0.0)
    extinctions.x = volume.spectra_extinction_rayleigh(path.wavelength)
    extinctions.y = volume.spectra_extinction_mie(path.wavelength)
    extinctions.z = volume.spectra_extinction_ozone(path.wavelength, o3_crossec_buff)
    extinctions.w = volume.clouds_extinct

    primary_ray_did_not_intersect = False

    in_scattering = 0.0
    throughput = 1.0
    for scatter_count in range(0, 25):

        if scatter_count > 9: 
            extinctions.w = 0.02
        

        max_extinction_rmo = (extinctions.xyz * max_densities_rmo).sum()
        max_extinction_cloud = extinctions.w * max_density_cloud

        # Intersect ray with surfaces
        earth_intersection = intersect_land(height_sampler, ray_pos, ray_dir, scene.land_height_scale)

        # Sample a particle interaction with volume
        event, interaction_dist, interaction_id = sample_interaction(ray_pos, 
                                                                          ray_dir,
                                                                          earth_intersection,
                                                                          extinctions,
                                                                          max_extinction_rmo,
                                                                          max_extinction_cloud,
                                                                          clouds_sampler)
        if scatter_count > 9 and interaction_id == volume.CLOUD_ID: 
            interaction_id = volume.ISOTROPIC_CLOUD_ID
        
        # Sample a direction to sun
        light_dir = sample_cone_oriented(scene.sun_cos_angle, scene.light_direction)
        if event == ABSORB_EVENT:
            break
        elif event == SCATTER_EVENT:
            ### Volume scattering

            interaction_pos = ray_pos + interaction_dist*ray_dir

            # Direct illumination
            # compute sunlight visibility, phase and transmittance. 
            # no parallax heightmap shadow because its insignificant at atmosphere height.
            direct_visibility = rsi(interaction_pos, light_dir, volume.planet_r).y > 0.0
            direct_transmittance = 0.0
            if not direct_visibility:
                direct_transmittance = sample_transmittance(interaction_pos,
                                                            light_dir,
                                                            -1.0,
                                                            extinctions,
                                                            max_extinction_rmo,
                                                            max_extinction_cloud,
                                                            clouds_sampler)
            direct_phase = volume.evaluate_phase(ray_dir, light_dir, interaction_id, scatter_count > 0)
            in_scattering += throughput * direct_transmittance * sun_irradiance * direct_phase

            scatter_dir, phase_div_pdf = volume.sample_phase(ray_dir, interaction_id, scatter_count > 0)

            ray_dir = scatter_dir
            ray_pos = interaction_pos
            throughput *= phase_div_pdf
                

        elif earth_intersection > 0.0:
            ### Surface scattering
            land_pos = ray_pos + ray_dir*earth_intersection
            sphere_normal = land_pos.normalized()
            land_normal = land_normal(height_sampler, land_pos, scene.land_height_scale)
            albedo_srgb, ocean, bathymetry, emissive_factor = get_land_material(albedo_sampler, 
                                                                                ocean_sampler, 
                                                                                bathymetry_sampler, 
                                                                                emissive_sampler,
                                                                                land_pos)
            albedo = srgb_to_spectrum(srgb_to_spectrum_buff, albedo_srgb, path.wavelength)

            # Emissive term
            in_scattering += throughput * emissive_factor * nightlights_power

            # Direct illumination
            # compute sunlight visibility and transmittance
            offset_pos = land_pos * (1.0 + 0.0001*scene.land_height_scale/12000.0)
            direct_visibility = intersect_land(height_sampler, offset_pos, light_dir, scene.land_height_scale) < 0.0
            # direct_transmittance = 1.0
            direct_transmittance = sample_transmittance(offset_pos,
                                                                light_dir,
                                                                -1.0 if direct_visibility else 0.0,
                                                                extinctions,
                                                                max_extinction_rmo,
                                                                max_extinction_cloud,
                                                                clouds_sampler)
            direct_brdf, direct_n_dot_l = surface.earth_brdf(albedo, ocean, bathymetry, -ray_dir, land_normal, light_dir)
            in_scattering += throughput * direct_transmittance * direct_visibility * sun_irradiance * direct_brdf * direct_n_dot_l

            # Sample scattered ray direction
            view_dir = -ray_dir
            ray_dir = sample_hemisphere_cosine_weighted(land_normal)
            ray_pos = offset_pos
            brdf, _ = surface.earth_brdf(albedo, ocean, bathymetry, view_dir, land_normal, ray_dir)
            throughput *= brdf * np.pi # NdotL and PI in denominator are cancelled due to cosine weighted PDF

        else:

            if scatter_count == 0: primary_ray_did_not_intersect = True
            break
                
                
        # Russian roulette path termination
        if scatter_count > 3:
            termination_p =  max(0.05, 1.0 - throughput)
            if ti.random() < termination_p:
                break
                        
            throughput /= 1.0 - termination_p

    # if primary_ray_did_not_intersect:
    #     # Draw sun for primary ray
    #     if scene.light_direction.dot(path.ray_dir) > scene.sun_cos_angle:
    #         in_scattering += sun_power

    #     # Stars radiance
    #     stars_srgb = sample_sphere_texture(stars_sampler, path.ray_dir).rgb
    #     stars_power = srgb_to_spectrum(srgb_to_spectrum_buff, stars_srgb, path.wavelength)
    #     in_scattering += stars_power * sun_power * 0.0000001


    if isinf(in_scattering) or isnan(in_scattering) or in_scattering < 0.0:
        in_scattering = 0.0

    return in_scattering


@ti.func
def ray_march_rmo(ray_pos: vec3, 
                  ray_dir: vec3,
                  t_start: float,
                  t_max: float,
                  sun_dir: vec3,
                  rmo_extinction: vec3,
                  rm_scattering: vec2,
                  wavelength: float,
                  trans_lut_sampler: ti.template(),
                  multi_scatter_lut_sampler: ti.template(),
                  clouds_sampler: ti.template()):
    steps = 32
    r_steps = 1.0 / (ti.cast(steps, ti.f32))

    dd = (t_max - t_start) * r_steps
    ray_step = ray_dir*dd
    ray_pos = ray_pos + ray_dir * t_start # ray_step*(0.33)

    cos_theta = ray_dir.dot(sun_dir)
    phase = vec2(volume.rayleigh_phase(cos_theta), volume.mie_phase(cos_theta))

    transmittance = 1.0
    in_scatter = 0.0

    for i in range(0, steps) :
        h = volume.get_elevation(ray_pos)
        density = volume.get_density(h)
        step_optical_depth = rmo_extinction.dot(density * dd)
        step_transmittance = saturate(exp(-step_optical_depth))

        step_integral = saturate((1.0 - step_transmittance)/step_optical_depth)
        visible_scattering = transmittance * step_integral

        sun_visibility = not rsi(ray_pos, sun_dir, volume.planet_r).y > 0.0
        # sun_transmittance = ray_march_transmittance(ray_pos, sun_dir, rmo_extinction)
        sun_transmittance = lut_transmittance(sun_dir.dot(ray_pos.normalized()), h, wavelength, trans_lut_sampler)
        # sun_transmittance = 1.0
        multiple_scattering = lut_multiple_scattering(ray_pos, sun_dir, wavelength, multi_scatter_lut_sampler)

        step_single_scattering = rm_scattering.dot(density.xy * phase)
        step_multi_scattering = rm_scattering.dot(density.xy)
        in_scatter += step_single_scattering * sun_visibility * sun_transmittance * visible_scattering * dd
        in_scatter += step_multi_scattering * multiple_scattering * visible_scattering * dd

        transmittance *= step_transmittance

        ray_pos += ray_step

    return in_scatter, transmittance

@ti.func
def ray_marcher(path: PathParameters,
                scene: SceneParameters,
                albedo_sampler: ti.template(),
                height_sampler: ti.template(),
                ocean_sampler: ti.template(),
                clouds_sampler: ti.template(),
                bathymetry_sampler: ti.template(),
                emissive_sampler: ti.template(),
                stars_sampler: ti.template(),
                trans_lut_sampler: ti.template(),
                multi_scatter_lut_sampler: ti.template(),
                srgb_to_spectrum_buff: ti.template(),
                o3_crossec_buff: ti.template()):
    
    ray_pos = path.ray_pos
    ray_dir = path.ray_dir
    
    sun_power = plancks(5778.0, path.wavelength)
    nightlights_power = plancks(2700.0, path.wavelength) * 0.0001
    sun_irradiance = sun_power * cone_angle_to_solid_angle(scene.sun_angular_radius)

    extinctions = vec4(0.0, 0.0, 0.0, 0.0)
    extinctions.x = volume.spectra_extinction_rayleigh(path.wavelength)
    extinctions.y = volume.spectra_extinction_mie(path.wavelength)
    extinctions.z = volume.spectra_extinction_ozone(path.wavelength, o3_crossec_buff)
    extinctions.w = volume.clouds_extinct

    scattering = vec2(extinctions.x * volume.rayleigh_albedo, extinctions.y * volume.aerosol_albedo)

    primary_ray_did_not_intersect = False

    accum = 0.0
    throughput = 1.0
    for scatter_count in range(0, 3):

        # Intersect ray
        # TODO: Move to its own function
        earth_intersection = intersect_land(height_sampler, ray_pos, ray_dir, scene.land_height_scale)
        atmos_isection = rsi(ray_pos, ray_dir, volume.atmos_upper_limit)
        t_start = max(0.0, atmos_isection.x)
        t_max = earth_intersection if earth_intersection > 0.0 else atmos_isection.y

        if atmos_isection.y < 0.0: 
            # ray doesnt cross atmosphere
            primary_ray_did_not_intersect = scatter_count == 0
            break
        
        
        # Sample a direction to sun
        light_dir = sample_cone_oriented(scene.sun_cos_angle, scene.light_direction)

        in_scatter, transmittance = ray_march_rmo(ray_pos, 
                                                  ray_dir, 
                                                  t_start, t_max, 
                                                  light_dir, 
                                                  extinctions.xyz, 
                                                  scattering, 
                                                  path.wavelength,
                                                  trans_lut_sampler,
                                                  multi_scatter_lut_sampler,
                                                  clouds_sampler)

        accum += throughput * in_scatter * sun_irradiance
        throughput *= transmittance



        # Surface scattering
        if earth_intersection > 0.0:

            land_pos = ray_pos + ray_dir*earth_intersection
            sphere_normal = land_pos.normalized()
            land_normal = land_normal(height_sampler, land_pos, scene.land_height_scale)
            albedo_srgb, ocean, bathymetry, emissive_factor = get_land_material(albedo_sampler, 
                                                                                ocean_sampler, 
                                                                                bathymetry_sampler, 
                                                                                emissive_sampler,
                                                                                land_pos)
            albedo = srgb_to_spectrum(srgb_to_spectrum_buff, albedo_srgb, path.wavelength)

            # Emissive term
            accum += throughput * emissive_factor * nightlights_power

            # Direct illumination
            # compute sunlight visibility and transmittance
            offset_pos = land_pos * (1.0 + 0.0001*scene.land_height_scale/12000.0)
            direct_visibility = intersect_land(height_sampler, offset_pos, light_dir, scene.land_height_scale) < 0.0
            # direct_transmittance = ray_march_transmittance(offset_pos, light_dir, extinctions.xyz)
            direct_transmittance = lut_transmittance(light_dir.dot(sphere_normal), 
                                                     volume.get_elevation(offset_pos), 
                                                     path.wavelength, 
                                                     trans_lut_sampler)
            direct_brdf, direct_n_dot_l = surface.earth_brdf(albedo, ocean, bathymetry, -ray_dir, land_normal, light_dir)
            accum += throughput * direct_transmittance * direct_visibility * sun_irradiance * direct_brdf * direct_n_dot_l

            # Sample scattered ray direction
            view_dir = -ray_dir
            ray_dir = sample_hemisphere_cosine_weighted(land_normal)
            ray_pos = offset_pos
            brdf, _ = surface.earth_brdf(albedo, ocean, bathymetry, view_dir, land_normal, ray_dir)
            throughput *= brdf * np.pi # NdotL and PI in denominator are cancelled due to cosine weighted PDF
        else:
            primary_ray_did_not_intersect = scatter_count == 0
            break

    # if primary_ray_did_not_intersect:
    #     # Draw sun for primary ray
    #     if scene.light_direction.dot(path.ray_dir) > scene.sun_cos_angle:
    #         accum += sun_power

    #     # Stars radiance
    #     stars_srgb = sample_sphere_texture(stars_sampler, path.ray_dir).rgb
    #     stars_power = srgb_to_spectrum(srgb_to_spectrum_buff, stars_srgb, path.wavelength)
    #     accum += stars_power * sun_power * 0.0000001


    if isinf(accum) or isnan(accum) or accum < 0.0:
        accum = 0.0

    return accum