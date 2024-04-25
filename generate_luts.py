import taichi as ti
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
from lib.textures import *
import lib.volume_rendering_models as volume
from lib.parameters import PathParameters, SceneParameters
from lib.geometry import *
from lib.ray_functions import *
from PIL import Image
from lib.lut_sampling import *

ti.init(arch=ti.vulkan)


transmittance_lut = ti.field(dtype=ti.f32, shape=TRANS_LUT_RES)
multiple_scattering_lut = ti.field(dtype=ti.f32, shape=MULTISCAT_LUT_RES)

transmittance_lut_tex = ti.Texture(ti.Format.r32f, TRANS_LUT_RES)
multiple_scattering_lut_tex0 = ti.Texture(ti.Format.r32f, MULTISCAT_LUT_RES)
multiple_scattering_lut_tex1 = ti.Texture(ti.Format.r32f, MULTISCAT_LUT_RES)

O3_crossec_LUT_buff = ti.field(dtype=ti.f32, shape=(O3_CROSSEC_LUT_RES))
with open(O3_CROSSEC_LUT_FILE, 'rb') as file:
    load_data = np.fromfile(file, dtype=np.float32, count=O3_CROSSEC_LUT_RES)
data_array = np.zeros(shape=(O3_CROSSEC_LUT_RES), dtype=np.float32)
for x in range (0, O3_CROSSEC_LUT_RES):
    data_array[x] = load_data[x]
O3_crossec_LUT_buff.from_numpy(data_array)



@ti.kernel
def compute_transmittance_lut(o3_crossec_buff: ti.template(), tex: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0)):
    for x, y, z in transmittance_lut:
        xyz = ti.Vector([ti.cast(x, ti.f32)/TRANS_LUT_RES[0], 
                         ti.cast(y, ti.f32)/TRANS_LUT_RES[1], 
                         ti.cast(z, ti.f32)/TRANS_LUT_RES[2]])
        cos_theta, h = uv_to_mu_r(xyz.xy)
        wavelength = 390.0 + xyz.z*441.0

        theta = acos(cos_theta)
        sin_theta = sin(theta)

        ray_pos = vec3(0.0, volume.planet_r + h, 0.0)
        ray_dir = vec3(sin_theta, cos_theta, 0.0)

        extinctions = vec3(0.0, 0.0, 0.0)
        extinctions.x = volume.spectra_extinction_rayleigh(wavelength)
        extinctions.y = volume.spectra_extinction_mie(wavelength)
        extinctions.z = volume.spectra_extinction_ozone(wavelength, o3_crossec_buff)
        
        transmittance = ray_march_transmittance(ray_pos, ray_dir, extinctions, 256)

        transmittance_lut[x, y, z] = transmittance
        tex.store(ti.Vector([x, y, z]), ti.Vector([transmittance, 0.0, 0.0, 0.0]))

@ti.func
def ray_march_scattering(ray_pos: vec3, 
                            ray_dir: vec3,
                            t_start: float,
                            t_max: float,
                            sun_dir: vec3,
                            rmo_extinction: vec3,
                            rm_scattering: vec2,
                            wavelength: float,
                            trans_lut_sampler: ti.template(),
                            multi_lut_sampler: ti.template(),
                            iteration: ti.i32):
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

        if iteration == 0:
            sun_visibility = not rsi(ray_pos, sun_dir, volume.planet_r).y > 0.0
            # sun_transmittance = ray_march_transmittance(ray_pos, sun_dir, rmo_extinction)
            sun_transmittance = lut_transmittance(sun_dir.dot(ray_pos.normalized()), h, wavelength, trans_lut_sampler)

            step_scattering = rm_scattering.dot(density.xy * phase) * dd
            in_scatter += step_scattering * sun_visibility * sun_transmittance * visible_scattering
        else:
            multiple_scattering = lut_multiple_scattering(ray_pos, sun_dir, wavelength, multi_lut_sampler)
            step_scattering = rm_scattering.dot(density.xy * phase) * dd
            multipleScatteringEnergy = 1.0
            in_scatter += multipleScatteringEnergy * step_scattering * multiple_scattering * visible_scattering

        transmittance *= step_transmittance

        ray_pos += ray_step

    return in_scatter, transmittance

@ti.kernel
def compute_multiple_scattering_lut(o3_crossec_buff: ti.template(), 
                                    transmittance_lut_sampler: ti.types.texture(num_dimensions=3), 
                                    multiple_scattering_lut_read: ti.types.texture(num_dimensions=3), 
                                    multiple_scattering_lut_write: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0),
                                    z: ti.i32,
                                    iterations: ti.i32):
    for x, y in ti.ndrange(MULTISCAT_LUT_RES[0], MULTISCAT_LUT_RES[1]):
        wavelength = 390.0 + z + 0.5

        sun_angle_coordinate = (ti.cast(x, ti.f32) + 0.5) / ti.cast(MULTISCAT_LUT_RES[0], ti.f32)
        view_height_coordinate = (ti.cast(y, ti.f32) + 0.5) / ti.cast(MULTISCAT_LUT_RES[1], ti.f32)

        view_height = volume.atmos_height * pow(view_height_coordinate, 2.0)

        sun_angle = (2.0 * sun_angle_coordinate) - 1.0
        sun_angle = acos(sign(sun_angle) * pow(sun_angle, 2.0))

        ray_pos = vec3(0.0, view_height + volume.planet_r, 0.0)
        sun_dir = vec3(0.0, cos(sun_angle), sin(sun_angle))

        extinctions = vec3(0.0, 0.0, 0.0)
        extinctions.x = volume.spectra_extinction_rayleigh(wavelength)
        extinctions.y = volume.spectra_extinction_mie(wavelength)
        extinctions.z = volume.spectra_extinction_ozone(wavelength, O3_crossec_LUT_buff)

        scattering = vec2(extinctions.x * volume.rayleigh_albedo, extinctions.y * volume.aerosol_albedo)

        

        accum = 0.0
        MS_SAMPLE_COUNT = 512
        for i in range(0, MS_SAMPLE_COUNT):
            sample_dir = golden_spiral_sample(i, MS_SAMPLE_COUNT) 

            earth_intersection = rsi(ray_pos, sample_dir, volume.planet_r).x
            atmos_isection = rsi(ray_pos, sample_dir, volume.atmos_upper_limit)
            t_max = earth_intersection if earth_intersection > 0.0 else atmos_isection.y



            in_scatter, transmittance = ray_march_scattering(ray_pos, 
                                                            sample_dir, 
                                                            0.0, t_max, 
                                                            sun_dir, 
                                                            extinctions, 
                                                            scattering, 
                                                            wavelength, 
                                                            transmittance_lut_sampler,
                                                            multiple_scattering_lut_read,
                                                            iterations)
            accum += in_scatter
            if iterations == 0:
                # ground albedo
                ground_pos = ray_pos + sample_dir * t_max
                ground_normal = ground_pos.normalized()
                sun_transmittance = lut_transmittance(sun_dir.dot(ground_pos.normalized()), 0.0, wavelength, transmittance_lut_sampler)
                ground_irradiance = volume.ground_albedo * max(ground_normal.dot(sun_dir), 0.0) * sun_transmittance

                accum += ground_irradiance * transmittance


        accum /= ti.cast(MS_SAMPLE_COUNT, ti.f32)
        prev = multiple_scattering_lut[x, y, z] if iterations > 0 else 0.0
        multiple_scattering_lut[x, y, z] = prev + accum
        multiple_scattering_lut_write.store(ti.Vector([x, y, z]), ti.Vector([prev + accum, 0.0, 0.0, 0.0]))



        



compute_transmittance_lut(O3_crossec_LUT_buff, transmittance_lut_tex)
SCATTER_EVENTS = 15
for i in range(0, SCATTER_EVENTS):
    write_to_0 = i % 2 == 0
    last_percent = 0
    for z in range(0, MULTISCAT_LUT_RES[2]):
        compute_multiple_scattering_lut(O3_crossec_LUT_buff, 
                                        transmittance_lut_tex, 
                                        multiple_scattering_lut_tex0 if not write_to_0 else multiple_scattering_lut_tex1, 
                                        multiple_scattering_lut_tex0 if write_to_0 else multiple_scattering_lut_tex1, 
                                        z,
                                        i)
        percent = int((z / MULTISCAT_LUT_RES[2]) * 100)
        if percent == 99:
            percent = 100

        ti.sync()
        if percent % 25 == 0 and percent != last_percent:
            print(f"iteration {i+1}/{SCATTER_EVENTS} {percent}%")
            last_percent = percent

array = transmittance_lut.to_numpy()
array.tofile('LUT/transmittance_lut.dat')

array = multiple_scattering_lut.to_numpy()
array.tofile('LUT/multiple_scattering_lut.dat')

# Read from the newly created file into a new array
# read_array = np.fromfile('LUT/transmittance_lut.dat', dtype=np.float32)
# read_array = read_array.reshape(transmittance_lut.shape)

# Verify that the contents after reading match
# if read_array.shape == transmittance_lut.shape:
#     difference = np.abs(read_array - transmittance_lut.to_numpy())
#     if np.all(difference < 0.0001):
#         print("The contents match.")
#     else:
#         print("The contents do not match.")
# else:
#     print("The shapes do not match.")


slice_000 = array[:, :, 200]
image = Image.fromarray((slice_000 * 255 * 10).astype(np.uint8))
image.save('slice_000.png')

print("Done")