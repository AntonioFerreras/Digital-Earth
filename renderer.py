import os
import taichi as ti
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
from lib.textures import *
from lib.parameters import PathParameters, SceneParameters
from lib.OpenDRT import openDR_transform
import lib.AgX as agx
import pathtracer as pt



@ti.data_oriented
class Renderer:
    def __init__(self, image_res, up):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.aspect_scale = ti.field(dtype=ti.f32, shape=())

        self.exposure = ti.field(dtype=ti.f32, shape=())
        self.selected_crf = ti.field(dtype=ti.i32, shape=())
        self.crf_count = ti.field(dtype=ti.i32, shape=())
        self.gamma = ti.field(dtype=ti.f32, shape=())


        self.sun_angle = ti.field(dtype=ti.f32, shape=())
        self.sun_path_rot = ti.field(dtype=ti.f32, shape=())

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        # By interleaving with 16x8 blocks,
        # each thread block will process 16x8 pixels in a batch instead of a 32 pixel row in a batch
        # Thus we pay less divergence penalty on hard paths
        ti.root.dense(ti.ij, (image_res[0] // 16, image_res[1] // 8)).dense(ti.ij, (16, 8)).place(self.color_buffer)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(np.radians(27.)*0.5)
        self.set_aspect_scale(1.0)
        self.set_exposure(2.5)
        self.set_gamma(1.0)
        self.set_crf(0)
        self.set_sun_angle(np.radians(60.0))
        self.set_sun_path_rot(np.radians(-45.0))

        self.land_height_scale = 7800.0

        # Load Textures
        self.albedo_tex = ti.Texture(ti.Format.rgba8, ALBEDO_TEX_RES)
        self.albedo_buff = ti.Vector.field(3, dtype=ti.u8, shape=ALBEDO_TEX_RES)
        load_image = ti.tools.imread(ALBEDO_TEX_FILE)
        self.albedo_buff.from_numpy(load_image)

        self.topography_tex = ti.Texture(ti.Format.r8, TOPOGRAPHY_TEX_RES)
        self.topography_buff = ti.field(dtype=ti.u8, shape=TOPOGRAPHY_TEX_RES)
        load_image = ti.tools.imread(TOPOGRAPHY_TEX_FILE)[:, :, 0]
        self.topography_buff.from_numpy(load_image)

        self.ocean_tex = ti.Texture(ti.Format.r8, OCEAN_TEX_RES)
        self.ocean_buff = ti.field(dtype=ti.u8, shape=OCEAN_TEX_RES)
        load_image = ti.tools.imread(OCEAN_TEX_FILE)[:, :, 0]
        self.ocean_buff.from_numpy(load_image)

        self.clouds_tex = ti.Texture(ti.Format.r8, CLOUDS_TEX_RES)
        self.clouds_buff = ti.field(dtype=ti.u8, shape=CLOUDS_TEX_RES)
        load_image = ti.tools.imread(CLOUDS_TEX_FILE)[:, :, 0]
        self.clouds_buff.from_numpy(load_image)

        self.bathymetry_tex = ti.Texture(ti.Format.r8, BATHYMETRY_TEX_RES)
        self.bathymetry_buff = ti.field(dtype=ti.u8, shape=BATHYMETRY_TEX_RES)
        load_image = ti.tools.imread(BATHYMETRY_TEX_FILE)[:, :, 0]
        self.bathymetry_buff.from_numpy(load_image)

        self.emissive_tex = ti.Texture(ti.Format.r8, EMISSIVE_TEX_RES)
        self.emissive_buff = ti.field(dtype=ti.u8, shape=EMISSIVE_TEX_RES)
        load_image = ti.tools.imread(EMISSIVE_TEX_FILE)[:, :, 0]
        self.emissive_buff.from_numpy(load_image)

        self.stars_tex = ti.Texture(ti.Format.rgba8, STARS_TEX_RES)
        self.stars_buff = ti.Vector.field(3, dtype=ti.u8, shape=STARS_TEX_RES)
        load_image = ti.tools.imread(STARS_TEX_FILE)
        self.stars_buff.from_numpy(load_image)

        # LUTS
        self.CIE_LUT_tex = ti.Texture(ti.Format.rgba16f, CIE_LUT_RES)
        self.CIE_LUT_buff = ti.Vector.field(3, dtype=ti.f32, shape=CIE_LUT_RES)
        with open(CIE_LUT_FILE, 'rb') as file:
            load_data = np.fromfile(file, dtype=np.float32, count=CIE_LUT_RES[0]*CIE_LUT_RES[1]*3)
        data_array = np.zeros(shape=(CIE_LUT_RES[0], CIE_LUT_RES[1], 3), dtype=np.float32)
        for x in range (0, CIE_LUT_RES[0]):
            for y in range (0, CIE_LUT_RES[1]):
                data_array[x, y, 0] = load_data[(x + y*CIE_LUT_RES[0])*3]
                data_array[x, y, 1] = load_data[(x + y*CIE_LUT_RES[0])*3 + 1]
                data_array[x, y, 2] = load_data[(x + y*CIE_LUT_RES[0])*3 + 2]
        self.CIE_LUT_buff.from_numpy(data_array)

        self.srgb_to_spectrum_buff = ti.Vector.field(3, dtype=ti.f16, shape=(300))
        with open(SRGB2SPEC_LUT_FILE, 'rb') as file:
            load_data = np.fromfile(file, dtype=np.float16, count=300*3)
        data_array = np.zeros(shape=(300, 3), dtype=np.float16)
        for x in range (0, 300):
            data_array[x, 0] = load_data[x*3]
            data_array[x, 1] = load_data[x*3 + 1]
            data_array[x, 2] = load_data[x*3 + 2]
        self.srgb_to_spectrum_buff.from_numpy(data_array)

        self.O3_crossec_LUT_buff = ti.field(dtype=ti.f32, shape=(O3_CROSSEC_LUT_RES))
        with open(O3_CROSSEC_LUT_FILE, 'rb') as file:
            load_data = np.fromfile(file, dtype=np.float32, count=O3_CROSSEC_LUT_RES)
        data_array = np.zeros(shape=(O3_CROSSEC_LUT_RES), dtype=np.float32)
        for x in range (0, O3_CROSSEC_LUT_RES):
            data_array[x] = load_data[x]
        self.O3_crossec_LUT_buff.from_numpy(data_array)

        self.transmittance_lut_tex = ti.Texture(ti.Format.r32f, TRANS_LUT_RES)
        self.transmittance_lut_buff = ti.field(dtype=ti.f32, shape=TRANS_LUT_RES)
        with open(TRANS_LUT_FILE, 'rb') as file:
            load_data = np.fromfile(file, dtype=np.float32, count=TRANS_LUT_RES[0]*TRANS_LUT_RES[1]*TRANS_LUT_RES[2])
        # data_array = np.zeros(shape=(TRANS_LUT_RES[0], TRANS_LUT_RES[1], TRANS_LUT_RES[2]), dtype=np.float32)
        # for x in range (0, TRANS_LUT_RES[0]):
        #     for y in range (0, TRANS_LUT_RES[1]):
        #         for z in range (0, TRANS_LUT_RES[2]):
        #             data_array[x, y, z] = load_data[x + y * TRANS_LUT_RES[0] + z * TRANS_LUT_RES[0] * TRANS_LUT_RES[1]]
        data_array = np.fromfile(TRANS_LUT_FILE, dtype=np.float32)
        data_array = data_array.reshape(TRANS_LUT_RES)
        self.transmittance_lut_buff.from_numpy(data_array)

        # CRF
        self.crf_names = []
        data_array = self.load_crfs()
        self.crf_lut_res = (1024, len(self.crf_names))
        self.crf_tex = ti.Texture(ti.Format.rgba32f, self.crf_lut_res)
        self.crf_buff = ti.Vector.field(3, dtype=ti.f32, shape=self.crf_lut_res)
        self.crf_buff.from_numpy(data_array)
        self.set_crf_count(self.crf_lut_res[1])

    def copy_textures(self):
        self.copy_albedo_texture(self.albedo_tex)
        self.copy_topography_texture(self.topography_tex)
        self.copy_ocean_texture(self.ocean_tex)
        self.copy_clouds_texture(self.clouds_tex)
        self.copy_bathymetry_texture(self.bathymetry_tex)
        self.copy_emissive_texture(self.emissive_tex)
        self.copy_stars_texture(self.stars_tex)
        self.copy_CIE_LUT_texture(self.CIE_LUT_tex)
        self.copy_CRF_LUT_texture(self.crf_tex)
        self.copy_transmittance_LUT_texture(self.transmittance_lut_tex)

    def load_crfs(self):
        # Re-running the code with the updated directory path
        directory = os.path.join(os.getcwd(), 'LUT/camera_response_functions/')

        # Resetting the lists for file names and data
        crf_data = []

        filenames = os.listdir(directory)
        filenames.insert(0, filenames.pop(filenames.index('Neutral.rf'))) # Moving the neutral file to the front of the list
        for filename in filenames:
            if (filename.endswith(".txt") or filename.endswith(".rf")) and not "README" in filename:  # Ensuring to read only the relevant .txt files
                self.crf_names.append(filename)

                with open(os.path.join(directory, filename), 'r') as file:
                    lines = file.readlines()
                    file_data = [list(map(float, line.split()))[1:] for line in lines]  # Ignore the irradiance float
                    crf_data.append(file_data)

        # Convert the list to a numpy array with the specified shape (1024, n, 3)
        crf_array = np.array(crf_data, dtype=np.float32).transpose(1, 0, 2)
        return crf_array


    @ti.kernel
    def copy_albedo_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0)):
        for i, j in ti.ndrange(ALBEDO_TEX_RES[0], ALBEDO_TEX_RES[1]):
            val = ti.cast(self.albedo_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val.x, val.y, val.z, 0.0]))

    @ti.kernel
    def copy_topography_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r8, lod=0)):
        for i, j in ti.ndrange(TOPOGRAPHY_TEX_RES[0], TOPOGRAPHY_TEX_RES[1]):
            val = ti.cast(self.topography_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val, 0.0, 0.0, 0.0]))

    @ti.kernel
    def copy_ocean_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r8, lod=0)):
        for i, j in ti.ndrange(OCEAN_TEX_RES[0], OCEAN_TEX_RES[1]):
            val = ti.cast(self.ocean_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val, 0.0, 0.0, 0.0]))
    
    @ti.kernel
    def copy_clouds_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r8, lod=0)):
        for i, j in ti.ndrange(CLOUDS_TEX_RES[0], CLOUDS_TEX_RES[1]):
            val = ti.cast(self.clouds_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val, 0.0, 0.0, 0.0]))

    @ti.kernel
    def copy_bathymetry_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r8, lod=0)):
        for i, j in ti.ndrange(BATHYMETRY_TEX_RES[0], BATHYMETRY_TEX_RES[1]):
            val = ti.cast(self.bathymetry_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val, 0.0, 0.0, 0.0]))

    @ti.kernel
    def copy_emissive_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r8, lod=0)):
        for i, j in ti.ndrange(EMISSIVE_TEX_RES[0], EMISSIVE_TEX_RES[1]):
            val = ti.cast(self.emissive_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val, 0.0, 0.0, 0.0]))

    @ti.kernel
    def copy_stars_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0)):
        for i, j in ti.ndrange(STARS_TEX_RES[0], STARS_TEX_RES[1]):
            val = ti.cast(self.stars_buff[i, j], ti.f32) / 255.0
            tex.store(ti.Vector([i, j]), ti.Vector([val.x, val.y, val.z, 0.0]))

    @ti.kernel
    def copy_CIE_LUT_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba16f, lod=0)):
        for i, j in ti.ndrange(CIE_LUT_RES[0], CIE_LUT_RES[1]):
            val = ti.cast(self.CIE_LUT_buff[i, j], ti.f32)
            tex.store(ti.Vector([i, j]), ti.Vector([val.x, val.y, val.z, 0.0]))
    
    @ti.kernel
    def copy_CRF_LUT_texture(self, tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f, lod=0)):
        for i, j in ti.ndrange(self.crf_lut_res[0], self.crf_lut_res[1]):
            val = ti.cast(self.crf_buff[i, j], ti.f32)
            tex.store(ti.Vector([i, j]), ti.Vector([val.x, val.y, val.z, 0.0]))

    @ti.kernel
    def copy_transmittance_LUT_texture(self, tex: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0)):
        for i, j, k in ti.ndrange(TRANS_LUT_RES[0], TRANS_LUT_RES[1], TRANS_LUT_RES[2]):
            val = ti.cast(self.transmittance_lut_buff[i, j, k], ti.f32)
            tex.store(ti.Vector([i, j, k]), ti.Vector([val, 0.0, 0.0, 0.0]))

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

    @ti.kernel
    def set_aspect_scale(self, scale: ti.f32):
        self.aspect_scale[None] = scale

    @ti.kernel
    def set_exposure(self, exposure: ti.f32):
        self.exposure[None] = exposure

    @ti.kernel
    def set_gamma(self, gam: ti.f32):
        self.gamma[None] = gam

    @ti.kernel
    def set_crf(self, index: ti.i32):
        self.selected_crf[None] = index

    @ti.kernel
    def set_crf_count(self, num: ti.i32):
        self.crf_count[None] = num

    @ti.kernel
    def set_sun_angle(self, ang: ti.f32):
        self.sun_angle[None] = ang
    
    @ti.kernel
    def set_sun_path_rot(self, ang: ti.f32):
        self.sun_path_rot[None] = ang


    @ti.func
    def get_cast_dir(self, u, v):
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)*self.aspect_scale[None]
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    

    @ti.kernel
    def render(self, albedo_sampler: ti.types.texture(num_dimensions=2),
                     height_sampler: ti.types.texture(num_dimensions=2),
                     ocean_sampler: ti.types.texture(num_dimensions=2),
                     clouds_sampler: ti.types.texture(num_dimensions=2),
                     bathymetry_sampler: ti.types.texture(num_dimensions=2),
                     emissive_sampler: ti.types.texture(num_dimensions=2),
                     stars_sampler: ti.types.texture(num_dimensions=2),
                     cie_lut_sampler: ti.types.texture(num_dimensions=2),
                     trans_lut_sampler: ti.types.texture(num_dimensions=3)):

        scene_params = SceneParameters()
        scene_params.land_height_scale = self.land_height_scale

        # Sun parameters
        sun_radius   = 6.95e8
        sun_distance = 1.4959e11
        scene_params.sun_angular_radius = sun_radius / sun_distance
        scene_params.sun_cos_angle      = ti.cos(scene_params.sun_angular_radius)
        sun_rot = vec2( -sin(self.sun_path_rot[None]), cos(self.sun_path_rot[None]))
        scene_params.light_direction = vec3(-sin(self.sun_angle[None]), cos(self.sun_angle[None]) * sun_rot)

        ti.loop_config(block_dim=256)
        for u, v in self.color_buffer:
            
            
            
            # Sample a path from sensor
            wavelength, response, wavelength_rcp_pdf = spectrum_sample(cie_lut_sampler, CIE_LUT_RES[0])
            path_params = PathParameters()
            path_params.wavelength = wavelength
            path_params.ray_dir = self.get_cast_dir(u, v)
            path_params.ray_pos = self.camera_pos[None]

            # Sample incoming radiance for path
            sample = pt.path_tracer(path_params, scene_params, 
                                    albedo_sampler, 
                                    height_sampler, 
                                    ocean_sampler, 
                                    clouds_sampler, 
                                    bathymetry_sampler,
                                    emissive_sampler,
                                    stars_sampler,
                                    self.srgb_to_spectrum_buff,
                                    self.O3_crossec_LUT_buff)
            # Convert spectrum sample to sRGB and accumulate
            xyz = sample * response * wavelength_rcp_pdf
            self.color_buffer[u, v] += xyzToRGBMatrix_D65 @ xyz 


    @ti.func
    def camera_response(self, crf_sampler: ti.template(), tristimulus: vec3):

        tristimulus = clamp(tristimulus, 0.0, 1.0)

        slice_v = (ti.cast(self.selected_crf[None], ti.f32) + 0.5) / ti.cast(self.crf_count[None], ti.f32)
        u_offset = 0.5 / 1024.0
        u_lookup = min(tristimulus + u_offset, 1.0 - u_offset)
        red = crf_sampler.sample_lod(ti.Vector([u_lookup.r, slice_v]), 0.0).r
        green = crf_sampler.sample_lod(ti.Vector([u_lookup.g, slice_v]), 0.0).g
        blue = crf_sampler.sample_lod(ti.Vector([u_lookup.b, slice_v]), 0.0).b
        return clamp( vec3(red, green, blue), 0.0, 1.0)

    @ti.kernel
    def _render_to_image(self, samples: ti.i32, crf_sampler: ti.types.texture(num_dimensions=2)):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)
            linear = self.color_buffer[i, j]/samples * darken * ti.pow(2.0, self.exposure[None])
            # output = srgb_transfer(agx.display_transform(linear))
            tonemapped = openDR_transform(linear.r, linear.g, linear.b)
            camera = self.camera_response(crf_sampler, tonemapped)

            gamma = pow(camera, self.gamma[None])

            output = srgb_transfer(gamma)


            self._rendered_image[i, j] = output

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self):
        self.render(self.albedo_tex, 
                    self.topography_tex, 
                    self.ocean_tex, 
                    self.clouds_tex, 
                    self.bathymetry_tex, 
                    self.emissive_tex, 
                    self.stars_tex,
                    self.CIE_LUT_tex,
                    self.transmittance_lut_tex)
        self.current_spp += 1

    def fetch_image(self):
        self._render_to_image(self.current_spp, self.crf_tex)
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c):
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r
