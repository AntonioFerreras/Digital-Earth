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
import lib.volume_rendering_models as volume
import lib.bruneton_mappings as bruneton
import torch
from model import MLP



@ti.data_oriented
class Renderer:
    def __init__(self, image_res, up):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0
        
        # Flag to switch between path tracing and MLP inference
        self.use_mlp = False

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.aspect_scale = ti.field(dtype=ti.f32, shape=())

        self.exposure = ti.field(dtype=ti.f32, shape=())
        self.selected_crf = ti.field(dtype=ti.i32, shape=())
        self.crf_count = ti.field(dtype=ti.i32, shape=())
        self.gamma = ti.field(dtype=ti.f32, shape=())

        # Buffers for MLP inference
        self.uvwz_buffer = ti.Vector.field(4, dtype=ti.f32, shape=image_res)
        self.rgb_buffer = ti.Vector.field(3, dtype=ti.f32, shape=image_res)

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

        # CRF
        self.crf_names = []
        data_array = self.load_crfs()
        self.crf_lut_res = (1024, len(self.crf_names))
        self.crf_tex = ti.Texture(ti.Format.rgba32f, self.crf_lut_res)
        self.crf_buff = ti.Vector.field(3, dtype=ti.f32, shape=self.crf_lut_res)
        self.crf_buff.from_numpy(data_array)
        self.set_crf_count(self.crf_lut_res[1])

        # Load the MLP model
        self.load_mlp_model()

    def copy_textures(self):
        self.copy_CIE_LUT_texture(self.CIE_LUT_tex)
        self.copy_CRF_LUT_texture(self.crf_tex)

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
    def render(self, cie_lut_sampler: ti.types.texture(num_dimensions=2)):

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
            
            spp = 32.0
            for i in range(spp):
                # Sample a path from sensor
                wavelength, response, wavelength_rcp_pdf = spectrum_sample(cie_lut_sampler, CIE_LUT_RES[0])
                path_params = PathParameters()
                path_params.wavelength = wavelength
                path_params.ray_dir = self.get_cast_dir(u, v)
                path_params.ray_pos = self.camera_pos[None]

                # Sample incoming radiance for path
                sample = pt.path_tracer(path_params, scene_params, 
                                        self.srgb_to_spectrum_buff,
                                        self.O3_crossec_LUT_buff)

                # Convert spectrum sample to sRGB and accumulate
                xyz = sample * response * wavelength_rcp_pdf
                self.color_buffer[u, v] += (xyzToRGBMatrix_D65 @ xyz )/spp

    

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
        """Accumulate rendered image, using either path tracer or MLP inference"""
        if self.use_mlp and self.mlp_loaded:
            # Use MLP inference for rendering
            self.render_with_mlp(self.CIE_LUT_tex)
            
            # Call the MLP inference in Python scope
            self.mlp_inference_batch()
        else:
            # Use path tracing (original method)
            self.render(self.CIE_LUT_tex)
        
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

    @ti.kernel
    def batch_path_trace(self, 
                    uvwz: ti.types.vector(4, ti.f32),
                    cie_lut_sampler: ti.types.texture(num_dimensions=2)) -> ti.types.vector(3, ti.f32):
        
        scene_params = SceneParameters()
        scene_params.land_height_scale = self.land_height_scale

        # Sun parameters
        sun_radius   = 6.95e8
        sun_distance = 1.4959e11
        scene_params.sun_angular_radius = sun_radius / sun_distance
        scene_params.sun_cos_angle      = ti.cos(scene_params.sun_angular_radius)
        
        # Convert uvwz to position and directions
        # Use a fixed phi angle for consistency
        phi = 0.0
        position, view_dir, sun_dir = bruneton.UvwzToRayParams(uvwz)
        
        # Set the sun direction for the scene
        scene_params.light_direction = sun_dir
        
        # Initialize accumulator for the final result
        final_result = ti.Vector([0.0, 0.0, 0.0])
        
        # Total number of samples (batch_size * samples_per_batch)
        total_samples = 512 * 32 
        
        # Use a single loop for better parallelization
        ti.loop_config(block_dim=256)
        for i in range(total_samples):
            # Sample a wavelength for spectral rendering
            wavelength, response, wavelength_rcp_pdf = spectrum_sample(cie_lut_sampler, CIE_LUT_RES[0])
            
            # Setup path parameters
            path_params = PathParameters()
            path_params.wavelength = wavelength
            path_params.ray_dir = view_dir
            path_params.ray_pos = position
            
            # Sample incoming radiance for path
            sample = pt.path_tracer(path_params, scene_params, 
                                    self.srgb_to_spectrum_buff,
                                    self.O3_crossec_LUT_buff)
            
            # Convert spectrum sample to sRGB and accumulate
            xyz = sample * response * wavelength_rcp_pdf
            rgb = xyzToRGBMatrix_D65 @ xyz
            # Atomic add to the final result
            ti.atomic_add(final_result[0], rgb[0] / total_samples)
            ti.atomic_add(final_result[1], rgb[1] / total_samples)
            ti.atomic_add(final_result[2], rgb[2] / total_samples)
        
        return final_result

    def load_mlp_model(self):
        """Load the MLP model from the saved state dict file"""
        try:
            # Create the model
            self.mlp_model = MLP()
            
            # Load the checkpoint (which contains model_state_dict, not just the weights)
            checkpoint = torch.load("mlp_best.pth")
            
            # Extract just the model weights from the checkpoint
            if 'model_state_dict' in checkpoint:
                # This is a full checkpoint with optimizer state etc.
                state_dict = checkpoint['model_state_dict']
            else:
                # This is just a model state dict
                state_dict = checkpoint
            
            # Check if keys have "_orig_mod." prefix and fix them
            if all("_orig_mod." in key for key in state_dict.keys()):
                print("Detected '_orig_mod.' prefix in state dict keys, removing prefix...")
                # Create a new state dict with corrected keys
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace("_orig_mod.", "")
                    fixed_state_dict[new_key] = value
                state_dict = fixed_state_dict
                
            self.mlp_model.load_state_dict(state_dict)
            self.mlp_model.eval()  # Set to evaluation mode
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mlp_model = self.mlp_model.to(self.device)
            
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.mlp_model = torch.compile(self.mlp_model)
                    print("Using torch.compile for improved performance")
                except Exception as e:
                    print(f"Could not use torch.compile: {e}")
            
            print(f"MLP model loaded successfully on {self.device}")
            self.mlp_loaded = True
        except Exception as e:
            print(f"Error loading MLP model: {e}")
            self.mlp_loaded = False

    @ti.kernel
    def render_with_mlp(self, cie_lut_sampler: ti.types.texture(num_dimensions=2)):
        """Render using MLP inference instead of path tracing"""
        
        scene_params = SceneParameters()
        scene_params.land_height_scale = self.land_height_scale

        # Sun parameters
        sun_radius   = 6.95e8
        sun_distance = 1.4959e11
        scene_params.sun_angular_radius = sun_radius / sun_distance
        scene_params.sun_cos_angle      = ti.cos(scene_params.sun_angular_radius)
        sun_rot = vec2(-sin(self.sun_path_rot[None]), cos(self.sun_path_rot[None]))
        scene_params.light_direction = vec3(-sin(self.sun_angle[None]), cos(self.sun_angle[None]) * sun_rot)

        # Flag to track whether camera is outside atmosphere
        camera_outside_atmosphere = length(self.camera_pos[None]) > volume.atmos_upper_limit

        # Calculate uvwz values for each pixel
        for u, v in self.uvwz_buffer:
            # Calculate primary ray position and direction (same as path tracer)
            ray_dir = self.get_cast_dir(u, v)
            ray_pos = self.camera_pos[None]
            light_dir = scene_params.light_direction
            
            if camera_outside_atmosphere:
                # Check for intersection with atmosphere using ray-sphere intersection
                # rsi returns (t_min, t_max) where t_min is distance to entry point, t_max is distance to exit point
                hit_info = rsi(ray_pos, ray_dir, volume.atmos_upper_limit)
                
                if hit_info.x > 0:  # Ray intersects the atmosphere
                    # Move ray origin to the intersection point with the top of the atmosphere
                    ray_pos = ray_pos + hit_info.x * ray_dir
                else:
                    # Ray doesn't hit atmosphere, set a flag in uvwz to indicate black output
                    # We use a special value (negative w component) to signal this
                    self.uvwz_buffer[u, v] = vec4(0.0, 0.0, 0.0, -1.0)
                    continue
            
            # Convert ray parameters to uvwz values
            r, mu, mu_s, nu = bruneton.RayParamsToBruneton(ray_pos, ray_dir, light_dir)
            ray_pos, ray_dir, light_dir = bruneton.BrunetonToRayParams(r, mu, mu_s, nu)
            uvwz = bruneton.RayParamsToUvwz(ray_pos, ray_dir, light_dir)
            
            # Store uvwz values for batch processing
            self.uvwz_buffer[u, v] = uvwz

    @ti.kernel
    def update_color_buffer_from_rgb(self):
        """Update the color buffer with RGB values from the rgb_buffer"""
        for u, v in self.color_buffer:
            self.color_buffer[u, v] += self.rgb_buffer[u, v]

    def mlp_inference_batch(self):
        """Perform batch inference with MLP model in Python scope"""
        if not self.mlp_loaded:
            print("MLP model not loaded, skipping inference")
            return
        
        # Convert Taichi field to NumPy array
        uvwz_np = self.uvwz_buffer.to_numpy()
        
        # Create RGB array for results
        rgb_np = np.zeros((self.image_res[0], self.image_res[1], 3), dtype=np.float32)
        
        # Create a mask for points that hit the atmosphere (w >= 0)
        valid_mask = uvwz_np[..., 3] >= 0
        
        if np.any(valid_mask):  # Only run inference if there are valid points
            # Extract valid uvwz points
            valid_indices = np.where(valid_mask)
            valid_uvwz = uvwz_np[valid_indices]
            
            # Convert to PyTorch tensor
            uvwz_tensor = torch.tensor(valid_uvwz, dtype=torch.float32, device=self.device)
            
            # Process in smaller batches to avoid CUDA out of memory issues
            batch_size = 4096*4  # Adjust based on available memory
            rgb_flat = torch.zeros((valid_uvwz.shape[0], 3), dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                for i in range(0, valid_uvwz.shape[0], batch_size):
                    end_idx = min(i + batch_size, valid_uvwz.shape[0])
                    batch = uvwz_tensor[i:end_idx]
                    rgb_flat[i:end_idx] = self.mlp_model(batch)
            
            # Convert back to NumPy
            valid_rgb = rgb_flat.cpu().numpy()
            
            # Put the results back into the full array
            rgb_np[valid_indices] = valid_rgb
            
        # Update the RGB buffer (includes zeros for points that didn't hit atmosphere)
        self.rgb_buffer.from_numpy(rgb_np)
        
        # Update the color buffer using the kernel
        self.update_color_buffer_from_rgb()

    def toggle_mlp(self, enabled=None):
        """Toggle between MLP inference and path tracing"""
        if enabled is not None:
            self.use_mlp = enabled
        else:
            self.use_mlp = not self.use_mlp
        
        # Reset framebuffer when switching rendering methods
        self.reset_framebuffer()
        
        if self.use_mlp:
            method = "MLP inference" if self.mlp_loaded else "MLP (model not loaded, falling back to path tracing)"
        else:
            method = "path tracing"
            
        print(f"Rendering method: {method}")
        return self.use_mlp
