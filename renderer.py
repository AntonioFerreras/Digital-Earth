import taichi as ti
from atmos import *
import numpy as np
from math_utils import *

MAX_RAY_DEPTH = 4
USE_LOW_QUAL_TEXTURES = False
TEX_RES_4K = (3840, 1920)
TEX_RES_10K = (10800, 5400)
ALBEDO_4K = 'textures/earth_color_4K.png'
ALBEDO_10K = 'textures/earth_color_10K.png'
TOPOGRAPHY_4K = 'textures/topography_4k.png'
TOPOGRAPHY_10K = 'textures/topography_10k.png'

ALBEDO_TEX_FILE = ALBEDO_4K if USE_LOW_QUAL_TEXTURES else ALBEDO_10K
ALBEDO_TEX_RES = TEX_RES_4K if USE_LOW_QUAL_TEXTURES else TEX_RES_10K
TOPOGRAPHY_TEX_FILE = TOPOGRAPHY_4K if USE_LOW_QUAL_TEXTURES else TOPOGRAPHY_10K
TOPOGRAPHY_TEX_RES = ALBEDO_TEX_RES

@ti.data_oriented
class Renderer:
    def __init__(self, image_res, up, exposure=3):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.23)

        self.atmos = Atmos()

        self.land_height_scale = 20000.0

        # Load Textures
        self.albedo_tex = ti.Texture(ti.Format.rgba8, ALBEDO_TEX_RES)
        self.albedo_buff = ti.Vector.field(3, dtype=ti.u8, shape=ALBEDO_TEX_RES)
        load_image = ti.tools.imread(ALBEDO_TEX_FILE)
        self.albedo_buff.from_numpy(load_image)

        self.topography_tex = ti.Texture(ti.Format.r8, TOPOGRAPHY_TEX_RES)
        self.topography_buff = ti.field(dtype=ti.u8, shape=TOPOGRAPHY_TEX_RES)
        load_image = ti.tools.imread(TOPOGRAPHY_TEX_FILE)[:, :, 0]
        self.topography_buff.from_numpy(load_image)

    def copy_textures(self):
        self.copy_albedo_texture(self.albedo_tex)
        self.copy_topography_texture(self.topography_tex)

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

    @ti.func
    def get_cast_dir(self, u, v):
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    @ti.func
    def land_sdf(self, heightmap: ti.template(), pos):
        # bump-mapped sphere SDF for a sphere centered at origin
        return length(pos) - self.atmos.planet_r - \
                              self.land_height_scale*sample_sphere_texture(heightmap, pos).x
    
    @ti.func
    def land_normal(self, heightmap: ti.template(), pos):
        d = self.land_sdf(heightmap, pos)

        e = ti.Vector([0.5*2.0*np.pi*self.atmos.planet_r/TOPOGRAPHY_TEX_RES[0], 0.0])

        n = d - vec3(self.land_sdf(heightmap, pos - e.xyy),
                     self.land_sdf(heightmap, pos - e.yxy),
                     self.land_sdf(heightmap, pos - e.yyx))
        return n.normalized()
    
    @ti.func
    def intersect_land(self, heightmap: ti.template(), pos, dir):
        ray_dist = 0.
        max_ray_dist = self.atmos.planet_r*10.0
    
        for i in range(0, 150):
            ro = pos + dir * ray_dist

            dist = self.land_sdf(heightmap, ro)
            ray_dist += dist
            
            if ray_dist > max_ray_dist or abs(dist) < ray_dist*0.0001:
                break
        
        return ray_dist if ray_dist < max_ray_dist else -1.0

    @ti.func
    def shadow_intersect_land(self, heightmap: ti.template(), pos, dir):
        ray_dist = 0.
        max_ray_dist = self.atmos.planet_r*10.0
        visibility = 1.0
        for i in range(0, 150):
            ro = pos + dir * ray_dist

            dist = self.land_sdf(heightmap, ro)
            ray_dist += dist
            if abs(dist) < ray_dist*0.0001:
                visibility = 0.0
                break
            if ray_dist > max_ray_dist:
                break
        
        return visibility
    
    @ti.kernel
    def render(self, albedo_sampler: ti.types.texture(num_dimensions=2),
                     height_sampler: ti.types.texture(num_dimensions=2)):
        self.light_direction[None] = vec3(-0.5, 0.5, -0.5).normalized()

        ti.loop_config(block_dim=256)
        for u, v in self.color_buffer:
            d = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            earth_intersection = self.intersect_land(height_sampler, pos, d)

            

            
            if earth_intersection > 0.0:
                land_pos = pos + d*earth_intersection
                sphere_normal = land_pos.normalized()
                land_normal = self.land_normal(height_sampler, land_pos)
                albedo = sample_sphere_texture(albedo_sampler, land_pos).rgb

                shadow_pos = land_pos + land_normal*self.atmos.planet_r*0.0001
                # visibility = self.shadow_intersect_land(height_sampler, shadow_pos, self.light_direction[None])

                # shadow_sphere_intersection = rsi(shadow_pos, self.light_direction[None], self.atmos.planet_r).x
                # shadow = shadow_sphere_intersection < 0.0

                earth_land_shadow = saturate(4.0*land_pos.normalized().dot(self.light_direction[None]))

                contrib += albedo * earth_land_shadow * saturate(land_normal.dot(self.light_direction[None]))
            else:
                contrib += ti.Vector([0.0, 0.0, 0.0])
            self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = ti.sqrt(
                    self.color_buffer[i, j][c] * darken * self.exposure /
                    samples)

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self):
        self.render(self.albedo_tex, self.topography_tex)
        self.current_spp += 1

    def fetch_image(self):
        self._render_to_image(self.current_spp)
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
