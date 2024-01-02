import taichi as ti
from taichi.math import *

@ti.dataclass
class PathParameters:
    wavelengths: vec3
    ray_dir: vec3
    ray_pos: vec3

@ti.dataclass
class SceneParameters:
    light_direction: vec3
    sun_cos_angle: float
    sun_angular_radius: float
    land_height_scale: float
