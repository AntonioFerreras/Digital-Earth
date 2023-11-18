import taichi as ti
import numpy as np
from taichi.math import *
from lib.math_utils import *



## DIRECTION SAMPLING




@ti.func
def sample_cone(cos_theta_max: ti.f32):
    u0 = ti.random()
    u1 = ti.random()
    cos_theta = (1.0 - u0) + u0 * cos_theta_max
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * np.pi * u1
    x = sin_theta * ti.cos(phi)
    y = sin_theta * ti.sin(phi)
    z = cos_theta
    return ti.Vector([x, y, z])

@ti.func
def sample_cone_oriented(cos_theta_max: ti.f32, n: vec3):
    mat_dir = make_tangent_space(n) @ sample_cone(cos_theta_max)
    return ti.Vector([mat_dir[0], mat_dir[1], mat_dir[2]])

@ti.func
def sample_hemisphere_cosine_weighted(n: vec3):
    # Shirley, et al, 2019. Sampling Transformation Zoo. Chapter 16, Ray Tracing Gems, p240
    u = ti.Vector([ti.random(), ti.random()])
    a = 1.0 - 2.0 * u[0]
    b = ti.sqrt(1.0 - a * a)
    a *= 1.0 - 1e-5
    b *= 1.0 - 1e-5 # Grazing angle precision fix
    phi = 2.0 * np.pi * u[1]
    return ti.Vector([n.x + b * ti.cos(phi), n.y + b * ti.sin(phi), n.z + a]).normalized()

@ti.func
def sample_sphere(rand: vec2):
    rand.x *= np.pi * 2.0; rand.y = rand.y * 2.0 - 1.0
    ground = vec2(sin(rand.x), cos(rand.x)) * sqrt(1.0 - rand.y * rand.y)
    return vec3(ground.x, ground.y, rand.y).normalized()