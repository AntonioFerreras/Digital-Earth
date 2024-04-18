import math
import taichi as ti
from taichi.math import *
import numpy as np

eps = 1e-4
inf = 1e10

@ti.func
def sqr(x):
    return x*x

@ti.func
def cone_angle_to_solid_angle(x: ti.f32):
    return np.pi*2*(1.0 - ti.cos(x))

@ti.func
def rsi(pos: vec3, dir: vec3, r: ti.f32):
    b     = pos.dot(dir)
    discr   = b*b - pos.dot(pos) + r*r
    discr     = ti.sqrt(discr)

    return ti.select(discr < 0.0, ti.Vector([-1.0, -1.0]), ti.Vector([-b, -b]) + ti.Vector([-discr, discr]))

@ti.func
def sphere_UV_map(n: vec3):
    return ti.Vector([ (ti.atan2(n.z, -n.x) / np.pi + 1.) / 2.,
                       (ti.asin(n.y) / np.pi + 0.5) ])

@ti.func
def UV_to_index(uv, res):
    return ti.cast(uv * res, ti.i32)

@ti.func
def UV_to_index_stochastic(uv, res):
    return ti.cast(uv * res + ti.random()*2.0 - 1.0, ti.i32)

@ti.func
def sample_sphere_texture(sampler: ti.template(), pos, scale=1.0):
    uv = sphere_UV_map(pos.normalized())
        # coord = UV_to_index_stochastic(uv, 
        #                                ti.Vector([self.albedo_buff.shape[0], 
        #                                           self.albedo_buff.shape[1]]))
    return sampler.sample_lod(fract(uv*scale), 0.0)# self.albedo_buff[coord.x, coord.y].xyz/255.0

@ti.func
def saturate(x):
    return ti.math.clamp(x, 0.0, 1.0)


@ti.func
def normal_distribution(x: ti.f32, mean: ti.f32, stdev: ti.f32):
    return (1.0 / (stdev * sqrt(2.0 * np.pi))) * exp(-0.5 * sqr((x - mean) / stdev))

@ti.func
def make_orthonormal_basis(n: vec3):
    h = ti.select(ti.abs(n.y) > 0.9, ti.math.vec3(1.0, 0.0, 0.0), ti.math.vec3(0.0, 1.0, 0.0))
    y = n.cross(h).normalized()
    x = n.cross(y)
    return x, y

@ti.func
def make_tangent_space(n: vec3):
    x, y = make_orthonormal_basis(n)
    return ti.math.mat3(x, y, n).transpose()

@ti.func
def spherical_direction(sin_theta: ti.f32, cos_theta: ti.f32, phi: ti.f32, x: vec3, y: vec3, z: vec3):
    return sin_theta * ti.cos(phi) * x + sin_theta * ti.sin(phi) * y + cos_theta * z

@ti.func
def hash12(p: vec2):
    p3  = fract(vec3(p.xyx) * .1031)
    p3 += dot(p3, p3.yzx + 19.19)
    return fract((p3.x + p3.y) * p3.z)

@ti.func
def hash22(p: vec2):
    p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973))
    p3 += dot(p3, p3.yzx+19.19)
    return fract((p3.xx+p3.yz)*p3.zy)

def np_normalize(v):
    # https://stackoverflow.com/a/51512965/12003165
    return v / np.sqrt(np.sum(v**2))


def np_rotate_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # https://stackoverflow.com/a/6802723/12003165
    axis = np_normalize(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])
