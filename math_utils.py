import math
import taichi as ti
import numpy as np

eps = 1e-4
inf = 1e10

@ti.func
def cone_angle_to_solid_angle(x):
    return np.pi*2*(1.0 - ti.cos(x))

@ti.func
def out_dir(n):
    u = ti.Vector([1.0, 0.0, 0.0])
    if ti.abs(n[1]) < 1 - 1e-3:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2 * math.pi * ti.random(ti.f32)
    r = ti.random(ti.f32)
    ay = ti.sqrt(r)
    ax = ti.sqrt(1 - r)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n

@ti.func
def rsi(pos, dir, r):
    b     = pos.dot(dir)
    discr   = b*b - pos.dot(pos) + r*r
    discr     = ti.sqrt(discr)

    return ti.select(discr < 0.0, ti.Vector([-1.0, -1.0]), ti.Vector([-b, -b]) + ti.Vector([-discr, discr]))

@ti.func
def sphere_UV_map(n):
    return ti.Vector([ (ti.atan2(n.z, -n.x) / np.pi + 1.) / 2.,
                       (ti.asin(n.y) / np.pi + 0.5) ])

@ti.func
def UV_to_index(uv, res):
    return ti.cast(uv * res, ti.i32)

@ti.func
def UV_to_index_stochastic(uv, res):
    return ti.cast(uv * res + ti.random()*2.0 - 1.0, ti.i32)

@ti.func
def sample_sphere_texture(sampler: ti.template(), pos):
    uv = sphere_UV_map(pos.normalized())
        # coord = UV_to_index_stochastic(uv, 
        #                                ti.Vector([self.albedo_buff.shape[0], 
        #                                           self.albedo_buff.shape[1]]))
    return sampler.sample_lod(uv, 0.0)# self.albedo_buff[coord.x, coord.y].xyz/255.0

@ti.func
def saturate(x):
    return ti.math.clamp(x, 0.0, 1.0)

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
