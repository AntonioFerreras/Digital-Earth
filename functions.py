import taichi as ti
from atmos import *
import numpy as np
from math_utils import *

## COLOR
xyzToRGBMatrix_D65 = mat3([
	 3.2409699419, -1.5373831776, -0.4986107603,
	-0.9692436363,  1.8759675015,  0.0415550574,
	 0.0556300797, -0.2039769589,  1.0569715142
])

@ti.func
def spectrum_sample(cie_lut_sampler: ti.template(), res):

    # sample = ti.random()
    # wavelength = 390.0 + 441.0 * sample
    # response = cie_lut_sampler.sample_lod(ti.Vector([sample, 0.75]), 0.0).xyz
    # return wavelength, response, 1.0

    # # Binary search of CIE LUT on that primary
    sample = ti.random()
    lo = 0.0
    hi = 1.0
    mid = (lo + hi)/2.0

    for x in range (0, log2(res)):

        val = saturate(vec3(1./3., 1./3., 1./3.).dot(cie_lut_sampler.sample_lod(ti.Vector([mid, 0.25]), 0.0).xyz))

        if val < sample:
            lo = mid
        elif val > sample:
            hi = mid
        else:
            break

        mid = (lo + hi)/2.0

    wavelength = 390.0 + 441.0 * mid
    response = cie_lut_sampler.sample_lod(ti.Vector([mid, 0.75]), 0.0).xyz
    RGB_cmf_max = cie_lut_sampler.sample_lod(ti.Vector([1.0, 0.25]), 0.0).xyz
    pdf = response.dot(RGB_cmf_max)

    rcp_pdf = 0.0
    if pdf > 1e-3 and not (isinf(pdf) or isnan(pdf)):
        rcp_pdf = 1.0 / pdf

    return wavelength, response, rcp_pdf

@ti.func
def plancks(temperature, wavelength):
    h = 6.62607015e-16
    c = 2.9e17
    k = 1.38e-5

    p1 = 2.0 * h * pow(c, 2.0) / pow(wavelength, 5.0)
    p2 = exp((h * c) / (wavelength * k * temperature)) - 1.0

    return p1 / p2

@ti.func
def srgb_to_spectrum(lut: ti.template(), rgb, wavelength):
    w = ti.cast(wavelength - 400, ti.i32)
    f = w - (wavelength - 400.0)
    power = 0.
    if w > 0 and w < 299:
        coeff = mix(lut[w], lut[w+1], f)
        power = rgb.dot(coeff)
    
    return power
        

@ti.func
def srgb_transfer(linear):
    SRGBLo = linear * 12.92
    SRGBHi = (pow(abs(linear), vec3(1.0/2.4, 1.0/2.4, 1.0/2.4)) * 1.055) - 0.055
    SRGB = mix(SRGBHi, SRGBLo, step(linear, vec3(0.0031308, 0.0031308, 0.0031308)))
    return SRGB

@ti.func
def srgb_transfer_inverse(color):
    linearRGBLo = color / 12.92
    linearRGBHi = pow((color + 0.055) / 1.055, vec3(2.4, 2.4, 2.4))
    linearRGB = mix(linearRGBHi, linearRGBLo, step(color, vec3(0.04045, 0.04045, 0.04045)))
    return linearRGB

## DIRECTION SAMPLING
@ti.func
def sample_cosine_weighted_hemisphere(n):
    # Shirley, et al, 2019. Sampling Transformation Zoo. Chapter 16, Ray Tracing Gems, p240
    u = ti.Vector([ti.random(), ti.random()])
    a = 1.0 - 2.0 * u[0]
    b = ti.sqrt(1.0 - a * a)
    a *= 1.0 - 1e-5
    b *= 1.0 - 1e-5 # Grazing angle precision fix
    phi = 2.0 * np.pi * u[1]
    return ti.Vector([n.x + b * ti.cos(phi), n.y + b * ti.sin(phi), n.z + a]).normalized()

@ti.func
def make_orthonormal_basis(n):
    h = ti.select(ti.abs(n.y) > 0.9, ti.math.vec3(1.0, 0.0, 0.0), ti.math.vec3(0.0, 1.0, 0.0))
    y = n.cross(h).normalized()
    x = n.cross(y)
    return x, y

@ti.func
def make_tangent_space(n):
    x, y = make_orthonormal_basis(n)
    return ti.math.mat3(x, y, n).transpose()

@ti.func
def sample_cone(cos_theta_max):
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
def sample_cone_oriented(cos_theta_max, n):
    mat_dir = make_tangent_space(n) @ sample_cone(cos_theta_max)
    return ti.Vector([mat_dir[0], mat_dir[1], mat_dir[2]])