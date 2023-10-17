import taichi as ti
from atmos import *
import numpy as np
from math_utils import *

xyzToRGBMatrix_D65 = mat3([
	 3.2409699419, -1.5373831776, -0.4986107603,
	-0.9692436363,  1.8759675015,  0.0415550574,
	 0.0556300797, -0.2039769589,  1.0569715142
])

@ti.func
def spectrum_sample(cie_lut_sampler: ti.template()):
    sample = ti.random()
    wavelength = 390.0 + 441.0 * sample
    response = cie_lut_sampler.sample_lod(ti.Vector([sample, 0.75]), 0.0).xyz
    pdf = 1.0 / 441.0

    return wavelength, response, pdf

@ti.func
def srgb_to_spectrum(lut: ti.template(), rgb, wavelength):
    w = ti.cast(wavelength - 400, ti.i32)
    power = 0.
    if w > 0 and w < 300:
        power = rgb.dot(lut[w])
    
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