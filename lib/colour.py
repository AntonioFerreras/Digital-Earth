import taichi as ti
from taichi.math import *
from lib.math_utils import *

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

# Blackbody SPD
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
def srgb_to_spectrum_triplet(lut: ti.template(), rgb, wavelengths):
    w = ti.cast(wavelengths - 400, ti.i32)
    f = w - (wavelengths - 400.0)
    power = vec3(0.0)
    for i in range(3):
        if w[i] > 0 and w[i] < 299:
            coeff = mix(lut[w[i]], lut[w[i]+1], f)
            power[i] = rgb.dot(coeff)
    
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

@ti.func
def lum(x):
    return x.dot(vec3(0.2126729,  0.7151522,  0.0721750))

@ti.func
def lum3(x):
    y = lum(x)
    return vec3(y, y, y)