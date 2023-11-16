import math
import taichi as ti
from taichi.math import *
import numpy as np
from lib.math_utils import *

# Atmos Constants
air_num_density       = 2.5035422e25
ozone_peak = 8e-6
ozone_num_density     = air_num_density * 0.012588 * ozone_peak
ozone_cross_sec      = vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22) * 0.0001
ozone_peak_height = 25000.0 # peak density at 25km



scale_height_rayl = 8500.0
scale_height_mie  = 1200.0

scale_heights = ti.Vector([scale_height_rayl, scale_height_mie])

mie_g = 0.75

rayleigh_albedo = 1.0
aerosol_albedo = 0.98
ozone_albedo = 0.0
cloud_albedo = 0.99



planet_r = 6371e3
atmos_height  = 110e3
atmos_upper_limit = planet_r + atmos_height

# Cloud constants
clouds_extinct = 0.1
clouds_density = 1.0
clouds_height = 3000.0
clouds_thickness = 200.0
clouds_lower_limit = planet_r + clouds_height
clouds_upper_limit = clouds_lower_limit + clouds_thickness
#############


@ti.func
def rayleigh_phase(cos_theta: ti.f32):
    return 3.0/(16.0*np.pi)*(1.0 + cos_theta*cos_theta)

@ti.func
def hg_phase(cos_theta: ti.f32, g: ti.f32):
    # Henyey-Greenstein phase
    return (1-g*g)/(4.0*np.pi*pow(1.0 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def mie_phase(cos_theta: ti.f32):
    # Henyey-Greenstein phase
    return hg_phase(cos_theta, mie_g)

@ti.func
def air(wavelength: ti.f32):
    rcp_wavelength_sqr = 1.0 / (wavelength*wavelength)
    return 1.0+8.06051e-5+2.480990e-2/(132.274-rcp_wavelength_sqr)+1.74557e-4/(39.32957-rcp_wavelength_sqr)

@ti.func
def spectra_extinction_mie(wavelength: ti.f32):
    junge = 4.0
    turbidity = 1.0

    c = (0.6544 * turbidity - 0.6510) * 4e-18;
    K = (0.773335 - 0.00386891 * wavelength) / (1.0 - 0.00546759 * wavelength)
    return 0.434 * c * np.pi * pow(2.0*np.pi / (wavelength * 1e-9), junge - 2.0) * K


@ti.func
def spectra_extinction_rayleigh(wavelength: ti.f32):
    nanometers = wavelength * 1e-9

    F_N2 = 1.034 + 3.17e-4 * (1.0 / pow(wavelength, 2.0))
    F_O2 = 1.096 + 1.385e-3 * (1.0 / pow(wavelength, 2.0)) + 1.448e-4 * (1.0 / pow(wavelength, 4.0))
    CCO2 = 0.045
    king_factor = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 + CCO2 * 1.15) / (78.084 + 20.946 + 0.934 + CCO2)
    n = sqr(air(wavelength * 1e-3)) - 1.0

    return ((8.0 * pow(np.pi, 3.0) * pow(n, 2.0)) / (3.0 * air_num_density * pow(nanometers, 4.0))) * king_factor

@ti.func
def spectra_extinction_ozone(wavelength: ti.f32):
    # preetham fit by Jessie
    wavelength = wavelength - 390.0
    p1 = normal_distribution(wavelength, 202.0, 15.0) * 14.4
    p2 = normal_distribution(wavelength, 170.0, 10.0) * 6.5
    p3 = normal_distribution(wavelength, 50.0, 20.0) * 3.0
    p4 = normal_distribution(wavelength, 100.0, 25.0) * 7.0
    p5 = normal_distribution(wavelength, 140.0, 30.0) * 20.0
    p6 = normal_distribution(wavelength, 150.0, 10.0) * 3.0
    p7 = normal_distribution(wavelength, 290.0, 30.0) * 12.0
    p8 = normal_distribution(wavelength, 330.0, 80.0) * 10.0
    p9 = normal_distribution(wavelength, 240.0, 20.0) * 13.0
    p10 = normal_distribution(wavelength, 220.0, 10.0) * 2.0
    p11 = normal_distribution(wavelength, 186.0, 8.0) * 1.3
    return 0.0001 * ozone_num_density * ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11) / 1e20)


@ti.func
def get_ozone_density(h: ti.f32):
    # A curve that roughly fits measured data for ozone distribution.

    h_km = h * 0.001 # elevation in km

    h_peak_relative_sqr = h_km - ozone_peak_height * 0.001 # Square of difference between peak location
    h_peak_relative_sqr = h_peak_relative_sqr*h_peak_relative_sqr

    peak_density = 1. # density at the peak
        
    d = (peak_density - 0.375) * exp(-h_peak_relative_sqr / 49.0) # main peak
    d += 0.375 * exp(-h_peak_relative_sqr / 256.0) # "tail", makes falloff of peak more gradual
    d += ti.max(0.0, -0.000015 * pow(h_km - 15.0, 3.0)) # density becomes almost constant at low altitudes
                                                         # could modify the coefficients to model the small increase 
                                                         # in ozone at the very bottom that happens due to pollution

    return d

@ti.func
def get_density(h: ti.f32):
    h = ti.max(h, 0.0)
    return vec3(exp(-h/scale_height_rayl), exp(-h/scale_height_mie), get_ozone_density(h))

@ti.func
def get_elevation(pos: vec3):
    return ti.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - planet_r

