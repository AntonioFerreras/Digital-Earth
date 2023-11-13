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


scale_height_rayl = 8500.0
scale_height_mie  = 1200.0

scale_heights = ti.Vector([scale_height_rayl, scale_height_mie])

mie_g = 0.75

planet_r = 6378137.0
atmos_height  = 110e3
atmos_upper_limit = planet_r + atmos_height

# Cloud constants
cloud_height = 1000.0 + 1e3
cloud_thickness = 170.0*2.0
cloud_density = 0.27
cloud_extinc = 0.075
cloud_scatter = cloud_extinc
#############


@ti.func
def rayleigh_phase(cos_theta):
    return 3.0/(16.0*np.pi)*(1.0 + cos_theta*cos_theta)

@ti.func
def mie_phase(cos_theta, g):
    # Henyey-Greenstein phase
    return (1-g*g)/(4.0*np.pi*pow(1.0 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def air(wavelength):
    rcp_wavelength_sqr = 1.0 / (wavelength*wavelength)
    return 1.0+8.06051E-5+2.480990E-2/(132.274-rcp_wavelength_sqr)+1.74557E-4/(39.32957-rcp_wavelength_sqr)

@ti.func
def spectra_extinction_mie(wavelength):
    junge = 4.0
    turbidity = 1.0

    c = (0.6544 * turbidity - 0.6510) * 4e-18;
    K = (0.773335 - 0.00386891 * wavelength) / (1.0 - 0.00546759 * wavelength)
    return 0.434 * c * pi * pow(2.0*np.pi / (wavelength * 1e-9), junge - 2.0) * K


@ti.func
def spectra_extinction_rayleigh(wavelength):
    nanometers = wavelength * 1e-9
    micrometers = wavelength * 1e-6
    air_number_density  = 2.5035422e25

    F_N2 = 1.034 + 3.17e-4 * (1.0 / pow(wavelength, 2.0))
    F_O2 = 1.096 + 1.385e-3 * (1.0 / pow(wavelength, 2.0)) + 1.448e-4 * (1.0 / pow(wavelength, 4.0))
    CCO2 = 0.045
    king_factor = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 + CCO2 * 1.15) / (78.084 + 20.946 + 0.934 + CCO2)
    n = sqr(air(wavelength * 1e-3)) - 1.0

    return ((8.0 * pow(pi, 3.0) * pow(n, 2.0)) / (3.0 * air_number_density * pow(nanometers, 4.0))) * king_factor


@ti.func
def get_ozone_density(h):
    # A curve that roughly fits measured data for ozone distribution.

    h_km = h * 0.001 # elevation in km

    peak_height = 25.0
    h_peak_relative_sqr = h_km - peak_height # Square of difference between peak location
    h_peak_relative_sqr = h_peak_relative_sqr*h_peak_relative_sqr

    peak_density = 1. # density at the peak
        
    d = (peak_density - 0.375) * exp(-h_peak_relative_sqr / 49.0) # main peak
    d += 0.375 * exp(-h_peak_relative_sqr / 256.0) # "tail", makes falloff of peak more gradual
    d += ti.max(0.0, -0.000015 * pow(h_km - 15.0, 3.0)) # density becomes almost constant at low altitudes
                                                         # could modify the coefficients to model the small increase 
                                                         # in ozone at the very bottom that happens due to pollution

    return d

@ti.func
def get_density(h):
    h = ti.max(h, 0.0)
    return vec3(exp(-h/scale_height_rayl), exp(-h/scale_height_mie), get_ozone_density(h))

@ti.func
def get_elevation(pos):
    return ti.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - planet_r

