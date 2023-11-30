import math
import taichi as ti
from taichi.math import *
import numpy as np
from lib.math_utils import *
from lib.sampling import *

# Atmos Constants
air_num_density       = 2.5035422e25
ozone_peak = 8e-6
ozone_num_density     = air_num_density * 0.012588 * ozone_peak
ozone_cross_sec      = vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22) * 0.0001
ozone_peak_height = 25000.0 # peak density at 25km



mie_g = 0.75
mie_asymmetry = 3000.0
turbidity = 1.0

rayleigh_albedo = 1.0
aerosol_albedo = 0.98
ozone_albedo = 0.0
cloud_albedo = 0.99



planet_r = 6371e3
atmos_height  = 110e3
atmos_upper_limit = planet_r + atmos_height

# Cloud constants
clouds_extinct = 0.1
clouds_density = 0.025*0
clouds_height = 4000.0
clouds_thickness = 6000.0
clouds_lower_limit = planet_r + clouds_height
clouds_upper_limit = clouds_lower_limit + clouds_thickness
#############

# Refractive index of air
@ti.func
def air(wavelength: ti.f32):
    rcp_wavelength_sqr = 1.0 / (wavelength*wavelength)
    return 1.0+8.06051e-5+2.480990e-2/(132.274-rcp_wavelength_sqr)+1.74557e-4/(39.32957-rcp_wavelength_sqr)

# PHASE FUNCTIONS

# @ti.func
# def rayleigh_phase(cos_theta: ti.f32, wavelength: ti.f32):
#     return ((1.0 + sqr(cos_theta)) / (2.0 * 2.0 * sqr(1.0))) * \
#            pow(2.0 * pi / wavelength, 4.0) * sqr((sqr(air(wavelength * 1e-3)) - 1.0) / \
#            (sqr(air(wavelength * 1e-3)) + 2.0)) * pow(5.0 / 2.0, 6.0)

@ti.func
def rayleigh_phase(cos_theta: ti.f32):
    return 3.0/(16.0*np.pi)*(1.0 + cos_theta*cos_theta)

@ti.func
def mie_phase(cos_theta: ti.f32):
    # Henyey-Greenstein phase
    return klein_nishina_phase(cos_theta, mie_asymmetry)

@ti.func
def sample_mie_phase(view: vec3):
    return sample_klein_nishina_phase(view, mie_asymmetry)

@ti.func
def hg_phase(cos_theta: ti.f32, g: ti.f32):
    # Henyey-Greenstein phase
    return (1-g*g)/(4.0*np.pi*pow(1.0 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def sample_hg_phase(view: vec3, g: ti.f32):
    sqr_term = (1 - g * g) / (1 - g + 2 * g * ti.random())
    cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g)
    sin_theta = sqrt(max(0.0, 1 - cos_theta * cos_theta))
    phi = 2.0 * pi * ti.random()
    tang, bitang = make_orthonormal_basis(view)
    return spherical_direction(sin_theta, cos_theta, phi, tang, bitang, view)

@ti.func
def klein_nishina_phase(cos_theta: ti.f32, e: ti.f32):
    return e / (2.0 * pi * (e * (1.0 - cos_theta) + 1.0) * log(2.0 * e + 1.0))

@ti.func
def sample_klein_nishina_phase(view: vec3, e: ti.f32):
    cos_theta = (-pow(2.0 * e + 1.0, 1.0 - ti.random()) + e + 1.0) / e
    sin_theta = sqrt(max(0.0, 1 - cos_theta * cos_theta))
    phi = 2.0 * pi * ti.random()
    tang, bitang = make_orthonormal_basis(view)
    return spherical_direction(sin_theta, cos_theta, phi, tang, bitang, view)


# SPDX-FileCopyrightText: Copyright (c) <2023> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
@ti.func
def draine_phase(cos_theta: ti.f32, g: ti.f32, a: ti.f32):
    return ((1 - g*g)*(1 + a*cos_theta*cos_theta))/(4.*(1 + (a*(1 + 2*g*g))/3.) * pi * pow(1 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def sample_draine(view: vec3, g: ti.f32, a: ti.f32):
    xi = ti.random()
    g2 = g * g
    g3 = g * g2
    g4 = g2 * g2
    g6 = g2 * g4
    pgp1_2 = (1 + g2) * (1 + g2)
    T1 = (-1 + g2) * (4 * g2 + a * pgp1_2)
    T1a = -a + a * g4
    T1a3 = T1a * T1a * T1a
    T2 = -1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2)
    T3 = 3 * g2 * (1 + g * (-1 + 2 * xi)) + a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * xi))
    T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3
    T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6
    T4b3 = T4b * T4b * T4b
    T4 = T4a + sqrt(-4 * T4b3 + T4a * T4a)
    T4p3 = pow(T4, 1.0 / 3.0)
    T6 = (2 * T1a + (48 * pow(2, 1.0 / 3.0) *
		(-(a * g2) + 2 * a * g4 - a * g6)) / T4p3 + T4p3 / (3. * pow(2, 1.0 / 3.0))) / (a - a * g2)
    T5 = 6 * (1 + g2) + T6
    cos_theta = (1 + g2 - pow(-0.5 * sqrt(T5) + sqrt(6 * (1 + g2) - (8 * T3) / (a * (-1 + g2) * sqrt(T5)) - T6) / 2., 2)) / (2. * g)
    sin_theta = sqrt(max(0.0, 1 - cos_theta * cos_theta))
    phi = 2.0 * pi * ti.random()
    tang, bitang = make_orthonormal_basis(view)
    return spherical_direction(sin_theta, cos_theta, phi, tang, bitang, view)
    
# END OF NVIDIA CORPORATION & AFFILIATES CODE

@ti.func
def cloud_phase(cos_theta: ti.f32):
    # d = 35.0 # droplet size
    # g_hg = exp( -0.0990567 / (d - 1.67154) )
    # g_draine = exp( -2.20679 / (d + 3.91029) - 0.428934 )
    # alpha_draine = exp( 3.62489 - 8.29288 / (d + 5.52825) )
    # w_draine = exp( -0.599085 / (d - 0.641583) - 0.665888 )

    # return mix(hg_phase(cos_theta, g_hg), draine_phase(cos_theta, g_draine, alpha_draine), w_draine)
    return mix(hg_phase(cos_theta, -0.4), hg_phase(cos_theta, 0.8), 0.7)

@ti.func
def sample_cloud_phase(view: vec3):
    # d = 35.0 # droplet size
    # g_hg = exp( -0.0990567 / (d - 1.67154) )
    # g_draine = exp( -2.20679 / (d + 3.91029) - 0.428934 )
    # alpha_draine = exp( 3.62489 - 8.29288 / (d + 5.52825) )
    # w_draine = exp( -0.599085 / (d - 0.641583) - 0.665888 )

    # dir = vec3(0.0, 0.0, 0.0)
    # if ti.random() < w_draine:
    #     dir = sample_draine(view, g_draine, alpha_draine)
    # else:
    #     dir = sample_hg_phase(view, g_hg)
    dir = vec3(0.0, 0.0, 0.0)
    if ti.random() < 0.7:
        dir = sample_hg_phase(view, 0.8)
    else:
        dir = sample_hg_phase(view, -0.4)
    return dir

##############################

# SPECTRA

@ti.func
def spectra_extinction_mie2(wavelength: ti.f32):
    B = 0.0009
    return B/wavelength

@ti.func
def spectra_extinction_mie(wavelength: ti.f32):
    junge = 4.0
    
    c = (0.6544 * turbidity - 0.6510) * 4e-18
    K = (0.773335 - 0.00386891 * wavelength) / (1.0 - 0.00546759 * wavelength)
    return 0.434 * c * np.pi * pow(2.0*np.pi / (wavelength * 1e-9), junge - 2.0) * K


@ti.func
def spectra_extinction_rayleigh(wavelength: ti.f32):
    nanometers = wavelength * 1e-9

    # depolarization 
    F_N2 = 1.034 + 3.17e-4 * (1.0 / pow(wavelength, 2.0))
    F_O2 = 1.096 + 1.385e-3 * (1.0 / pow(wavelength, 2.0)) + 1.448e-4 * (1.0 / pow(wavelength, 4.0))

    # concentration of CO2 in ppm
    CCO2 = 0.0421

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

#######################

# DENSITY FUNCTIONS
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

    return d*2.0

@ti.func
def get_rayl_density(h: ti.f32):
    # Gaussian curve fit to US standard atmosphere
    density_sea_level = 1.225
    return 3.68082 * exp( -pow(h + 24239.99, 2.0)/532307548.4168 ) # / density_sea_level

@ti.func
def get_mie_density(h: ti.f32):
    # A smooth-ish version of the OPAC aerosol density function
    dens = 0.0
    if h > 11500.0:
        dens = 0.0918 * exp(-1.0e-6*pow(h - 11500.0, 2.0))
    elif h > 2400.0:
        dens = 0.3000 * exp(-2.5e-9*pow(h + 2500.00, 2.0)) - 0.092
    elif h > 1300.0:
        dens = 0.6500 * exp(-5.0e-6*pow(h - 1300.00, 2.0)) + 0.18899
    else:
        dens = 1.0 - h/8136.646
    
    return dens*turbidity


@ti.func
def get_density(h: ti.f32):
    h = ti.max(h, 0.0)
    return vec3(get_rayl_density(h), get_mie_density(h), get_ozone_density(h))

@ti.func
def get_elevation(pos: vec3):
    return ti.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - planet_r

#######################