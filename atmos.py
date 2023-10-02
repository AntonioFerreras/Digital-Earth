import math
import taichi as ti
from taichi.math import *
import numpy as np
from math_utils import *



@ti.func
def rayleigh_phase(cos_theta):
    return 3.0/(16.0*np.pi)*(1.0 + cos_theta*cos_theta)

@ti.func
def mie_phase(cos_theta, g):
    # Henyey-Greenstein phase
    return (1-g*g)/(4.0*np.pi*pow(1.0 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def get_unit_vec(rand):
    rand.x *= np.pi * 2.0; rand.y = rand.y * 2.0 - 1.0
    ground = vec2(sin(rand.x), cos(rand.x)) * sqrt(1.0 - rand.y * rand.y)
    return vec3(ground.x, ground.y, rand.y).normalized()


@ti.data_oriented
class Atmos:
    def __init__(self):
        # Atmos Constants
        self.air_num_density       = 2.5035422e25
        self.ozone_peak = 8e-6
        self.ozone_num_density     = self.air_num_density * 0.012588 * self.ozone_peak
        self.ozone_cross_sec      = vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22) * 0.0001

        self.rayleigh_coeff = vec3(0.00000519673, 0.0000121427, 0.0000296453) # scattering coeff
        self.mie_coeff = 8.6e-6 # scattering coeff
        self.ozone_coeff    = self.ozone_cross_sec*self.ozone_num_density
        self.extinc_mat = ti.Matrix([[self.rayleigh_coeff.x, self.rayleigh_coeff.y, self.rayleigh_coeff.z], \
                                     [self.mie_coeff*1.11, self.mie_coeff*1.11, self.mie_coeff*1.11]      , \
                                     [self.ozone_coeff.x, self.ozone_coeff.y, self.ozone_coeff.z]], ti.f32).transpose()
        self.scatter_mat = ti.Matrix([[self.rayleigh_coeff.x, self.rayleigh_coeff.y, self.rayleigh_coeff.z], \
                                      [self.mie_coeff, self.mie_coeff, self.mie_coeff]      , \
                                      [0.0, 0.0, 0.0]], ti.f32).transpose()

        self.scale_height_rayl = 8500.0
        self.scale_height_mie  = 1200.0

        self.scale_heights = ti.Vector([self.scale_height_rayl, self.scale_height_mie])

        self.mie_g = 0.75

        self.planet_r = 6378137
        self.atmos_height  = 110e3

        # Cloud constants
        self.cloud_height = 1000.0 + 1e3
        self.cloud_thickness = 170.0*2.0
        self.cloud_density = 0.27
        self.cloud_extinc = 0.075
        self.cloud_scatter = self.cloud_extinc
        #############


    @ti.func
    def get_ozone_density(self, h):
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

        return d * 4.

    @ti.func
    def get_density(self, h):
        h = ti.max(h, 0.0)
        return vec3(exp(-h/self.scale_height_rayl), exp(-h/self.scale_height_mie), self.get_ozone_density(h))

    @ti.func
    def get_elevation(self, pos):
        return ti.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - self.planet_r
    
    ######
