import taichi as ti
from atmos import *
import numpy as np
from math_utils import *

# DIFFUSE BRDF & SAMPLING
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
def disney_diffuse(roughness, n_dot_l, n_dot_v, l_dot_h):

    R_R = 2.0 * roughness * sqr(l_dot_h)
    F_L = pow(1.0 - n_dot_l, 5.0)
    F_V = pow(1.0 - n_dot_v, 5.0)

    f_lambert = 1.0 / np.pi
    f_retro   = f_lambert * R_R * (F_L + F_V + F_L*F_V*(R_R - 1.0))

    f_d = f_lambert * (1.0 - 0.5*F_L) * (1.0 - 0.5*F_V) + f_retro

    return f_d

# Specular
@ti.func
def GGX_D(n_dot_h, alpha2):
    den = (alpha2 - 1.0) * n_dot_h * n_dot_h + 1.0
    return alpha2 / (np.pi * den * den)


@ti.func
def lambda_smith(NdotX, alpha2):
    n_dot_x2 = NdotX * NdotX
    return (-1.0 + ti.sqrt(alpha2 * (1.0 - n_dot_x2) / n_dot_x2 + 1.0)) * 0.5

# Masking function
@ti.func
def G1_smith(n_dot_v, alpha2):
    lambdaV = lambda_smith(n_dot_v, alpha2)
    return 1.0 / (1.0 + lambdaV)

# Height Correlated Masking-shadowing function
@ti.func
def G2_smith(n_dot_l, n_dot_v, alpha2):
    lambdaV = lambda_smith(n_dot_v, alpha2)
    lambdaL = lambda_smith(n_dot_l, alpha2)
    return 1.0 / (1.0 + lambdaV + lambdaL)

@ti.func
def sclick_fresnel(v_dot_h, F_0):
    return F_0 + (1 - F_0) * pow(1.0 - v_dot_h, 5.0)

@ti.func
def sample_ggx_vndf(V_tangent, rand, alpha):
    # stretch view direction
    V_tangent_stretched = vec3(V_tangent.xy * alpha, V_tangent.z).normalized()

    # sample spherical cap in (-wi.z, 1]
    phi = np.pi * 2.0 * rand.x
    
    hemisphere = vec3(ti.cos(phi), ti.sin(phi), 0.0)

	# normalize z
    hemisphere.z = (1.0 - rand.y) * (1.0 + V_tangent_stretched.z) + -V_tangent_stretched.z;	

    # normalize xy
    hemisphere.xy *= ti.sqrt(saturate(1.0 - hemisphere.z * hemisphere.z))

    # half vector
    hemisphere += V_tangent_stretched

    # unstretch and normalize
    return vec3(hemisphere.xy * alpha, hemisphere.z).normalized()

@ti.func
def disney_specular(roughness, F_0, \
                        n_dot_l, n_dot_v, \
                        l_dot_h, n_dot_h):
        

    alpha2 = roughness*roughness
    D = GGX_D(n_dot_h, alpha2)
    G = G2_smith(n_dot_l, n_dot_v, alpha2)
    F = sclick_fresnel(l_dot_h, F_0)

    return D * G* F / ti.max(4.0 * n_dot_l * n_dot_v, 1e-5)

@ti.func
def earth_brdf(albedo, roughness, F_0, v, n, l):
    n_dot_l = saturate(n.dot(l))
    n_dot_v = saturate(n.dot(v))

    h = (v+l).normalized()
    l_dot_h = saturate(l.dot(h))
    n_dot_h = saturate(n.dot(h))

    diffuse = disney_diffuse(roughness, n_dot_l, n_dot_v, l_dot_h)
    specular = disney_specular(roughness, F_0, n_dot_l, n_dot_v, l_dot_h, n_dot_h)

    return albedo*diffuse*n_dot_l + specular