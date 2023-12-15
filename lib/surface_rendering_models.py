import taichi as ti
from taichi.math import *
import numpy as np
from lib.math_utils import *

DIFFUSE_FACTOR =  0.25
SPECULAR_FACTOR = 0.75

@ti.func
def earth_brdf(albedo: ti.f32, oceanness: ti.f32, bathymetry: ti.f32, v: vec3, n: vec3, l: vec3):

    h = (v+l).normalized()

    n_dot_l = saturate(n.dot(l))
    n_dot_v = saturate(n.dot(v))

    l_dot_h = saturate(l.dot(h))
    n_dot_h = saturate(n.dot(h))
    
    alpha = 0.75
    # bathymetry = 0.5 * (2.0*bathymetry - 1.0)/(alpha + (1.0 + alpha)*abs(2.0 * bathymetry - 1.0)) + 0.5

    land_roughness = 0.6
    ocean_roughness = mix(0.23+0.03, 0.23-0.06, smoothstep(0.3, 0.7, bathymetry))
    land_F_0 = 0.04
    ocean_F_0 = 0.02

    diffuse = disney_diffuse(land_roughness, n_dot_l, n_dot_v, l_dot_h)
    land_specular = GGX_smith_specular(land_roughness, land_F_0, n_dot_l, n_dot_v, l_dot_h, n_dot_h)
    ocean_specular_ggx = GGX_smith_specular(ocean_roughness, ocean_F_0, n_dot_l, n_dot_v, l_dot_h, n_dot_h)
    ocean_specular_beckmann = 0.65*beckmann_specular(ocean_roughness, ocean_F_0, n_dot_l, n_dot_v, l_dot_h, n_dot_h)
    ocean_specular = mix(ocean_specular_beckmann, ocean_specular_ggx, clamp(smoothstep(0.2, 0.95, n_dot_v), 0.05, 0.94)) # pow(n_dot_v, 0.75)

    brdf = albedo*diffuse*DIFFUSE_FACTOR + mix(land_specular, ocean_specular, oceanness)*SPECULAR_FACTOR

    return brdf, n_dot_l

@ti.func
def disney_diffuse(roughness: ti.f32, n_dot_l: ti.f32, n_dot_v: ti.f32, l_dot_h: ti.f32):

    R_R = 2.0 * roughness * sqr(l_dot_h)
    F_L = pow(1.0 - n_dot_l, 5.0)
    F_V = pow(1.0 - n_dot_v, 5.0)

    f_lambert = 1.0 / np.pi
    f_retro   = f_lambert * R_R * (F_L + F_V + F_L*F_V*(R_R - 1.0))

    f_d = f_lambert * (1.0 - 0.5*F_L) * (1.0 - 0.5*F_V) + f_retro

    return f_d

@ti.func
def beckmann_specular(roughness: ti.f32, F_0: ti.f32, \
                        n_dot_l: ti.f32, n_dot_v: ti.f32, \
                        l_dot_h: ti.f32, n_dot_h: ti.f32):

    alpha = roughness
    alpha *= alpha * 2.0
    D = beckmann_isotropic_ndf(n_dot_h, alpha)
    V = G2_VCavity(n_dot_l, n_dot_v, n_dot_h, l_dot_h) # beckmann_isotropic_visibility(n_dot_v, n_dot_l, alpha)
    F = fresnel_dielectric(l_dot_h, F_0)

    brdf = D*V*F
    # if isinf(brdf) or isnan(brdf) or brdf < 0.0:
    #     brdf = 0.0
    return brdf

@ti.func
def GGX_smith_specular(roughness: ti.f32, F_0: ti.f32, \
                        n_dot_l: ti.f32, n_dot_v: ti.f32, \
                        l_dot_h: ti.f32, n_dot_h: ti.f32):
        

    alpha2 = roughness*roughness
    D = GGX_D(n_dot_h, alpha2)
    G = G2_smith(n_dot_l, n_dot_v, alpha2)
    F = fresnel_dielectric(l_dot_h, F_0)

    return D * G* F / ti.max(4.0 * n_dot_l * n_dot_v, 1e-5)

@ti.func
def GGX_D(n_dot_h: ti.f32, alpha2: ti.f32):
    den = (alpha2 - 1.0) * n_dot_h * n_dot_h + 1.0
    return alpha2 / (np.pi * den * den)


@ti.func
def lambda_smith(NdotX: ti.f32, alpha2: ti.f32):
    n_dot_x2 = NdotX * NdotX
    return (-1.0 + ti.sqrt(alpha2 * (1.0 - n_dot_x2) / n_dot_x2 + 1.0)) * 0.5

# Masking function
@ti.func
def G1_smith(n_dot_v: ti.f32, alpha2: ti.f32):
    lambdaV = lambda_smith(n_dot_v, alpha2)
    return 1.0 / (1.0 + lambdaV)

# Height Correlated Masking-shadowing function
@ti.func
def G2_smith(n_dot_l: ti.f32, n_dot_v: ti.f32, alpha2: ti.f32):
    lambdaV = lambda_smith(n_dot_v, alpha2)
    lambdaL = lambda_smith(n_dot_l, alpha2)
    return 1.0 / (1.0 + lambdaV + lambdaL)

@ti.func
def sclick_fresnel(v_dot_h: ti.f32, F_0: ti.f32):
    return F_0 + (1 - F_0) * pow(1.0 - v_dot_h, 5.0)

@ti.func
def fresnel_dielectric(v_dot_h: ti.f32, F_0: ti.f32):
    F_0 = sqrt(F_0)
    F_0 = (1.0 + F_0) / (1.0 - F_0)

    sin_theta_I = sqrt(saturate(1.0 - sqr(v_dot_h)))
    sin_theta_T = sin_theta_I / max(F_0, 1e-8)
    cos_theta_T = sqrt(1.0 - sqr(sin_theta_T))

    R_s        = sqr((v_dot_h - (F_0 * cos_theta_T)) / max(v_dot_h + (F_0 * cos_theta_T), 1e-8))
    R_p        = sqr((cos_theta_T - (F_0 * v_dot_h)) / max(cos_theta_T + (F_0 * v_dot_h), 1e-8))

    return saturate((R_s + R_p) * 0.5)

@ti.func
def sample_ggx_vndf(V_tangent: ti.f32, rand: ti.f32, alpha: ti.f32):
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
def beckmann_isotropic_ndf(n_dot_h: ti.f32, alpha: ti.f32):
    cosTheta2 = n_dot_h*n_dot_h
    alpha2 = alpha*alpha
    exponent = (1.0-cosTheta2)/(alpha2*cosTheta2)
    denom = np.pi * alpha2 * cosTheta2 * cosTheta2
    return exp(-exponent) / max(denom, 1e-5)

@ti.func
def beckmann_isotropic_lambda(n_dot_v: ti.f32, alpha: ti.f32):
    result = 0.0
    cosTheta2 = n_dot_v*n_dot_v
    sinTheta2 = 1.0-cosTheta2
    tanTheta2 = sinTheta2 / cosTheta2

    nu = 1.0 / ti.sqrt( alpha * ti.sqrt(tanTheta2) )
    if(nu < 1.6):
        result = (1.0 - 1.259*nu + 0.396*nu*nu) / (3.535*nu + 2.181*nu*nu)
    if isinf(result) or isnan(result) or result < 0.0:
        result = 0.0
    return result

# V-Cavity Masking-shadowing function
@ti.func
def G2_VCavity(n_dot_l: ti.f32, n_dot_v: ti.f32, n_dot_h: ti.f32, v_dot_h: ti.f32):
    return min(1.0, min(2.0 * n_dot_v * n_dot_h / v_dot_h, 2.0 * n_dot_l * n_dot_h / v_dot_h))

@ti.func
def beckmann_isotropic_visibility(n_dot_v: ti.f32, n_dot_l: ti.f32, alpha: ti.f32):
    lambda_wo = beckmann_isotropic_lambda(n_dot_v, alpha)
    lambda_wi = beckmann_isotropic_lambda(n_dot_l, alpha)
    denom = (1.0 + lambda_wo + lambda_wi)*n_dot_l*n_dot_v*4.0
    result = 1.0 / denom
    if isinf(result) or isnan(result) or result < 0.0:
        result = 0.0
    
    return result



