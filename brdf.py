import taichi as ti
from atmos import *
import numpy as np
from math_utils import *

@ti.func
def disney_diffuse(roughness, n_dot_l, n_dot_v, l_dot_h):

    R_R = 2.0 * roughness * sqr(l_dot_h)
    F_L = pow(1.0 - n_dot_l, 5.0)
    F_V = pow(1.0 - n_dot_v, 5.0)

    f_lambert = 1.0 / np.pi
    f_retro   = f_lambert * R_R * (F_L + F_V + F_L*F_V*(R_R - 1.0))

    f_d = f_lambert * (1.0 - 0.5*F_L) * (1.0 - 0.5*F_V) + f_retro

    return f_d

@ti.func
def GGX_D(n_dot_h, alpha2):
    den = (alpha2 - 1.0) * n_dot_h * n_dot_h + 1.0
    return alpha2 / (np.pi * den * den)


@ti.func
def lambda_smith(NdotX, alpha):
    a2 = alpha * alpha
    n_dot_x2 = NdotX * NdotX
    return (-1.0 + sqrt(a2 * (1.0 - n_dot_x2) / n_dot_x2 + 1.0)) * 0.5

# Masking function
@ti.func
def G1_Smith(n_dot_v, alpha):
	lambdaV = lambda_smith(n_dot_v, alpha)
	return 1.0 / (1.0 + lambdaV)

# Height Correlated Masking-shadowing function
@ti.func
def G2_Smith(n_dot_l, n_dot_v, alpha):
	lambdaV = lambda_smith(n_dot_v, alpha)
	lambdaL = lambda_smith(n_dot_l, alpha)
	return 1.0 / (1.0 + lambdaV + lambdaL)

@ti.func
def sclick_fresnel(v_dot_h, F_0):
    return F_0 + (1 - F_0) * pow(1.0 - v_dot_h, 5.0)

@ti.func
def sample_ggx_vndf(V_tangent, vec2 Xi, float alpha)
{
	//stretch the view direction
    vec3 V_tangent_stretched = normalize(vec3(V_tangent.xy * alpha, V_tangent.z));

	//sample a spherical cap in (-wi.z, 1]
    float phi = PI * 2.0 * Xi.x;
    
	vec3 hemisphere = vec3(cos(phi), sin(phi), 0.0);

	//normalize (z)
	hemisphere.z = (1.0 - Xi.y) * (1.0 + V_tangent_stretched.z) + -V_tangent_stretched.z;	

	//normalize (hemi * sin theta)
	hemisphere.xy *= sqrt(clamp(1.0 - hemisphere.z * hemisphere.z, 0.0, 1.0));

	//halfway direction
	hemisphere += V_tangent_stretched;

	//unstretch and normalize
	return normalize(vec3(hemisphere.xy * alpha, hemisphere.z));
}

@ti.func
def disney_specular(self, roughness, F_0, \
                        n_dot_l, n_dot_v, \
                        l_dot_h, n_dot_h, \
                        h_dot_x, h_dot_y, \
                        l_dot_x, l_dot_y, \
                        v_dot_x, v_dot_y, \
                        tang, bitang): # specular REFLECTION
        

    ax = sqr(mat.roughness)

    D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ax)
    G = self.smithG_GGX_aniso(n_dot_l, l_dot_x, l_dot_y, ax, ax) \
    * self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ax)
    F = self.sclick_fresnel(l_dot_h, F_0)

    return D * G* F# / ti.max(4.0 * n_dot_l * n_dot_v, 1e-5)