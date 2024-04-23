import taichi as ti
from taichi.math import *
from lib.colour import *

#  OpenDRT -------------------------------------------------/
#     v0.2.2
#     Originally written by Jed Smith, ported to Taichi
#     https://github.com/jedypod/open-display-transform
#
#     License: GPL v3

# Colour space IDs
i_xyz = 0
i_ap0 = 1
i_ap1 = 2
i_p3d65 = 3
i_rec2020 = 4
i_rec709 = 5
i_awg3 = 6
i_awg4 = 7
i_rwg = 8
i_sgamut3 = 9
i_sgamut3cine = 10
i_bmdwg = 11
i_egamut = 12
i_davinciwg = 13

Rec709 = 0
P3D65 = 1
Rec2020 = 2

lin = 0
srgb = 1
rec1886 = 2
dci = 3
pq = 4
hlg = 5

in_gamut = i_rec709
display_gamut = Rec709
EOTF = lin

# Constants
Lp = 100.0  # Lp
gb = 0.12   # Lp exp boost
c = 1.4     # contrast
fl = 0.005  # flare
rw = 0.25   # red weight
bw = 0.35   # blue weight
dch = 0.35  # dechroma
dch_toe = 0.0  # toe dechroma
hs_r = 0.3  # hue shift red
hs_g = -0.1  # hue shift green
hs_b = -0.2  # hue shift blue
v_p = 0.5  # chroma value

# Gamut Conversion Matrices
matrix_ap0_to_xyz = mat3(vec3(0.93863094875, -0.00574192055, 0.017566898852), vec3(0.338093594922, 0.727213902811, -0.065307497733), vec3(0.000723121511, 0.000818441849, 1.0875161874))
matrix_ap1_to_xyz = mat3(vec3(0.652418717672, 0.127179925538, 0.170857283842), vec3(0.268064059194, 0.672464478993, 0.059471461813), vec3(-0.00546992851, 0.005182799977, 1.08934487929))
matrix_rec709_to_xyz = mat3(vec3(0.412390917540, 0.357584357262, 0.180480793118), vec3(0.212639078498, 0.715168714523, 0.072192311287), vec3(0.019330825657, 0.119194783270, 0.950532138348))
matrix_p3d65_to_xyz = mat3(vec3(0.486571133137, 0.265667706728, 0.198217317462), vec3(0.228974640369, 0.691738605499, 0.079286918044), vec3(-0.000000000000, 0.045113388449, 1.043944478035))
matrix_rec2020_to_xyz = mat3(vec3(0.636958122253, 0.144616916776, 0.168880969286), vec3(0.262700229883, 0.677998125553, 0.059301715344), vec3(0.000000000000, 0.028072696179, 1.060985088348))
matrix_arriwg3_to_xyz = mat3(vec3(0.638007619284, 0.214703856337, 0.097744451431), vec3(0.291953779, 0.823841041511, -0.11579482051), vec3(0.002798279032, -0.067034235689, 1.15329370742))
matrix_arriwg4_to_xyz = mat3(vec3(0.704858320407, 0.12976029517, 0.115837311474), vec3(0.254524176404, 0.781477732712, -0.036001909116), vec3(0.0, 0.0, 1.08905775076))
matrix_redwg_to_xyz = mat3(vec3(0.735275208950, 0.068609409034, 0.146571278572), vec3(0.286694079638, 0.842979073524, -0.129673242569), vec3(-0.079680845141, -0.347343206406, 1.516081929207))
matrix_sonysgamut3_to_xyz = mat3(vec3(0.706482713192, 0.128801049791, 0.115172164069), vec3(0.270979670813, 0.786606411221, -0.057586082034), vec3(-0.009677845386, 0.004600037493, 1.09413555865))
matrix_sonysgamut3cine_to_xyz = mat3(vec3(0.599083920758, 0.248925516115, 0.102446490178), vec3(0.215075820116, 0.885068501744, -0.100144321859), vec3(-0.032065849545, -0.027658390679, 1.14878199098))
matrix_bmdwg_to_xyz = mat3(vec3(0.606538414955, 0.220412746072, 0.123504832387), vec3(0.267992943525, 0.832748472691, -0.100741356611), vec3(-0.029442556202, -0.086612440646, 1.205112814903))
matrix_egamut_to_xyz = mat3(vec3(0.705396831036, 0.164041340351, 0.081017754972), vec3(0.280130714178, 0.820206701756, -0.100337378681), vec3(-0.103781513870, -0.072907261550, 1.265746593475))
matrix_davinciwg_to_xyz = mat3(vec3(0.700622320175, 0.148774802685, 0.101058728993), vec3(0.274118483067, 0.873631775379, -0.147750422359), vec3(-0.098962903023, -0.137895315886, 1.325916051865))

matrix_xyz_to_rec709 = mat3(vec3(3.2409699419, -1.53738317757, -0.498610760293), vec3(-0.969243636281, 1.87596750151, 0.041555057407), vec3(0.055630079697, -0.203976958889, 1.05697151424))
matrix_xyz_to_p3d65 = mat3(vec3(2.49349691194, -0.931383617919, -0.402710784451), vec3(-0.829488969562, 1.76266406032, 0.023624685842), vec3(0.035845830244, -0.076172389268, 0.956884524008))
matrix_xyz_to_rec2020 = mat3(vec3(1.71665118797, -0.355670783776, -0.253366281374), vec3(-0.666684351832, 1.61648123664, 0.015768545814), vec3(0.017639857445, -0.042770613258, 0.942103121235))

# Helper functions
@ti.func
def _logf(x):
	return log2(x) / log2(10.0)

@ti.func
def _expf(x):
	return pow(10.0, x)

# Multiply 3x3 matrix m and vec3 vector v
@ti.func
def vdot(m: mat3, v: vec3):
	return v @ m


# Safe division of a by b
@ti.func
def sdivf(a: ti.f32, b: ti.f32):
	result = 0.0
	if (abs(b) < 1e-4): result == 0.0
	else: result = a/b
	return result

# Safe division of vec3 a by b
@ti.func
def sdivf3f(a: vec3, b: ti.f32):
	return vec3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b))

# Safe division of vec3 a by vec3 b
@ti.func
def sdivf3f3(a: vec3, b: vec3):
	return vec3(sdivf(a.x, b.x), sdivf(a.y, b.y), sdivf(a.z, b.z))


# Safe power function raising a to power b
@ti.func
def spowf(a: ti.f32, b: ti.f32):
	result = 0.0
	if a <= 0.0: result = a
	else: result = pow(a, b)
	return result

# Safe power function raising vec3 a to power b
@ti.func
def spowf3(a: vec3, b: ti.f32):
	return vec3(pow(a.x, b), pow(a.y, b), pow(a.z, b))

# Return max of vec3 a and float mn
@ti.func
def maxf3(mn: ti.f32, a: vec3): 
	return vec3(max(a.x, mn), max(a.y, mn), max(a.z, mn))

# Return min of vec3 a and float mx
@ti.func
def minf3(mx: ti.f32, a: vec3): 
	return vec3(min(a.x, mx), min(a.y, mx), min(a.z, mx))

@ti.func
def eotf_hlg(rgb: vec3, inverse: ti.i32):
	# Apply the HLG Forward or Inverse EOTF. Implements the full ambient surround illumination model
	# ITU-R Rec BT.2100-2 https://www.itu.int/rec/R-REC-BT.2100
	# ITU-R Rep BT.2390-8: https://www.itu.int/pub/R-REP-BT.2390
	# Perceptual Quantiser (PQ) to Hybrid Log-Gamma (HLG) Transcoding: https://www.bbc.co.uk/rd/sites/50335ff370b5c262af000004/assets/592eea8006d63e5e5200f90d/BBC_HDRTV_PQ_HLG_Transcode_v2.pdf

	HLG_Lw = 1000.0
	#HLG_Lb = 0.0
	HLG_Ls = 5.0
	h_a = 0.17883277
	h_b = 1.0 - 4.0*0.17883277
	h_c = 0.5 - h_a*_logf(4.0*h_a)
	h_g = 1.2*pow(1.111, log2(HLG_Lw/1000.0))*pow(0.98, log2(max(1e-6, HLG_Ls)/5.0))
	
	if inverse == 1:
		Yd = 0.2627*rgb.x + 0.6780*rgb.y + 0.0593*rgb.z
		# HLG Inverse OOTF
		rgb = rgb*pow(Yd, (1.0 - h_g)/h_g)
		# HLG OETF
		rgb.x = sqrt(3.0*rgb.x) if rgb.x <= 1.0/12.0 else h_a*_logf(12.0*rgb.x - h_b) + h_c
		rgb.y = sqrt(3.0*rgb.y) if rgb.y <= 1.0/12.0 else h_a*_logf(12.0*rgb.y - h_b) + h_c
		rgb.z = sqrt(3.0*rgb.z) if rgb.z <= 1.0/12.0 else h_a*_logf(12.0*rgb.z - h_b) + h_c
	else:
		# HLG Inverse OETF
		rgb.x = rgb.x*rgb.x/3.0 if rgb.x <= 0.5 else (_expf((rgb.x - h_c)/h_a) + h_b)/12.0
		rgb.y = rgb.y*rgb.y/3.0 if rgb.y <= 0.5 else (_expf((rgb.y - h_c)/h_a) + h_b)/12.0
		rgb.z = rgb.z*rgb.z/3.0 if rgb.z <= 0.5 else (_expf((rgb.z - h_c)/h_a) + h_b)/12.0
		# HLG OOTF
		Ys = 0.2627*rgb.x + 0.6780*rgb.y + 0.0593*rgb.z
		rgb = rgb*pow(Ys, h_g - 1.0)
	return rgb

@ti.func
def eotf_pq(rgb: vec3, inverse: ti.i32):
	# Apply the ST-2084 PQ Forward or Inverse EOTF
	#   ITU-R Rec BT.2100-2 https://www.itu.int/rec/R-REC-BT.2100
	#   ITU-R Rep BT.2390-9 https://www.itu.int/pub/R-REP-BT.2390
	#   Note: in the spec there is a normalization for peak display luminance. 
	#   For this function we assume the input is already normalized such that 1.0 = 10,000 nits

	# Lp = 1.0
	m1 = 2610.0/16384.0
	m2 = 2523.0/32.0
	c1 = 107.0/128.0
	c2 = 2413.0/128.0
	c3 = 2392.0/128.0

	if inverse == 1:
		# rgb /= Lp
		rgb = spowf3(rgb, m1)
		rgb = spowf3((c1 + c2*rgb)/(1.0 + c3*rgb), m2)
	else:
		rgb = spowf3(rgb, 1.0/m2)
		rgb = spowf3((rgb - c1)/(c2 - c3*rgb), 1.0/m1)
		# rgb *= Lp
	return rgb

@ti.func
def narrow_hue_angles(v):
    return ti.Vector([
        min(2.0, max(0.0, v.x - (v.y + v.z))),
        min(2.0, max(0.0, v.y - (v.x + v.z))),
        min(2.0, max(0.0, v.z - (v.x + v.y)))
    ])

# tonescale function
@ti.func
def tonescale(x: ti.f32, m: ti.f32, s: ti.f32, c: ti.f32, invert: ti.i32):
	result = 0.0
	if invert == 0:
		result = spowf(m*x/(x + s), c)
	else:
		ip = 1.0/c
		result = spowf(s*x, ip)/(m - spowf(x, ip))
	return result

# flare function
@ti.func
def flare(x: ti.f32, fl: ti.f32, invert: ti.i32):
	result = 0.0
	if invert == 0:
		result = spowf(x, 2.0)/(x+fl)
	else:
		result = (x + sqrt(x*(4.0*fl + x)))/2.0
	return result

# OpenDRTransform function
@ti.func
def openDR_transform(p_R: float, p_G: float, p_B: float) -> vec3:
    # **************************************************
	# Parameter Setup
	# --------------------------------------------------

	# Input gamut conversion matrix (CAT02 chromatic adaptation to D65)
	in_to_xyz = mat3(0.0)
	if (in_gamut == i_xyz): in_to_xyz = mat3(1.0)
	elif (in_gamut == i_ap0): in_to_xyz = matrix_ap0_to_xyz
	elif (in_gamut == i_ap1): in_to_xyz = matrix_ap1_to_xyz
	elif (in_gamut == i_p3d65): in_to_xyz = matrix_p3d65_to_xyz
	elif (in_gamut == i_rec2020): in_to_xyz = matrix_rec2020_to_xyz
	elif (in_gamut == i_rec709): in_to_xyz = matrix_rec709_to_xyz
	elif (in_gamut == i_awg3): in_to_xyz = matrix_arriwg3_to_xyz
	elif (in_gamut == i_awg4): in_to_xyz = matrix_arriwg4_to_xyz
	elif (in_gamut == i_rwg): in_to_xyz = matrix_redwg_to_xyz
	elif (in_gamut == i_sgamut3): in_to_xyz = matrix_sonysgamut3_to_xyz
	elif (in_gamut == i_sgamut3cine): in_to_xyz = matrix_sonysgamut3cine_to_xyz
	elif (in_gamut == i_bmdwg): in_to_xyz = matrix_bmdwg_to_xyz
	elif (in_gamut == i_egamut): in_to_xyz = matrix_egamut_to_xyz
	elif (in_gamut == i_davinciwg): in_to_xyz = matrix_davinciwg_to_xyz

	xyz_to_display = mat3(0.0)
	if (display_gamut == Rec709): xyz_to_display = matrix_xyz_to_rec709
	elif (display_gamut == P3D65): xyz_to_display = matrix_xyz_to_p3d65
	elif (display_gamut == Rec2020): xyz_to_display = matrix_xyz_to_rec2020

	eotf = 0
	if (EOTF == lin):       eotf = 0
	elif (EOTF == srgb):    eotf = 1
	elif (EOTF == rec1886): eotf = 2
	elif (EOTF == dci):     eotf = 3
	elif (EOTF == pq):      eotf = 4
	elif (EOTF == hlg):     eotf = 5

	# Display Scale ---------------
	#   Remap peak white in display linear depending on the selected inverse EOTF.
	#   In our tonescale model, 1.0 is 100 nits, and as we scale up peak display luminance (Lp),
	#   we multiply up by the same amount. So if Lp=1,000, peak output of the tonescale model
	#   will be 10.0.
	#
	#   So in ST2084 PQ, 1.0 is 10,000 nits, so we need to divide by 100 to fit out output into the 
	#   container.
	#
	#   Similarly in HLG, 1.0 is 1,000 nits, so we need to divide by 10.
	#
	#   If we are in an SDR mode, instead we just scale the peak so it hits display 1.0.
	
	ds = 0.01 if eotf == 4 else (0.1 if eotf == 5 else 100.0/Lp)
	clamp_max = ds*Lp/100.0


	# Tonescale Parameters 
	#    ----------------------
	# For the tonescale compression function, we use one inspired by the wisdom shared by Daniele Siragusano
	# on the tonescale thread on acescentral: https://community.acescentral.com/t/output-transform-tone-scale/3498/224
	#
	# This is a variation which puts the power function _after_ the display-linear scale, which allows a simpler and exact
	# solution for the intersection constraints. The resulting function is pretty much identical.
	# Here is a desmos graph with the math. https://www.desmos.com/calculator/pmf3qc47v0
	#
	# And for more info on the derivation, see the "Michaelis-Menten Constrained" Tonescale Function here:
	# https://colab.research.google.com/drive/1aEjQDPlPveWPvhNoEfK4vGH5Tet8y1EB#scrollTo=Fb_8dwycyhlQ
	#
	# For the user parameter space, we include the following creative controls:
	# - Lp: display peak luminance. This sets the display device peak luminance and allows rendering for HDR.
	# - contrast: This is a pivoted power function applied after the hyperbolic compress function, 
	#     which keeps middle grey and peak white the same but increases contrast in between.
	# - flare: Applies a parabolic toe compression function after the hyperbolic compression function. 
	#     This compresses values near zero without clipping. Used for flare or glare compensation.
	# - gb: Grey Boost. This parameter controls how many stops to boost middle grey per stop of peak luminance increase.
	#
	# Notes on the other non user-facing parameters:
	# - (px, py): This is the peak luminance intersection constraint for the compression function.
	#     px is the input scene-linear x-intersection constraint. That is, the scene-linear input value 
	#     which is mapped to py through the compression function. By default this is set to 64 at Lp=100, and 128 at Lp=1000.
	#     Here is the regression calculation using a logarithmic function to match: https://www.desmos.com/calculator/d0jcaa8xzt
	# - (gx, gy): This is the middle grey intersection constraint for the compression function.
	#     Scene-linear input value gx is mapped to display-linear output gy through the function.
	#     Why is gy set to 0.11696 at Lp=100? This matches the position of middle grey through in the Rec709 system. 
	#     We use this value for consistency with the Arri and TCAM Rec.1886 display rendering transforms.


	# input scene-linear peak x intercept
	px = 128.0*_logf(Lp)/_logf(100.0) - 64.0
	# output display-linear peak y intercept
	py = Lp/100.0
	# input scene-linear middle grey x intercept
	gx = 0.18
	# output display-linear middle grey y intercept
	gy = 11.696/100.0*(1.0 + gb*_logf(py)/_logf(2.0))
	# s0 and s are input x scale for middle grey intersection constraint
	# m0 and m are output y scale for peak white intersection constraint
	s0 = flare(gy, fl, 1)
	m0 = flare(py, fl, 1)
	ip = 1.0/c
	s = (px*gx*(pow(m0, ip) - pow(s0, ip)))/(px*pow(s0, ip) - gx*pow(m0, ip))
	m = pow(m0, ip)*(s + px)/px



	# Rendering Code ------------------------------------------ 

	rgb = vec3(p_R, p_G, p_B)

	# Convert into display gamut
	rgb = vdot(in_to_xyz, rgb)
	rgb = vdot(xyz_to_display, rgb)


	# Take the min and the max of rgb. These are used to calculate hue angle, chrominance, and rgb ratios
	mx = max(rgb.x, max(rgb.y, rgb.z))
	mn = min(rgb.x, min(rgb.y, rgb.z))

	# Calculate RGB CMY hue angles from the input RGB.
	# The classical way of calculating hue angle from RGB is something like this
	# mx = max(r,g,b)
	# mn = min(r,g,b)
	# c = mx - mn
	# hue = (c==0?0:r==mx?((g-b)/c+6)%6:g==mx?(b-r)/c+2:b==mx?(r-g)/c+4:0)
	# With normalized chroma (distance from achromatic), being calculated like this
	# chroma = (mx - mn)/mx
	# chroma can also be calculated as 1 - mn/mx
	#
	# Here we split apart the calculation for hue and chroma so that we have access to RGB CMY
	# individually without having to linear step extract the result again.
	#
	# To do this, we first calculate the "wide" hue angle: 
	#   wide hue RGB = (RGB - mn)/mx
	#   wide hue CMY = (mx - RGB)/mx
	# and then "narrow down" the hue angle for each with channel subtraction (see narrow_hue_angles() function).

	h_rgb = sdivf3f(rgb - mn, mx)
	h_rgb = narrow_hue_angles(h_rgb)
	h_cmy = sdivf3f(mx - rgb, mx)
	h_cmy = narrow_hue_angles(h_cmy)

	# chroma here does not refer to any perceptual metric. It is merely the normalized distance from achromatic 
	ch = 1.0 - sdivf(mn, mx)


	#   Take the hypot (length) of the weighted sum of RGB.
	#   We use this as a vector norm for separating color and intensity.
	#   The R and B weights are creatively controlled by the user. 
	#   They specify the "vibrancy" of the colors. The weights are normalized
	#   so that achromatic peak does not change. Green weight is kept at 1.

	w = vec3(rw, 1.0, bw)
	w /= length(w) # normalize weights. these could be pre-calculated
	w *= maxf3(1e-5, rgb) # multiply rgb by weights
	lum = length(w) # take the norm


	# RGB Ratios
	rats = sdivf3f(rgb, lum)

	# Apply tonescale function to lum
	ts = tonescale(lum, m, s, c, 0)
	ts = flare(ts, fl, 0)

	# Unused inverse direction
	# ts = flare(ts, fl, 1)
	# ts = tonescale(lum, m, s, c, 1)

	# Apply display scale
	ts *= ds

	# Chroma Compression ------------------------------------------ *
	#   Here we set up the chroma compression function. For additional control, we use a different function
	#   than the tonescale for chroma compression: https://www.desmos.com/calculator/opsapnysjk
	#   It's the same type of hyperbolic compression, but with only a single scale, and processed in the inverse direction.
	#   if s0 is the scale, in the forward direction it would be y = s0*x/(s0*x+1)
	#   In the inverse direction it would be y = 1 - s0*x/(s0*x+1) = 1/(s0*x+1)
	#
	#   We also do not use a constant for s0. Instead we calculate a value of s0 per pixel from the RGB & CMY Dechroma weights.
	#   This gives us control over the chroma compression per hue angle for primaries and secondaries. Previously we had
	#   6 controls for dechroma amount for each hue angle, but in practice this seemed excessive. Now there are 2, one
	#   for primaries (RGB) and one for secondaries (CMY).
	#
	#   dch is the global dechroma user parameter. We also adjust dch by the tonescale scene-linear input scale parameter (s), 
	#   so that we compress chroma less when peak luminance is higher.
	
	dch_s = dch/s

	# Chroma compression factor, used to lerp towards 1 in rgb ratios, compressing chroma
	ccf = sdivf(1.0, lum*dch_s + 1.0)

	# Shadow Chroma Compression
	# We compress chroma in the toe as well, to avoid rainbow shadow grain and over-saturated shadows.
	# Adjusted by the dch_toe user parameter. Uses the same function as the flare compensation,
	# but constrained at (1, 1): https://www.desmos.com/calculator/igma4xoqlx
	# Here the full function would be y = x*x/(x + dch_toe)/x 
	# The last /x gives us the change of the function so that we can multiply this into our existing ccf.
	# To simplify, we remove the duplicate multiplications: y = x/(x+dch_toe)

	toe_ccf = (dch_toe + 1.0)*sdivf(lum, lum + dch_toe)*ccf

	# Chroma Compression Hue Shift ------------------------------------------ *
	#   Since we compress chroma by lerping in a straight line towards 1.0 in rgb ratios, this can result in perceptual hue shifts
	#   due to the Abney effect. For example, pure blue compressed in a straight line towards achromatic appears to shift in hue towards purple.
	#
	#   To combat this, and to add another import user control for image appearance, we add controls to curve the hue paths 
	#   as they move towards achromatic. We include only controls for primary colors: RGB. In my testing, it was of limited use to 
	#   control hue paths for secondary colors.
	#
	#   To accomplish this, we use the inverse of the chroma compression factor multiplied by the RGB hue angles as a factor
	#   for a lerp between the various rgb components.

	#   We don't include the toe chroma compression for this hue shift. It is mostly important for highlights.

	hs_w = (1.0 - ccf)*h_rgb

	# Apply hue shift to RGB Ratios
	rats = vec3(rats.x + hs_w.z*hs_b - hs_w.y*hs_g, rats.y + hs_w.x*hs_r - hs_w.z*hs_b, rats.z + hs_w.y*hs_g - hs_w.x*hs_r)

	# Apply chroma compression to RGB Ratios
	rats = 1.0 - toe_ccf + rats * toe_ccf

	# Clamp negative pixels
	rats = maxf3(0.0, rats)

	# Chroma Value Compression ------------------------------------------ *
	#   Using a weighted sum of RGB for our norm results in RGB ratios that are greater than 1.0.
	#   We compress chroma of the input towards peak achromatic, but there can still be values which extend beyond the display cube.
	#   These extensions can cause ugly halos and abrupt gradient changes, which are visually undesirable. To combat this, we 
	#   add a post-chroma compression normalization, which reduces the intensity of of saturated lights.
	#
	#   To achieve this, we first normalize the RGB ratios so that the max value is 1.0. We do this by taking the max of rgb, 
	#   then dividing the rgb ratios by it. Using this directly would result in midtones being way too dark, so we mix by 
	#   the tonescale * chroma, biased by a user-controlled power function (v_p) to adjust the intensity.

	# Post-chroma compression min and max
	rats_mx = max(rats.x, max(rats.y, rats.z))
	rats_mn = min(rats.x, min(rats.y, rats.z))

	# Chroma (normalized distance from achromatic) of post chroma compression RGB Ratios
	rats_ch = sdivf(rats_mx - rats_mn, rats_mx)

	# Normalization mix factor based on lum, with user adjustable strength (v_p)
	chf = spowf(rats_ch*ts, v_p)

	# Normalized rgb ratios
	rats_n = sdivf3f(rats, rats_mx)

	# Mix based on chf
	rats = rats_n*chf + rats*(1.0 - chf)

	# Apply tonescale to RGB Ratios
	rgb = rats*ts

	# Clamp to appropriate maximum
	rgb = minf3(clamp_max, rgb)


	# Apply inverse Display EOTF
	eotf_p = 2.0 + eotf * 0.2
	if (eotf > 0) and (eotf < 4):
		rgb = spowf3(rgb, 1.0/eotf_p)
	elif eotf == 4:
		rgb = eotf_pq(rgb, 1)
	elif eotf == 5:
		rgb = eotf_hlg(rgb, 1)

	return rgb