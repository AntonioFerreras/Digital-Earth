import taichi as ti
from taichi.math import *
from lib.colour import *

#Shader implementation of Troy Sobotka's AgX, translated to shader originally by Olivier Groulx
#https://github.com/sobotka/AgX
#https://github.com/sobotka/SB2383-Configuration-Generation
#https://github.com/sobotka/SuazoBrejon2383-Configuration
#https://github.com/macrofacet/horoma

#AgX Settings
MIDDLE_GREY = 0.18 # Default = 0.18
SLOPE = 2.3 # Default = 2.3
TOE_POWER = 1.9 # Default = 1.9
SHOULDER_POWER = 3.1 # Default = 3.1
COMPRESSION = 0.15 # Default = 0.15

#Demo Settings
MIN_EV = -10.0
MAX_EV = 6.5
SATURATION = 1.4

@ti.func
def InverseMat(m: mat3):
    d = m[0, 0] * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);
              
    id = 1.0 / d
    
    c = mat3(1,0,0,0,1,0,0,0,1)
    
    c[0, 0] = id * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2])
    c[0, 1] = id * (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2])
    c[0, 2] = id * (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1])
    c[1, 0] = id * (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2])
    c[1, 1] = id * (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0])
    c[1, 2] = id * (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2])
    c[2, 0] = id * (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1])
    c[2, 1] = id * (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1])
    c[2, 2] = id * (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1])
    
    return c

@ti.func
def xyYToXYZ(xyY: vec3):
    X = 0.0
    Y = 0.0
    Z = 0.0
    if(xyY.y != 0.0):
        Y = xyY.z
        X = (xyY.x * Y) / xyY.y
        Z = ((1.0 - xyY.x - xyY.y) * Y) / xyY.y

    return vec3(X, Y, Z)

@ti.func
def Unproject(xy: vec2):
    return xyYToXYZ(vec3(xy.x, xy.y, 1))		

@ti.func
def PrimariesToMatrix(xy_red: vec2, xy_green: vec2, xy_blue: vec2, xy_white: vec2):
    XYZ_red = Unproject(xy_red)
    XYZ_green = Unproject(xy_green)
    XYZ_blue = Unproject(xy_blue)

    XYZ_white = Unproject(xy_white)

    temp = mat3(XYZ_red.x,	XYZ_green.x, XYZ_blue.x, 1.0, 1.0, 1.0, XYZ_red.z, XYZ_green.z, XYZ_blue.z)

    inverse = InverseMat(temp)
    scale = inverse @ XYZ_white

    return mat3(scale.x * XYZ_red.x, scale.y * XYZ_green.x,	scale.z * XYZ_blue.x,
                scale.x * XYZ_red.y, scale.y * XYZ_green.y,	scale.z * XYZ_blue.y,
                scale.x * XYZ_red.z, scale.y * XYZ_green.z,	scale.z * XYZ_blue.z)

@ti.func
def ComputeCompressionMatrix(xyR: vec2, xyG: vec2, xyB: vec2, xyW: vec2, compression: ti.f32):
    scale_factor = 1.0 / (1.0 - compression)
    R = ((xyR - xyW) * scale_factor) + xyW
    G = ((xyG - xyW) * scale_factor) + xyW
    B = ((xyB - xyW) * scale_factor) + xyW
    W = xyW

    return PrimariesToMatrix(R, G, B, W)


@ti.func
def OpenDomainToNormalizedLog2(openDomain: vec3, minimum_ev: ti.f32, maximum_ev: ti.f32):
    total_exposure = maximum_ev - minimum_ev

    output_log = clamp(log2(openDomain / MIDDLE_GREY), minimum_ev, maximum_ev)

    return (output_log - minimum_ev) / total_exposure


@ti.func
def AgXScale(x_pivot: ti.f32, y_pivot: ti.f32, slope_pivot: ti.f32, power: ti.f32):
    return pow(pow((slope_pivot * x_pivot), -power) * (pow((slope_pivot * (x_pivot / y_pivot)), power) - 1.0), -1.0 / power)

@ti.func
def AgXHyperbolic(x: ti.f32, power: ti.f32):
    return x / pow(1.0 + pow(x, power), 1.0 / power)

@ti.func
def AgXTerm(x: ti.f32, x_pivot: ti.f32, slope_pivot: ti.f32, scale: ti.f32):
    return (slope_pivot * (x - x_pivot)) / scale

@ti.func
def AgXCurve(x: ti.f32, x_pivot: ti.f32, y_pivot: ti.f32, slope_pivot: ti.f32, toe_power: ti.f32, shoulder_power: ti.f32, scale: ti.f32):
    curve = 0.0
    if(scale < 0.0):
        curve = scale * AgXHyperbolic(AgXTerm(x, x_pivot, slope_pivot, scale), toe_power) + y_pivot
    else:
        curve = scale * AgXHyperbolic(AgXTerm(x,x_pivot, slope_pivot,scale), shoulder_power) + y_pivot
    return curve

@ti.func
def AgXFullCurve(x: ti.f32, x_pivot: ti.f32, y_pivot: ti.f32, slope_pivot: ti.f32, toe_power: ti.f32, shoulder_power: ti.f32):
    scale_x_pivot = 1.0 - x_pivot if x >= x_pivot else x_pivot
    scale_y_pivot = 1.0 - y_pivot if x >= x_pivot else y_pivot

    toe_scale = AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, toe_power)
    shoulder_scale = AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, shoulder_power)			

    scale = shoulder_scale if x >= x_pivot else -toe_scale

    return AgXCurve(x, x_pivot, y_pivot, slope_pivot, toe_power, shoulder_power, scale)

# Takes in HDR linear Rec.709/sRGB and returns LDR
@ti.func
def display_transform(workingColor: vec3):
    sRGB_to_XYZ = PrimariesToMatrix(vec2(0.64,0.33),
                                         vec2(0.3,0.6), 
                                         vec2(0.15,0.06), 
                                         vec2(0.3127, 0.3290))

    adjusted_to_XYZ = ComputeCompressionMatrix(vec2(0.64,0.33),
                                                    vec2(0.3,0.6), 
                                                    vec2(0.15,0.06), 
                                                    vec2(0.3127, 0.3290), COMPRESSION)

    								
    XYZ_to_adjusted = InverseMat(adjusted_to_XYZ)

    xyz = sRGB_to_XYZ @ workingColor
    adjustedRGB = XYZ_to_adjusted @ xyz

    x_pivot = abs(MIN_EV) / (MAX_EV - MIN_EV)
    y_pivot = 0.5

    logV = OpenDomainToNormalizedLog2(adjustedRGB, MIN_EV, MAX_EV)

    outputR = AgXFullCurve(logV.r, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER)
    outputG = AgXFullCurve(logV.g, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER)
    outputB = AgXFullCurve(logV.b, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER)

    workingColor = clamp(vec3(outputR, outputG, outputB), 0.0, 1.0)

    workingColor = mix(lum3(workingColor), workingColor, SATURATION)
    return clamp(workingColor, 0., 1.)