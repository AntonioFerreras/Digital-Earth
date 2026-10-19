"""OpenDRT v1.0.0b43 by Jed Smith (GPLv3), https://github.com/jedypod/open-display-transform

A float64 numpy port of the upstream DCTL at commit c8d698f
(display-transforms/opendrt/OpenDRT.dctl), the reference that
tests/goldens/post.json holds for shaders/post/opendrt.slang. It is written
from the DCTL, not from the Slang port, so the two check each other.

Specialized as the renderer uses it: linear Rec.709 in, 100 nit peak, D65
creative white, clamp on, linear Rec.709 display out. The look is the
DCTL's "Default" preset with three Hue Shift RGB values lowered
(`DEFAULT_LOOK`); the tonescale is one of `TONESCALE_PRESETS`. Two controls
sit on top of upstream: `saturation` scales Oklab chroma before the
transform, and `contrast` offsets the tonescale preset's contrast power.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
"""

import numpy as np

PEAK_LUMINANCE = 100.0
# The tonescale constraint solve raises to -1/contrast.
CONTRAST_MIN = 0.25

DEFAULT_LOOK = {
    "tn_gb": 0.13,
    "rs_sa": 0.25, "rs_rw": 0.25, "rs_bw": 0.4,
    "pt_r": 0.3, "pt_g": 1.0, "pt_b": 2.5, "pt_rng_low": 0.2, "pt_rng_high": 1.0,
    "ptm_low": 0.35, "ptm_low_st": 0.5, "ptm_high": -0.8, "ptm_high_st": 0.66,
    "brl_r": -0.4, "brl_g": -0.4, "brl_b": -0.1, "brl_c": 0.0, "brl_m": -0.15, "brl_y": 0.0, "brl_rng": 0.75,
    # Upstream: 0.35 each.
    "hs_r": 0.15, "hs_g": 0.2, "hs_b": 0.2,
    "hs_c": 0.2, "hs_m": 0.2, "hs_y": 0.2,
    "hc_r": 0.5,
}

# Upstream order. A module the preset disables has amount 0 here, the identity
# of both Contrast High and Contrast Low (upstream ACES-2.0 keeps 1.0 behind
# its disabled switches).
TONESCALE_PRESETS = {
    "high_contrast":       {"Lg": 11.1, "con": 1.4, "sh": 0.5, "toe": 0.001, "off": 0.005, "hcon": 0.0, "hcon_pv": 1.0, "hcon_st": 4.0, "lcon": 1.0, "lcon_w": 0.5, "lcon_pc": 1.0},
    "low_contrast":        {"Lg": 11.1, "con": 1.4, "sh": 0.5, "toe": 0.001, "off": 0.005, "hcon": 0.0, "hcon_pv": 1.0, "hcon_st": 4.0, "lcon": 0.0, "lcon_w": 0.5, "lcon_pc": 1.0},
    "aces_1":              {"Lg": 10.0, "con": 1.0, "sh": 0.245, "toe": 0.02, "off": 0.0, "hcon": 0.55, "hcon_pv": 0.0, "hcon_st": 2.0, "lcon": 1.13, "lcon_w": 1.0, "lcon_pc": 1.0},
    "aces_2":              {"Lg": 10.0, "con": 1.15, "sh": 0.5, "toe": 0.04, "off": 0.0, "hcon": 0.0, "hcon_pv": 1.0, "hcon_st": 1.0, "lcon": 0.0, "lcon_w": 0.6, "lcon_pc": 1.0},
    "marvelous_tonescape": {"Lg": 6.0, "con": 1.5, "sh": 0.5, "toe": 0.003, "off": 0.01, "hcon": 0.25, "hcon_pv": 0.0, "hcon_st": 4.0, "lcon": 1.0, "lcon_w": 1.0, "lcon_pc": 1.0},
    "arriba_tonecall":     {"Lg": 11.1, "con": 1.05, "sh": 0.5, "toe": 0.1, "off": 0.015, "hcon": 0.0, "hcon_pv": 0.0, "hcon_st": 2.0, "lcon": 2.0, "lcon_w": 0.2, "lcon_pc": 1.0},
    "dagrinchi_tone":      {"Lg": 10.42, "con": 1.2, "sh": 0.5, "toe": 0.02, "off": 0.0, "hcon": 0.0, "hcon_pv": 1.0, "hcon_st": 1.0, "lcon": 0.0, "lcon_w": 0.6, "lcon_pc": 1.0},
}

REC709_TO_XYZ = np.array([[0.412390917540, 0.357584357262, 0.180480793118],
                          [0.212639078498, 0.715168714523, 0.072192311287],
                          [0.019330825657, 0.119194783270, 0.950532138348]])
XYZ_TO_P3D65 = np.array([[2.49349691194, -0.931383617919, -0.402710784451],
                         [-0.829488969562, 1.76266406032, 0.023624685842],
                         [0.035845830244, -0.076172389268, 0.956884524008]])
P3_TO_REC709 = np.array([[1.224940181, -0.2249402404, 0.0],
                         [-0.04205697775, 1.042057037, -1.4901e-08],
                         [-0.01963755488, -0.07863604277, 1.098273635]])
# Oklab (Ottosson 2020), from XYZ.
XYZ_TO_LMS = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                       [0.0329845436, 0.9293118715, 0.0361456387],
                       [0.0482003018, 0.2643662691, 0.6338517070]])
LMS_TO_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                         [1.9779984951, -2.4285922050, 0.4505937099],
                         [0.0259040371, 0.7827717662, -0.8086757660]])


def spow(a, b):
    return np.where(a <= 0.0, a, np.power(np.maximum(a, 1e-300), b))


def sdiv(a, b):
    return np.where(b == 0.0, 0.0, a / np.where(b == 0.0, 1.0, b))


def compress_toe_quadratic(x, toe, inverse):
    if toe == 0.0:
        return x
    if inverse:
        return (x + np.sqrt(x * (4.0 * toe + x))) / 2.0
    return spow(x, 2.0) / (x + toe)


def compress_toe_cubic(x, m, w, inverse):
    if m == 1.0:
        return x
    x2 = x * x
    if not inverse:
        return x * (x2 + m * w) / (x2 + w)
    p0 = x2 - 3.0 * m * w
    p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w
    p2 = np.power(np.sqrt(x2 * p1 * p1 - 4.0 * p0 * p0 * p0) / 2.0 + x * p1 / 2.0, 1.0 / 3.0)
    return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0


def sigmoid_cubic_power(x, s, p):
    inside = (x >= 0.0) & (x <= 1.0)
    xc = 1.0 - np.power(1.0 - np.clip(x, 0.0, 1.0), 1.0 / p)
    return np.where(inside, 1.0 + s * (1.0 - 3.0 * xc * xc + 2.0 * xc * xc * xc), 1.0)


def contrast_high(x, p, pv, pv_lx):
    x0 = 0.18 * 2.0 ** pv
    if p == 1.0:
        return x
    o = x0 - x0 / p
    s0 = x0 ** (1.0 - p) / p
    x1 = x0 * 2.0 ** pv_lx
    k1 = p * s0 * x1 ** p / x1
    y1 = s0 * x1 ** p + o
    upper = np.where(x > x1, k1 * (x - x1) + y1, s0 * np.power(np.maximum(x, 0.0), p) + o)
    return np.where(x < x0, x, upper)


def softplus(x, s, x0, y0):
    if s < 1e-3:
        return x
    m = np.exp(y0 / s) if abs(y0) > 1e-6 else 1.0
    m -= np.exp(x0 / s)
    with np.errstate(over="ignore"):
        y = s * np.log(np.maximum(0.0, m + np.exp(np.minimum(x, 10.0 * s + y0 + 1.0) / s)))
    return np.where(x > 10.0 * s + y0, x, y)


def gauss_window(x, w):
    x = x / w
    return np.exp(-x * x)


def hue_offset(h, o):
    return np.fmod(h - o + np.pi, 2.0 * np.pi) - np.pi


def oklab_chroma_scale(xyz, saturation):
    lab = np.cbrt(xyz @ XYZ_TO_LMS.T) @ LMS_TO_OKLAB.T
    lab[:, 1:] *= saturation
    return (np.linalg.solve(LMS_TO_OKLAB, lab.T).T ** 3) @ np.linalg.inv(XYZ_TO_LMS).T


def opendrt(rgb, saturation, contrast, tonescale):
    """Linear Rec.709 (n, 3) to display-linear Rec.709 in [0, 1]."""
    look = DEFAULT_LOOK
    ts = TONESCALE_PRESETS[tonescale]
    con = max(ts["con"] + contrast, CONTRAST_MIN)
    rgb = np.array(rgb, dtype=np.float64, ndmin=2)
    assert rgb.shape[1] == 3

    ts_x1 = 2.0 ** (6.0 * ts["sh"] + 4.0)
    ts_y1 = PEAK_LUMINANCE / 100.0
    ts_x0 = 0.18 + ts["off"]
    ts_y0 = ts["Lg"] / 100.0 * (1.0 + look["tn_gb"] * np.log2(ts_y1))
    ts_s0 = compress_toe_quadratic(ts_y0, ts["toe"], True)
    ts_s10 = ts_x0 * (ts_s0 ** (-1.0 / con) - 1.0)
    ts_m1 = ts_y1 / (ts_x1 / (ts_x1 + ts_s10)) ** con
    ts_m2 = compress_toe_quadratic(ts_m1, ts["toe"], True)
    ts_s = ts_x0 * ((ts_s0 / ts_m2) ** (-1.0 / con) - 1.0)
    ts_dsc = 100.0 / PEAK_LUMINANCE
    # Purity HDR blends towards ts_s above 100 nits; at 100 nits it is s_Lp100.
    ts_s1 = ts_x0 * ((ts["Lg"] / 100.0) ** (-1.0 / con) - 1.0)

    xyz = oklab_chroma_scale(rgb @ REC709_TO_XYZ.T, saturation)
    rgb = xyz @ XYZ_TO_P3D65.T

    rs_w = np.array([look["rs_rw"], 1.0 - look["rs_rw"] - look["rs_bw"], look["rs_bw"]])
    rgb = (rgb @ rs_w)[:, None] * look["rs_sa"] + rgb * (1.0 - look["rs_sa"])
    rgb = rgb + ts["off"]

    if ts["lcon"] != 0.0:
        mcon_m = 2.0 ** -ts["lcon"]
        mcon_w = (ts["lcon_w"] / 4.0) ** 2
        rgb = rgb * (compress_toe_cubic(ts_x0, mcon_m, mcon_w, True) / ts_x0)
        mcon_nm = np.linalg.norm(np.maximum(rgb, 0.0), axis=1) / np.sqrt(3.0)
        mcon_sc = (mcon_nm ** 2 + mcon_m * mcon_w) / (mcon_nm ** 2 + mcon_w)
        mcon_rgb = compress_toe_cubic(rgb, mcon_m, mcon_w, False)
        mcon_ch = np.clip(1.0 - sdiv(rgb.min(axis=1), rgb.max(axis=1)), 0.0, 1.0) ** (4.0 * ts["lcon_pc"])
        rgb = mcon_sc[:, None] * rgb * mcon_ch[:, None] + mcon_rgb * (1.0 - mcon_ch[:, None])

    tsn = np.linalg.norm(np.maximum(rgb, 0.0), axis=1) / np.sqrt(3.0)
    pt_w = np.array([look["pt_r"], look["pt_g"], look["pt_b"]])
    ts_pt = np.sqrt(np.maximum(0.0, (rgb * rgb) @ pt_w))
    rgb = sdiv(np.maximum(rgb, -2.0), tsn[:, None])

    if ts["hcon"] != 0.0:
        hcon_p = 2.0 ** ts["hcon"]
        tsn = contrast_high(tsn, hcon_p, ts["hcon_pv"], ts["hcon_st"])
        ts_pt = contrast_high(ts_pt, hcon_p, ts["hcon_pv"], ts["hcon_st"])

    tsn = spow(tsn / (tsn + ts_s), con)
    ts_pt = spow(ts_pt / (ts_pt + ts_s1), con)

    opp_cy = rgb[:, 0] - rgb[:, 2]
    opp_gm = rgb[:, 1] - (rgb[:, 0] + rgb[:, 2]) / 2.0
    ach_d = np.sqrt(np.maximum(0.0, opp_cy * opp_cy + opp_gm * opp_gm)) / np.sqrt(3.0)
    ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25, False)
    hue = np.fmod(np.arctan2(opp_cy, opp_gm) + np.pi + 1.10714931, 2.0 * np.pi)
    ha_rgb = np.stack([gauss_window(hue_offset(hue, o), 0.8) for o in (0.0, 4.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)], axis=1)
    ha_cmy = np.stack([gauss_window(hue_offset(hue, o), 0.6) for o in (np.pi, 0.9, -np.pi / 3.0)], axis=1)

    ts_pt_cmp = np.power(1.0 - np.power(ts_pt, 1.0 / look["pt_rng_low"]), look["pt_rng_high"])

    brl_f = -(ha_rgb @ np.array([look["brl_r"], look["brl_g"], look["brl_b"]])) \
        - (ha_cmy @ np.array([look["brl_c"], look["brl_m"], look["brl_y"]]))
    brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f
    brl_f = softplus(brl_f, 0.25, -100.0, 0.0)
    brl_lim = spow(ts_pt, 1.0 - look["brl_rng"])
    brl_f = np.clip(brl_f * brl_lim + 1.0 - brl_lim, 0.0, 2.0)

    ptm_sc = sigmoid_cubic_power(ach_d, look["ptm_low"] * (1.0 - ts_pt), look["ptm_low_st"])
    ptm_sc = ptm_sc * sigmoid_cubic_power(ach_d, look["ptm_high"] * ts_pt, look["ptm_high_st"])

    ha_rgb = ha_rgb * ach_d[:, None]
    ha_cmy = ha_cmy * (1.5 * compress_toe_quadratic(ach_d, 0.5, False))[:, None]

    hc_ts = 1.0 - ts_pt
    hc_c = ((1.0 - ach_d) * hc_ts + ach_d * (1.0 - hc_ts)) * ha_rgb[:, 0]
    hc_f = look["hc_r"] * (hc_c - 2.0 * hc_c * hc_ts * hc_ts) + 1.0
    rgb = rgb * np.stack([np.ones_like(hc_f), hc_f, hc_f], axis=1)

    def rotate(hsf):
        return np.stack([hsf[:, 2] - hsf[:, 1], hsf[:, 0] - hsf[:, 2], hsf[:, 1] - hsf[:, 0]], axis=1)

    hs_rgb = ha_rgb * (ts_pt * ts_pt)[:, None]
    rgb = rgb + rotate(hs_rgb * np.array([look["hs_r"], -look["hs_g"], -look["hs_b"]]))
    hs_cmy = ha_cmy * (1.0 - ts_pt)[:, None]
    rgb = rgb + rotate(hs_cmy * np.array([-look["hs_c"], look["hs_m"], look["hs_y"]]))

    rgb = rgb * brl_f[:, None]
    ts_pt_cmp = ts_pt_cmp * ptm_sc
    rgb = rgb * ts_pt_cmp[:, None] + 1.0 - ts_pt_cmp[:, None]

    sat_l = rgb @ rs_w
    rgb = (sat_l[:, None] * look["rs_sa"] - rgb) / (look["rs_sa"] - 1.0)
    rgb = rgb @ P3_TO_REC709.T

    sum0 = rgb.sum(axis=1)
    rgb = np.stack([softplus(rgb[:, 0], 0.04, -0.3, 0.0),
                    softplus(rgb[:, 1], 0.06, -0.3, 0.0),
                    softplus(rgb[:, 2], 0.01, -0.05, 0.0)], axis=1)
    ptl_norm = softplus(sdiv(sum0, rgb.sum(axis=1)), 0.094, -100.0, 0.5)
    rgb = rgb * ptl_norm[:, None]

    tsn = compress_toe_quadratic(tsn * ts_m2, ts["toe"], False) * ts_dsc
    return np.clip(rgb * tsn[:, None], 0.0, 1.0)
