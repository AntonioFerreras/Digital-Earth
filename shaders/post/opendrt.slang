// OpenDRT v1.0.0b43 by Jed Smith (GPL v3), https://github.com/jedypod/open-display-transform
// Ported from display-transforms/opendrt/OpenDRT.dctl at commit c8d698f.
// Forward transform only, specialized to linear Rec.709 in, 100 nit peak, D65
// creative white, clamp on, and linear Rec.709 display out. python/opendrt.py
// is the float64 reference of the same specialization.

module opendrt;

static const float PEAK_LUMINANCE = 100.0;
// The tonescale constraint solve raises to -1/contrast.
static const float CONTRAST_MIN = 0.25;
static const float PI = 3.14159265358979323846;
static const float SQRT3 = 1.73205080756887729353;

// The "Default" look preset, with Hue Shift RGB lowered from upstream's 0.35.
static const float GREY_BOOST = 0.13;
static const float RENDER_SPACE_STRENGTH = 0.25;
static const float3 RENDER_SPACE_WEIGHTS = float3(0.25, 1.0 - 0.25 - 0.4, 0.4);
static const float3 PURITY_COMPRESS = float3(0.3, 1.0, 2.5);
static const float PURITY_RANGE_LOW = 0.2;
static const float PURITY_RANGE_HIGH = 1.0;
static const float MID_PURITY_LOW = 0.35;
static const float MID_PURITY_LOW_STRENGTH = 0.5;
static const float MID_PURITY_HIGH = -0.8;
static const float MID_PURITY_HIGH_STRENGTH = 0.66;
static const float3 BRILLIANCE_RGB = float3(-0.4, -0.4, -0.1);
static const float3 BRILLIANCE_CMY = float3(0.0, -0.15, 0.0);
static const float BRILLIANCE_RANGE = 0.75;
static const float3 HUE_SHIFT_RGB = float3(0.15, 0.2, 0.2);
static const float3 HUE_SHIFT_CMY = float3(0.2, 0.2, 0.2);
static const float HUE_CONTRAST_R = 0.5;

struct Tonescale
{
    float grey_luminance; // nits of scene-linear 0.18
    float contrast;
    float shoulder_clip;
    float toe;
    float offset;
    float high_contrast;
    float high_pivot;
    float high_strength;
    float low_contrast;
    float low_width;
    float low_per_channel;
};

// Upstream order, mirrored by `Post.TonescalePreset`. A module the preset
// disables has amount 0 here, the identity of Contrast High and Contrast Low.
static const Tonescale TONESCALE_PRESETS[7] = {
    { 11.1, 1.4, 0.5, 0.001, 0.005, 0.0, 1.0, 4.0, 1.0, 0.5, 1.0 },   // High-Contrast
    { 11.1, 1.4, 0.5, 0.001, 0.005, 0.0, 1.0, 4.0, 0.0, 0.5, 1.0 },   // Low-Contrast
    { 10.0, 1.0, 0.245, 0.02, 0.0, 0.55, 0.0, 2.0, 1.13, 1.0, 1.0 },  // ACES-1.x
    { 10.0, 1.15, 0.5, 0.04, 0.0, 0.0, 1.0, 1.0, 0.0, 0.6, 1.0 },     // ACES-2.0
    { 6.0, 1.5, 0.5, 0.003, 0.01, 0.25, 0.0, 4.0, 1.0, 1.0, 1.0 },    // Marvelous Tonescape
    { 11.1, 1.05, 0.5, 0.1, 0.015, 0.0, 0.0, 2.0, 2.0, 0.2, 1.0 },    // Arriba Tonecall
    { 10.42, 1.2, 0.5, 0.02, 0.0, 0.0, 1.0, 1.0, 0.0, 0.6, 1.0 },     // DaGrinchi Tone
};

static const float3x3 REC709_TO_XYZ = float3x3(
    0.412390917540, 0.357584357262, 0.180480793118,
    0.212639078498, 0.715168714523, 0.072192311287,
    0.019330825657, 0.119194783270, 0.950532138348);
static const float3x3 XYZ_TO_P3D65 = float3x3(
    2.49349691194, -0.931383617919, -0.402710784451,
    -0.829488969562, 1.76266406032, 0.023624685842,
    0.035845830244, -0.076172389268, 0.956884524008);
static const float3x3 P3_TO_REC709 = float3x3(
    1.224940181, -0.2249402404, 0.0,
    -0.04205697775, 1.042057037, -1.4901e-08,
    -0.01963755488, -0.07863604277, 1.098273635);
// Oklab (Ottosson 2020) from XYZ; the inverses are exact float64 inverses.
static const float3x3 XYZ_TO_LMS = float3x3(
    0.8189330101, 0.3618667424, -0.1288597137,
    0.0329845436, 0.9293118715, 0.0361456387,
    0.0482003018, 0.2643662691, 0.6338517070);
static const float3x3 LMS_TO_OKLAB = float3x3(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660);
static const float3x3 LMS_TO_XYZ = float3x3(
    1.2270138511, -0.557799980652, 0.281256148966,
    -0.0405801784233, 1.11225686962, -0.0716766786656,
    -0.0763812845057, -0.421481978418, 1.58616322044);
static const float3x3 OKLAB_TO_LMS = float3x3(
    0.999999998451, 0.396337792174, 0.215803758061,
    1.00000000888, -0.105561342324, -0.0638541747717,
    1.00000005467, -0.089484182095, -1.29148553786);

// Division that yields 0 for a zero divisor.
float safe_div(float a, float b)
{
    return b == 0.0 ? 0.0 : a / b;
}

// Power that passes non-positive bases through unchanged.
float safe_pow(float a, float b)
{
    return a <= 0.0 ? a : pow(a, b);
}

float compress_toe_quadratic(float x, float toe)
{
    return toe == 0.0 ? x : safe_pow(x, 2.0) / (x + toe);
}

float compress_toe_quadratic_inverse(float x, float toe)
{
    return toe == 0.0 ? x : (x + sqrt(x * (4.0 * toe + x))) / 2.0;
}

float compress_toe_cubic(float x, float m, float w)
{
    float x2 = x * x;
    return x * (x2 + m * w) / (x2 + w);
}

float compress_toe_cubic_inverse(float x, float m, float w)
{
    float x2 = x * x;
    float p0 = x2 - 3.0 * m * w;
    float p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w;
    float p2 = pow(sqrt(x2 * p1 * p1 - 4.0 * p0 * p0 * p0) / 2.0 + x * p1 / 2.0, 1.0 / 3.0);
    return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0;
}

float sigmoid_cubic_power(float x, float s, float p)
{
    if (x < 0.0 || x > 1.0)
        return 1.0;
    x = 1.0 - pow(1.0 - x, 1.0 / p);
    return 1.0 + s * (1.0 - 3.0 * x * x + 2.0 * x * x * x);
}

// Highlight exposure power above `pivot` stops over middle grey, continued
// linearly from `strength` stops above the pivot.
float contrast_high(float x, float p, float pivot, float strength)
{
    float x0 = 0.18 * exp2(pivot);
    if (x < x0)
        return x;
    float o = x0 - x0 / p;
    float s0 = pow(x0, 1.0 - p) / p;
    float x1 = x0 * exp2(strength);
    float k1 = p * s0 * pow(x1, p) / x1;
    float y1 = s0 * pow(x1, p) + o;
    return x > x1 ? k1 * (x - x1) + y1 : s0 * pow(x, p) + o;
}

// Softplus through (x0, y0); the identity far above y0.
float softplus(float x, float s, float x0, float y0)
{
    if (x > 10.0 * s + y0)
        return x;
    float m = abs(y0) > 1e-6 ? exp(y0 / s) : 1.0;
    m -= exp(x0 / s);
    return s * log(max(0.0, m + exp(x / s)));
}

float gauss_window(float x, float w)
{
    x /= w;
    return exp(-x * x);
}

float hue_offset(float h, float o)
{
    return fmod(h - o + PI, 2.0 * PI) - PI;
}

float3 cbrt(float3 x)
{
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

// Scene-linear Rec.709 to display-linear Rec.709 in [0, 1]. `saturation`
// scales Oklab chroma before the transform (1 keeps upstream's look);
// `contrast` offsets the tonescale preset's contrast power (0 keeps it).
public float3 opendrt(float3 rgb, float saturation, float contrast, uint tonescale_preset)
{
    Tonescale ts = TONESCALE_PRESETS[tonescale_preset];
    float con = max(ts.contrast + contrast, CONTRAST_MIN);

    // Constrains grey to `grey_luminance` and the shoulder clip to the peak.
    float ts_x1 = exp2(6.0 * ts.shoulder_clip + 4.0);
    float ts_y1 = PEAK_LUMINANCE / 100.0;
    float ts_x0 = 0.18 + ts.offset;
    float ts_y0 = ts.grey_luminance / 100.0 * (1.0 + GREY_BOOST * log2(ts_y1));
    float ts_s0 = compress_toe_quadratic_inverse(ts_y0, ts.toe);
    float ts_s10 = ts_x0 * (pow(ts_s0, -1.0 / con) - 1.0);
    float ts_m1 = ts_y1 / pow(ts_x1 / (ts_x1 + ts_s10), con);
    float ts_m2 = compress_toe_quadratic_inverse(ts_m1, ts.toe);
    float ts_s = ts_x0 * (pow(ts_s0 / ts_m2, -1.0 / con) - 1.0);
    // Purity compression's scale; Purity HDR would move it only above 100 nits.
    float ts_s1 = ts_x0 * (pow(ts.grey_luminance / 100.0, -1.0 / con) - 1.0);

    float3 lab = mul(LMS_TO_OKLAB, cbrt(mul(XYZ_TO_LMS, mul(REC709_TO_XYZ, rgb))));
    lab.yz *= saturation;
    float3 lms = mul(OKLAB_TO_LMS, lab);
    rgb = mul(XYZ_TO_P3D65, mul(LMS_TO_XYZ, lms * lms * lms));

    // Render space: ratios are taken in a gamut desaturated from P3.
    float sat_l = dot(rgb, RENDER_SPACE_WEIGHTS);
    rgb = sat_l * RENDER_SPACE_STRENGTH + rgb * (1.0 - RENDER_SPACE_STRENGTH);
    rgb += ts.offset;

    if (ts.low_contrast != 0.0)
    {
        float mcon_m = exp2(-ts.low_contrast);
        float mcon_w = ts.low_width / 4.0;
        mcon_w *= mcon_w;
        // Keeps middle grey in place.
        rgb *= compress_toe_cubic_inverse(ts_x0, mcon_m, mcon_w) / ts_x0;
        float mcon_nm = length(max(rgb, 0.0)) / SQRT3;
        float mcon_sc = (mcon_nm * mcon_nm + mcon_m * mcon_w) / (mcon_nm * mcon_nm + mcon_w);
        float3 mcon_rgb = float3(compress_toe_cubic(rgb.x, mcon_m, mcon_w),
                                 compress_toe_cubic(rgb.y, mcon_m, mcon_w),
                                 compress_toe_cubic(rgb.z, mcon_m, mcon_w));
        // Per-channel near achromatic, ratio-preserving towards the gamut boundary.
        float mx = max(rgb.x, max(rgb.y, rgb.z));
        float mn = min(rgb.x, min(rgb.y, rgb.z));
        float mcon_ch = pow(clamp(1.0 - safe_div(mn, mx), 0.0, 1.0), 4.0 * ts.low_per_channel);
        rgb = mcon_sc * rgb * mcon_ch + mcon_rgb * (1.0 - mcon_ch);
    }

    float tsn = length(max(rgb, 0.0)) / SQRT3;
    float ts_pt = sqrt(max(0.0, dot(rgb * rgb, PURITY_COMPRESS)));
    float3 floored = max(rgb, -2.0);
    rgb = float3(safe_div(floored.x, tsn), safe_div(floored.y, tsn), safe_div(floored.z, tsn));

    if (ts.high_contrast != 0.0)
    {
        float hcon_p = exp2(ts.high_contrast);
        tsn = contrast_high(tsn, hcon_p, ts.high_pivot, ts.high_strength);
        ts_pt = contrast_high(ts_pt, hcon_p, ts.high_pivot, ts.high_strength);
    }

    tsn = safe_pow(tsn / (tsn + ts_s), con);
    ts_pt = safe_pow(ts_pt / (ts_pt + ts_s1), con);

    // Opponent space for a smooth achromatic distance and hue angle (red at 0).
    float opp_cy = rgb.x - rgb.z;
    float opp_gm = rgb.y - (rgb.x + rgb.z) / 2.0;
    float ach_d = sqrt(max(0.0, opp_cy * opp_cy + opp_gm * opp_gm)) / SQRT3;
    ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25);
    float hue = fmod(atan2(opp_cy, opp_gm) + PI + 1.10714931, 2.0 * PI);
    float3 ha_rgb = float3(gauss_window(hue_offset(hue, 0.0), 0.8),
                           gauss_window(hue_offset(hue, 4.0 * PI / 3.0), 0.8),
                           gauss_window(hue_offset(hue, 2.0 * PI / 3.0), 0.8));
    // Magenta sits at 0.9 rather than PI/3, rotated towards red.
    float3 ha_cmy = float3(gauss_window(hue_offset(hue, PI), 0.6),
                           gauss_window(hue_offset(hue, 0.9), 0.6),
                           gauss_window(hue_offset(hue, -PI / 3.0), 0.6));

    float ts_pt_cmp = pow(1.0 - pow(ts_pt, 1.0 / PURITY_RANGE_LOW), PURITY_RANGE_HIGH);

    float brl_f = -dot(ha_rgb, BRILLIANCE_RGB) - dot(ha_cmy, BRILLIANCE_CMY);
    brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f;
    brl_f = softplus(brl_f, 0.25, -100.0, 0.0);
    float brl_lim = safe_pow(ts_pt, 1.0 - BRILLIANCE_RANGE);
    brl_f = clamp(brl_f * brl_lim + 1.0 - brl_lim, 0.0, 2.0);

    float ptm_sc = sigmoid_cubic_power(ach_d, MID_PURITY_LOW * (1.0 - ts_pt), MID_PURITY_LOW_STRENGTH)
                 * sigmoid_cubic_power(ach_d, MID_PURITY_HIGH * ts_pt, MID_PURITY_HIGH_STRENGTH);

    ha_rgb *= ach_d;
    ha_cmy *= 1.5 * compress_toe_quadratic(ach_d, 0.5);

    float hc_ts = 1.0 - ts_pt;
    float hc_c = ((1.0 - ach_d) * hc_ts + ach_d * (1.0 - hc_ts)) * ha_rgb.x;
    rgb.yz *= HUE_CONTRAST_R * (hc_c - 2.0 * hc_c * hc_ts * hc_ts) + 1.0;

    float3 hs = ha_rgb * (ts_pt * ts_pt) * HUE_SHIFT_RGB * float3(1.0, -1.0, -1.0);
    rgb += float3(hs.z - hs.y, hs.x - hs.z, hs.y - hs.x);
    hs = ha_cmy * (1.0 - ts_pt) * HUE_SHIFT_CMY * float3(-1.0, 1.0, 1.0);
    rgb += float3(hs.z - hs.y, hs.x - hs.z, hs.y - hs.x);

    rgb *= brl_f;
    ts_pt_cmp *= ptm_sc;
    rgb = rgb * ts_pt_cmp + 1.0 - ts_pt_cmp;

    sat_l = dot(rgb, RENDER_SPACE_WEIGHTS);
    rgb = (sat_l * RENDER_SPACE_STRENGTH - rgb) / (RENDER_SPACE_STRENGTH - 1.0);
    rgb = mul(P3_TO_REC709, rgb);

    // Purity Compress Low.
    float sum0 = rgb.x + rgb.y + rgb.z;
    rgb = float3(softplus(rgb.x, 0.04, -0.3, 0.0), softplus(rgb.y, 0.06, -0.3, 0.0), softplus(rgb.z, 0.01, -0.05, 0.0));
    rgb *= softplus(safe_div(sum0, rgb.x + rgb.y + rgb.z), 0.094, -100.0, 0.5);

    tsn = compress_toe_quadratic(tsn * ts_m2, ts.toe) * (100.0 / PEAK_LUMINANCE);
    return clamp(rgb * tsn, 0.0, 1.0);
}
