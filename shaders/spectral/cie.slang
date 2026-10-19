// CIE 1931 2-degree observer: wavelength sampling, colour matching, and XYZ to
// linear Rec.709.
//
// The `cie` buffer is LUT/CIE.dat uploaded verbatim: 2 x 441 x 3 f32.
//   cie[3*i + c]         CDF of channel c (x, y, z) at 390 + i nm, 0 at i = 0, 1 at i = 440
//   cie[3*(441 + i) + c] colour matching function of channel c at 390 + i nm

module cie;

import common.constants;

// The original divided by 0.5 * sum(cmf) instead of the normalized pdf, which
// scaled its estimates by this many stops at every wavelength alike
// (python/goldens.py original_spectral_scale).
public static const float ORIGINAL_ESTIMATOR_STOPS = -7.406;

public struct WavelengthSample
{
    public float wavelength;
    public float3 cmf;
    // 1 / pdf in nanometres; the XYZ estimate is L(wavelength) * cmf * rcp_pdf.
    public float rcp_pdf;
};

// Row 0 holds the CDFs, row 1 the colour matching functions.
float3 cie_entry(StructuredBuffer<float> cie, uint row, uint i)
{
    uint base = 3 * (row * WAVELENGTH_COUNT + i);
    return float3(cie[base], cie[base + 1], cie[base + 2]);
}

// Colour matching functions linearly interpolated between 1 nm entries,
// clamped to [390, 830] nm.
public float3 cie_cmf(StructuredBuffer<float> cie, float wavelength)
{
    WavelengthTablePosition position = WavelengthTablePosition(wavelength);
    return lerp(cie_entry(cie, 1, position.index), cie_entry(cie, 1, position.index + 1), position.weight);
}

// Exact inversion of the piecewise-linear channel-averaged CDF, so the pdf is
// its slope and integrates to one over [390, 830] nm. The search runs on
// channel sums against 3u: the pdf is then a difference of sums loaded from
// the table, which no compiler can contract differently in the flat red tail.
public WavelengthSample sample_wavelength(StructuredBuffer<float> cie, float u)
{
    const float3 ones = float3(1.0);
    float target = u * 3.0;
    // Largest segment start with a CDF sum <= target; 2^9 > 440 segments.
    uint lo = 0;
    uint hi = WAVELENGTH_COUNT - 1;
    for (uint iteration = 0; iteration < 9; iteration++)
    {
        uint mid = (lo + hi) / 2;
        if (dot(cie_entry(cie, 0, mid), ones) <= target)
            lo = mid;
        else
            hi = mid;
    }
    uint i = min(lo, WAVELENGTH_COUNT - 2);
    float sum0 = dot(cie_entry(cie, 0, i), ones);
    float rise = dot(cie_entry(cie, 0, i + 1), ones) - sum0;

    WavelengthSample s;
    s.wavelength = WAVELENGTH_MIN + float(i) + (rise > 0.0 ? saturate((target - sum0) / rise) : 0.0);
    s.cmf = cie_cmf(cie, s.wavelength);
    s.rcp_pdf = rise > 0.0 ? 3.0 / rise : 0.0;
    return s;
}

// Wavelengths that one sample carries together, drawn from one uniform number u:
// member k inverts the CDF at (u + k) / BUNDLE_WAVELENGTH_COUNT. Each member is
// importance sampled within its stratum of the CDF. That removes most of the
// chromatic noise of a single wavelength, as Wilkie et al. 2014's hero wavelengths
// do, and leaves the bundle symmetric: any member could have come first.
public static const int BUNDLE_WAVELENGTH_COUNT = 4;
// A value per bundle member.
public typealias Spectrum = vector<float, BUNDLE_WAVELENGTH_COUNT>;

public struct WavelengthBundle
{
    // The uniform number in [0, 1] that places every member.
    public float u;
    public Spectrum wavelength;
    // Column k is member k's colour matching functions over its pdf and the member
    // count, so mul(xyz_weight, radiance) estimates XYZ.
    public matrix<float, 3, BUNDLE_WAVELENGTH_COUNT> xyz_weight;
};

public WavelengthBundle sample_wavelength_bundle(StructuredBuffer<float> cie, float u)
{
    WavelengthBundle bundle;
    bundle.u = u;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
    {
        WavelengthSample s = sample_wavelength(cie, (u + float(k)) / float(BUNDLE_WAVELENGTH_COUNT));
        bundle.wavelength[k] = s.wavelength;
        float3 weight = s.cmf * s.rcp_pdf / float(BUNDLE_WAVELENGTH_COUNT);
        bundle.xyz_weight[0][k] = weight.x;
        bundle.xyz_weight[1][k] = weight.y;
        bundle.xyz_weight[2][k] = weight.z;
    }
    return bundle;
}

public float3 xyz_to_rec709(float3 xyz)
{
    return mul(float3x3(
        3.2409699419, -1.5373831776, -0.4986107603,
        -0.9692436363, 1.8759675015, 0.0415550574,
        0.0556300797, -0.2039769589, 1.0569715142), xyz);
}
