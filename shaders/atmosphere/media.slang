// Rayleigh, aerosol (Mie), and ozone media: altitude profiles, spectral
// extinction, and single-scattering albedos. `h` is altitude above the
// planet radius in metres, `wavelength` is in nanometres.

module media;

import common.constants;
import spectral.cie;

static const float AIR_NUMBER_DENSITY = 2.5035422e25;
// LUT/ozone_profile.dat: number density (m^-3) at altitudes 0, 500, ..., 110000 m.
static const float OZONE_PROFILE_STEP = 500.0;
static const uint OZONE_PROFILE_COUNT = 221;

public static const float RAYLEIGH_ALBEDO = 1.0;
public static const float AEROSOL_ALBEDO = 0.95;
public static const float OZONE_ALBEDO = 0.0;

// Gaussian fit to the US Standard Atmosphere 1976.
public float rayleigh_density(float h, float sea_level_density)
{
    float x = h + 24239.99;
    return 3.68082 * exp(-x * x / 532307548.4168) / sea_level_density;
}

// Smoothed OPAC aerosol profile.
public float mie_density(float h, float turbidity)
{
    float density;
    if (h > 11500.0)
        density = 0.0918 * exp(-1.0e-6 * (h - 11500.0) * (h - 11500.0));
    else if (h > 2400.0)
        density = 0.3 * exp(-2.5e-9 * (h + 2500.0) * (h + 2500.0)) - 0.092;
    else if (h > 1300.0)
        density = 0.65 * exp(-5.0e-6 * (h - 1300.0) * (h - 1300.0)) + 0.18899;
    else
        density = 1.0 - h / 8136.646;
    return density * turbidity;
}

// Ozone number density (m^-3) at `h` >= 0, linear between the profile's
// samples and 0 above its last one.
public float ozone_density(StructuredBuffer<float> profile, float h)
{
    // The comparison stays in metres: the division may round 110 km past the last sample.
    float top = OZONE_PROFILE_STEP * float(OZONE_PROFILE_COUNT - 1);
    if (h > top)
        return 0.0;
    float x = h / OZONE_PROFILE_STEP;
    uint i = min(uint(x), OZONE_PROFILE_COUNT - 2);
    return lerp(profile[i], profile[i + 1], x - float(i));
}

// The largest sample bounds the linear interpolation everywhere.
public float ozone_density_max(StructuredBuffer<float> profile)
{
    float density_max = 0.0;
    for (uint i = 0; i < OZONE_PROFILE_COUNT; i++)
        density_max = max(density_max, profile[i]);
    return density_max;
}

// (Rayleigh, Mie, ozone) densities with the altitude clamped at sea level.
public float3 media_density(StructuredBuffer<float> ozone_profile, float h, float turbidity,
                            float rayleigh_sea_level_density)
{
    h = max(h, 0.0);
    return float3(rayleigh_density(h, rayleigh_sea_level_density), mie_density(h, turbidity),
                  ozone_density(ozone_profile, h));
}

// Peck and Reeder 1972 dispersion of standard air; `wavelength_um` in micrometres.
public float air_refractive_index(float wavelength_um)
{
    float rcp_w2 = 1.0 / (wavelength_um * wavelength_um);
    return 1.0 + 8.06051e-5 + 2.480990e-2 / (132.274 - rcp_w2) + 1.74557e-4 / (39.32957 - rcp_w2);
}

public float rayleigh_extinction(float wavelength)
{
    // The N2 and O2 depolarization fits expect micrometres but receive
    // nanometres, which pins the King factor at its long-wavelength limit
    // (about 1.047); the look is tuned to it.
    float rcp_w2 = 1.0 / (wavelength * wavelength);
    float f_n2 = 1.034 + 3.17e-4 * rcp_w2;
    float f_o2 = 1.096 + 1.385e-3 * rcp_w2 + 1.448e-4 * rcp_w2 * rcp_w2;
    const float co2 = 0.0421;
    float king = (78.084 * f_n2 + 20.946 * f_o2 + 0.934 + co2 * 1.15) / (78.084 + 20.946 + 0.934 + co2);

    float n = air_refractive_index(wavelength * 1e-3);
    float n2m1 = n * n - 1.0;
    float w_m = wavelength * 1e-9;
    float w_m2 = w_m * w_m;
    return 8.0 * PI * PI * PI * n2m1 * n2m1 / (3.0 * AIR_NUMBER_DENSITY * w_m2 * w_m2) * king;
}

// Preetham et al. 1999, Junge exponent 4.
public float mie_extinction(float wavelength, float turbidity)
{
    float c = (0.6544 * turbidity - 0.6510) * 4e-18;
    float k = (0.773335 - 0.00386891 * wavelength) / (1.0 - 0.00546759 * wavelength);
    float wavenumber = 2.0 * PI / (wavelength * 1e-9);
    return 0.434 * c * PI * wavenumber * wavenumber * k;
}

// Cross section per molecule in m^2. `cross_section` is LUT/ozone_cross_section.dat:
// cm^2 per molecule, entry i the mean over the 1 nm bin centred on 390 + i nm.
public float ozone_extinction(StructuredBuffer<float> cross_section, float wavelength)
{
    WavelengthTablePosition position = WavelengthTablePosition(wavelength);
    return 1e-4 * lerp(cross_section[position.index], cross_section[position.index + 1], position.weight);
}

// (Rayleigh, Mie, ozone) extinction at unit density; ozone's unit is one molecule per m^3.
public float3 media_extinction(StructuredBuffer<float> ozone_cross_section, float wavelength, float turbidity)
{
    return float3(rayleigh_extinction(wavelength), mie_extinction(wavelength, turbidity),
                  ozone_extinction(ozone_cross_section, wavelength));
}

// (1 - exp(-od)) / od: the fraction of a homogeneous step's in-scattering that
// survives the step. Below 1e-2 the f32 subtraction loses more than 1e-6
// relative, so the Taylor series (truncation < 1e-10) takes over.
public float step_integral(float optical_depth)
{
    float od = optical_depth;
    if (od < 1e-2)
        return 1.0 - od * (1.0 / 2.0 - od * (1.0 / 6.0 - od * (1.0 / 24.0)));
    return saturate((1.0 - saturate(exp(-od))) / od);
}

// The atmosphere at one wavelength, with the scene's media constants.
public struct SpectralMedium
{
    public float wavelength;
    // Rayleigh, Mie, and ozone extinction at unit density.
    public float3 extinction;
    // Rayleigh and Mie scattering at unit density.
    public float2 scattering;
    public float turbidity;
    public float rayleigh_sea_level_density;
    public StructuredBuffer<float> ozone_profile;

    // `media` is SceneUniforms.media.
    public __init(StructuredBuffer<float> ozone_cross_section, StructuredBuffer<float> ozone_profile, float wavelength,
                  float4 media)
    {
        this.wavelength = wavelength;
        this.ozone_profile = ozone_profile;
        turbidity = media.x;
        rayleigh_sea_level_density = media.w;
        extinction = media_extinction(ozone_cross_section, wavelength, turbidity);
        scattering = extinction.xy * float2(RAYLEIGH_ALBEDO, AEROSOL_ALBEDO);
    }

    public float3 density(float3 position)
    {
        return media_density(ozone_profile, length(position) - PLANET_RADIUS, turbidity, rayleigh_sea_level_density);
    }
};

// The atmosphere at the members of a wavelength bundle, one row each.
public struct SpectralMediumBundle
{
    // Rayleigh, Mie, and ozone extinction at unit density.
    public matrix<float, BUNDLE_WAVELENGTH_COUNT, 3> extinction;
    // Rayleigh and Mie scattering at unit density.
    public matrix<float, BUNDLE_WAVELENGTH_COUNT, 2> scattering;
    // The first member's medium; its densities serve every member.
    public SpectralMedium first;

    public __init(StructuredBuffer<float> ozone_cross_section, StructuredBuffer<float> ozone_profile,
                  Spectrum wavelength, float4 media)
    {
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        {
            SpectralMedium medium = SpectralMedium(ozone_cross_section, ozone_profile, wavelength[k], media);
            extinction[k] = medium.extinction;
            scattering[k] = medium.scattering;
            if (k == 0)
                first = medium;
        }
    }
};
