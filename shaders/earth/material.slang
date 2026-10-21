module material;

public static const float3 REC709_LUMINANCE = float3(0.2126729, 0.7151522, 0.0721750);

// Grades the sRGB-encoded albedo map into the surface albedo (still sRGB
// encoded): greenery darkened and desaturated, deserts saturated and warmed,
// oceans desaturated and darkened. `ocean` is the ocean mask (1 = water).
public float3 grade_albedo(float3 albedo, float ocean)
{
    float3 grey = dot(albedo, REC709_LUMINANCE).xxx;
    float3 boosted = lerp(grey, albedo, 6.5);
    float green_ratio = boosted.y / dot(boosted, REC709_LUMINANCE);
    float greenery = smoothstep(1.5, 1.9, green_ratio * green_ratio);

    float3 land = albedo / (greenery * 0.7 + 1.0);
    land = lerp(dot(land, REC709_LUMINANCE).xxx, land, 1.4 - greenery * 0.45);
    land = lerp(land, land * float3(255.0, 128.0, 64.0) / 255.0, 0.2 * (1.0 - greenery));

    float3 water = lerp(grey, albedo, 0.75) * 0.9;
    return lerp(land, water, ocean);
}

// Decodes sRGB-encoded `encoded` to linear.
public float3 srgb_eotf(float3 encoded)
{
    return select(encoded <= 0.04045, encoded / 12.92, pow((encoded + 0.055) / 1.055, 2.4));
}

// Blue Marble fills every deep ocean with this one encoded sRGB colour; shallow water
// keeps its measured colour.
static const float3 DEEP_OCEAN_FILL = float3(2.0, 5.0, 20.0) / 255.0;

// How much of the albedo map's colour is the deep-ocean fill: 1 at the fill, falling
// to 0 by a few sRGB steps away, which JPEG noise and averaged levels stay within.
public float deep_ocean_fill(float3 albedo_srgb)
{
    return 1.0 - smoothstep(4.0 / 255.0, 12.0 / 255.0, length(albedo_srgb - DEEP_OCEAN_FILL));
}

// Reflectance of clear open ocean seen from straight above at `wavelength` nm, about
// Morel and Maritorena's (2001) water at 0.03 mg/m^3 of chlorophyll.
public float clear_ocean_reflectance(float wavelength)
{
    static const float WAVELENGTHS[7] = { 400.0, 440.0, 490.0, 550.0, 600.0, 650.0, 700.0 };
    static const float REFLECTANCES[7] = { 0.024, 0.022, 0.014, 0.0032, 0.001, 0.0003, 0.0 };
    if (wavelength <= WAVELENGTHS[0])
        return REFLECTANCES[0];
    for (uint i = 1; i < 7; i++)
    {
        if (wavelength <= WAVELENGTHS[i])
            return lerp(REFLECTANCES[i - 1], REFLECTANCES[i], (wavelength - WAVELENGTHS[i - 1]) / (WAVELENGTHS[i] - WAVELENGTHS[i - 1]));
    }
    return 0.0;
}
