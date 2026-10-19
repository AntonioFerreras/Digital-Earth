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
