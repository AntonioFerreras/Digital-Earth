// Lens vignette and display encoding.

module output;

// Radial falloff measured from the pixel's lower-left corner, as the original
// indexed it; the image centre keeps full brightness.
public float vignette(uint2 pixel, uint2 size)
{
    const float strength = 0.9;
    float2 uv = float2(pixel) / float2(size);
    return 1.0 - strength * length(uv - 0.5);
}

public float3 srgb_oetf(float3 linear)
{
    float3 lo = linear * 12.92;
    float3 hi = pow(abs(linear), 1.0 / 2.4) * 1.055 - 0.055;
    return select(linear <= 0.0031308, lo, hi);
}
