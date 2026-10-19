// sRGB-encoded reflectance to spectral reflectance through a three-basis table.
//
// The `basis` buffer is LUT/srgb2spec.dat converted from f16 to f32:
// basis[3*i + c] is the weight of channel c (r, g, b) at 400 + i nm, i < 300.

module uplift;

static const float BASIS_MIN = 400.0;
static const uint BASIS_COUNT = 300;

float3 basis_entry(StructuredBuffer<float> basis, uint i)
{
    return float3(basis[3 * i], basis[3 * i + 1], basis[3 * i + 2]);
}

// Wavelengths outside [400, 699] nm take the nearest edge row.
public float srgb_to_spectrum(StructuredBuffer<float> basis, float3 rgb, float wavelength)
{
    float x = clamp(wavelength - BASIS_MIN, 0.0, float(BASIS_COUNT - 1));
    uint i = min(uint(x), BASIS_COUNT - 2);
    return dot(rgb, lerp(basis_entry(basis, i), basis_entry(basis, i + 1), x - float(i)));
}
