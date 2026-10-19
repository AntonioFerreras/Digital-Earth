// A dense 4D table over uvwz = (nu, mu_s, mu, r) with texel counts
// shape = (NU, NV, NW, NZ), stored as Bruneton's 3D texture: nu folded into
// the width as NU blocks of NV mu_s texels, mu along the height, r along the
// depth. Texel (ku, kv, kw, kz) sits at (ku * NV + kv, kw, kz).
//
// Texels are vertices: texel k of an axis holds the radiance at fraction
// k / (N - 1) of the axis's uvwz range (atmosphere/bruneton.slang), so the
// first and last texels sit on the ends of the domain. mu holds two ranges of
// NW / 2 texels each, the ground rays and then the sky rays.
// datagen_format.gridPoint (src/assets/datagen_format.zig) generates the same
// points; lut4d_sample inverts it.
//
// A vertex's value lives at its texel centre, (k + 0.5) / N. The sampler
// (linear filtering, clamp addressing) interpolates mu_s, mu, and r at its
// fixed-point weight precision; nu interpolates exactly between two blocks.
// Coordinates clamp to the range of their axis, or of their mu half, so
// filtering never crosses into the neighbouring nu block or mu half.

module lut4d;

import atmosphere.bruneton;

// Where `u` lies in [low, high], from 0 to 1.
float range_fraction(float u, float low, float high)
{
    return saturate((u - low) / (high - low));
}

public float3 lut4d_sample(Texture3D<float4> lut, SamplerState linear_clamp, uint4 shape, float4 uvwz)
{
    float4 size = float4(shape);
    float nu = range_fraction(uvwz.x, 0.0, 1.0) * (size.x - 1.0);
    float nu_lower = min(floor(nu), size.x - 2.0);
    float mu_s = range_fraction(uvwz.y, V_LOW, V_HIGH) * (size.y - 1.0) + 0.5;
    float mu_half = 0.5 * size.z;
    float mu = uvwz.z < 0.5 ? range_fraction(uvwz.z, W_GROUND_LOW, W_GROUND_HIGH) * (mu_half - 1.0) + 0.5
                            : range_fraction(uvwz.z, W_SKY_LOW, W_SKY_HIGH) * (mu_half - 1.0) + 0.5 + mu_half;
    float r = range_fraction(uvwz.w, Z_LOW, Z_HIGH) * (size.w - 1.0) + 0.5;
    float2 mu_r = float2(mu / size.z, r / size.w);
    float width = size.x * size.y;
    float3 lower = lut.SampleLevel(linear_clamp, float3((nu_lower * size.y + mu_s) / width, mu_r), 0.0).rgb;
    float3 upper = lut.SampleLevel(linear_clamp, float3(((nu_lower + 1.0) * size.y + mu_s) / width, mu_r), 0.0).rgb;
    return lerp(lower, upper, nu - nu_lower);
}
