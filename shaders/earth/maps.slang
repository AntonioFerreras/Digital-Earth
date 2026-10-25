module maps;

import common.constants;

// Equirectangular coordinates of the unit `n`: +Y is north, u = 0.5 is the
// Greenwich meridian on -X, u = 0.75 is 90 E on +Z, and the seam is +X. v = 0
// is the south pole, matching the south-first rows of the map cache.
public float2 sphere_uv(float3 n)
{
    return float2((atan2(n.z, -n.x) / PI + 1.0) / 2.0, asin(n.y) / PI + 0.5);
}

// atan2 within 2e-6 rad: an odd minimax polynomial of min / max over [0, 1], then
// the octant restored. Zero when both are zero.
public float atan2_fast(float y, float x)
{
    float ax = abs(x);
    float ay = abs(y);
    float a = min(ax, ay) / max(max(ax, ay), 1e-30);
    float s = a * a;
    float r = a * (0.999977219 + s * (-0.332622829 + s * (0.193540368 + s * (-0.116426444 +
                   s * (0.052647299 + s * -0.011719113)))));
    r = ay > ax ? 0.5 * PI - r : r;
    r = x < 0.0 ? PI - r : r;
    return y < 0.0 ? -r : r;
}

// `sphere_uv` of the direction of `p`, which need not be unit, within 1e-6 of the
// exact coordinates, a small fraction of the finest map's texel. For lookups many per
// pixel, where atan2 and asin cost more than the texture fetch.
public float2 sphere_uv_fast(float3 p)
{
    return float2((atan2_fast(p.z, -p.x) / PI + 1.0) / 2.0, atan2_fast(p.y, length(p.xz)) / PI + 0.5);
}
