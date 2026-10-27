// Direction sampling from uniform numbers `u` in [0, 1), each function consuming its
// components in the order the original drew them, and the taps of texture filters.

module sampling;

import common.constants;
import common.geometry;

// Uniform over the cone of directions within acos(cos_theta_max) of the unit `axis`.
public float3 sample_cone(float2 u, float cos_theta_max, float3 axis)
{
    float cos_theta = (1.0 - u.x) + u.x * cos_theta_max;
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    float phi = 2.0 * PI * u.y;
    return local_to_world(float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta), axis);
}

// Cosine-weighted hemisphere about the unit `n` (Shirley et al. 2019, Ray
// Tracing Gems ch. 16): a unit-sphere point offset by `n`.
public float3 sample_hemisphere_cosine(float2 u, float3 n)
{
    float z = 1.0 - 2.0 * u.x;
    // Shrinking the sphere keeps grazing samples off the tangent plane.
    float shrink = 1.0 - 1e-5;
    float ring = sqrt(1.0 - z * z) * shrink;
    float phi = 2.0 * PI * u.y;
    return normalize(n + float3(ring * cos(phi), ring * sin(phi), z * shrink));
}

public float3 sample_sphere_uniform(float2 u)
{
    float phi = 2.0 * PI * u.x;
    float z = 2.0 * u.y - 1.0;
    float ring = sqrt(1.0 - z * z);
    return normalize(float3(sin(phi) * ring, cos(phi) * ring, z));
}

// The cubic B-spline through a texture's texels, smooth in slope across them, as the
// two filtered fetches per axis it splits into (Sigg and Hadwiger 2005): at texel
// coordinate `p` of a texture of `size`, the fetch at `lower` weighs `g0` and that at
// `upper` `g1`, per axis, in texture coordinates. Its support reaches two texels.
public struct CubicTaps<let N : int>
{
    public vector<float, N> lower;
    public vector<float, N> upper;
    public vector<float, N> g0;
    public vector<float, N> g1;

    public __init(vector<float, N> p, vector<float, N> size)
    {
        vector<float, N> i = floor(p);
        vector<float, N> f = p - i;
        vector<float, N> f2 = f * f;
        vector<float, N> f3 = f2 * f;
        vector<float, N> w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
        vector<float, N> w1 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0;
        vector<float, N> w3 = f3 / 6.0;
        g0 = w0 + w1;
        g1 = 1.0 - g0;
        lower = (i - 0.5 + w1 / g0) / size;
        upper = (i + 1.5 + w3 / g1) / size;
    }
};
