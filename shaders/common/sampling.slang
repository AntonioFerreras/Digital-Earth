// Direction sampling from uniform numbers `u` in [0, 1); each function consumes
// its components in the order the original drew them.

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
