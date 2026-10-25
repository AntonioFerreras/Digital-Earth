// Clarberg's equal-area octahedral map between the unit square and the unit sphere
// (Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"), in
// pbrt-v4's form: equal areas of the square cover equal solid angles.

module octahedral;

import common.constants;

public float3 equal_area_square_to_sphere(float2 p)
{
    float2 uv = 2.0 * p - 1.0;
    float2 a = abs(uv);
    float signed_distance = 1.0 - (a.x + a.y);
    float r = 1.0 - abs(signed_distance);
    float phi = (r == 0.0 ? 1.0 : (a.y - a.x) / r + 1.0) * 0.25 * PI;
    float z = (signed_distance < 0.0 ? -1.0 : 1.0) * (1.0 - r * r);
    float ring = r * sqrt(max(2.0 - r * r, 0.0));
    float cos_phi = (uv.x < 0.0 ? -1.0 : 1.0) * cos(phi);
    float sin_phi = (uv.y < 0.0 ? -1.0 : 1.0) * sin(phi);
    return float3(cos_phi * ring, sin_phi * ring, z);
}

public float2 equal_area_sphere_to_square(float3 dir)
{
    float3 a = abs(dir);
    float r = sqrt(max(1.0 - a.z, 0.0));
    float larger = max(a.x, a.y);
    float ratio = larger == 0.0 ? 0.0 : min(a.x, a.y) / larger;
    float phi = atan(ratio) * 2.0 / PI;
    if (a.x < a.y)
        phi = 1.0 - phi;
    float v = phi * r;
    float u = r - v;
    if (dir.z < 0.0)
    {
        float swapped = u;
        u = 1.0 - v;
        v = 1.0 - swapped;
    }
    u = dir.x < 0.0 ? -u : u;
    v = dir.y < 0.0 ? -v : v;
    return 0.5 * float2(u, v) + 0.5;
}
