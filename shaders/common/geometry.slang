// Ray-sphere intersection and local frames.

module geometry;

// Both distances negative, so every `t > 0` and `t >= 0` test reads it as no
// forward intersection, as the original NaN comparisons did.
public static const float2 RAY_SPHERE_MISS = float2(-1.0, -1.0);

// Signed distances along the unit `dir` to the near and far intersections of a
// sphere of `radius` centred at the origin, or RAY_SPHERE_MISS.
public float2 ray_sphere(float3 pos, float3 dir, float radius)
{
    float b = dot(pos, dir);
    float discriminant = b * b - dot(pos, pos) + radius * radius;
    if (discriminant < 0.0)
        return RAY_SPHERE_MISS;
    float s = sqrt(discriminant);
    return float2(-b - s, -b + s);
}

// Maps `local` from the frame whose z axis is the unit `n` to world space; the
// tangent choice matches the original so sampled directions coincide.
public float3 local_to_world(float3 local, float3 n)
{
    float3 h = abs(n.y) > 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    float3 bitangent = normalize(cross(n, h));
    float3 tangent = cross(n, bitangent);
    return local.x * tangent + local.y * bitangent + local.z * n;
}
