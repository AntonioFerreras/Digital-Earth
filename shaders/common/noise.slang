// Gradient noise for procedural detail that must stay put on the surface: a pure
// function of position, like the random numbers.

module noise;

import common.rng;

// Pseudo-random gradient of lattice point `cell`, each component in [-1, 1].
float3 lattice_gradient(int3 cell)
{
    uint3 h = pcg4d(uint4(asuint(cell), 0u)).xyz;
    return float3(h >> 16u) / 32767.5 - 1.0;
}

// Perlin's (2002) improved gradient noise at `p`: 0 on the lattice, mostly within
// [-1, 1], continuous in value and slope.
public float gradient_noise(float3 p)
{
    float3 floor_p = floor(p);
    int3 cell = int3(floor_p);
    float3 f = p - floor_p;
    float3 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float corners[8];
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        int3 offset = int3(i & 1, (i >> 1) & 1, i >> 2);
        corners[i] = dot(lattice_gradient(cell + offset), f - float3(offset));
    }
    float4 x = lerp(float4(corners[0], corners[2], corners[4], corners[6]), float4(corners[1], corners[3], corners[5], corners[7]), w.x);
    float2 y = lerp(x.xz, x.yw, w.y);
    return lerp(y.x, y.y, w.z);
}
