// Counter-based random numbers: every draw is a pure hash of
// (pixel, frame, dimension), so a frame renders identically on every run and
// no generator state lives in memory.

module rng;

// Jarzynski and Olano 2020, "Hash Functions for GPU Rendering", pcg4d.
public uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

public struct Rng
{
    public uint2 pixel;
    public uint frame;
    public uint dimension;

    public __init(uint2 pixel, uint frame)
    {
        this.pixel = pixel;
        this.frame = frame;
        this.dimension = 0;
    }

    // Four independent numbers in [0, 1) per dimension; callers needing fewer
    // drop the rest so that sample dimensions stay stable across code changes.
    [mutating]
    public float4 next4()
    {
        uint4 bits = pcg4d(uint4(pixel, frame, dimension));
        dimension += 1;
        // The top 24 bits fill the f32 mantissa exactly.
        return float4(bits >> 8u) * (1.0 / 16777216.0);
    }
};
