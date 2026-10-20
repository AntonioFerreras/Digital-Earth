// The streamed cube-face tile quadtrees of src/render/TilePool.zig: lookups that find a
// tile's finest resident ancestor through the indirection and read it. Layout, mapping,
// and sample positions follow docs/plans/20261020-streamed-earth-tiles.md.

module tiles;

import common.constants;
import shared.uniforms;

public ConstantBuffer<TileUniforms> earth_tiles;
// `slot << INDIRECTION_LEVEL_BITS | level` of the finest resident tile containing each
// (face, level, x, y), height layer first, then surface.
public StructuredBuffer<uint> earth_tile_indirection;
// Metres, vertex-centred.
public Texture2DArray<int> earth_tile_heights;
// Water fraction, vertex-centred like the heights.
public Texture2DArray<float> earth_tile_water;
// Encoded sRGB albedo, pixel-centred.
public Texture2DArray<float4> earth_tile_surface;
public SamplerState earth_tile_sampler;
// One bit per indirection entry: a lookup that wanted a tile finer than the resident one
// sets the wanted tile's bit, which the host reads back to stream it in.
public RWStructuredBuffer<uint> earth_tile_requests;

static const uint INDIRECTION_LEVEL_BITS = 4;
// Asks for the finest level a layer has; lookups clamp it to the layer's last level.
public static const uint TILE_LEVEL_FINEST = (1u << INDIRECTION_LEVEL_BITS) - 1;
// The arc of a cube face's middle row, which its tiles split 2^level ways.
static const float FACE_SPAN = 0.5 * PI * PLANET_RADIUS;

// The level whose texels match a ray footprint `width` metres across.
public uint tile_level(float width)
{
    float level = ceil(log2(FACE_SPAN / (float(TILE_INTERVALS) * max(width, 1.0))));
    return uint(clamp(level, 0.0, float(TILE_LEVEL_FINEST)));
}
static const uint TILE_INTERVALS = 192;
static const uint TILE_BORDER = 2;
static const uint HEIGHT_SIDE = TILE_INTERVALS + 1 + 2 * TILE_BORDER;
static const uint SURFACE_SIDE = TILE_INTERVALS + 2 * TILE_BORDER;

// A point on a cube face: (u, v) in [0, 1] along the face's s and t axes.
public struct FacePoint
{
    public uint face;
    public float2 uv;

    // The face point of the unit direction `n`, through the equal-angle warp.
    public __init(float3 n)
    {
        float3 m = abs(n);
        float3 s_axis, t_axis;
        float major;
        if (m.x >= m.y && m.x >= m.z)
        {
            face = n.x > 0.0 ? 0 : 1;
            s_axis = float3(0.0, 0.0, n.x > 0.0 ? -1.0 : 1.0);
            t_axis = float3(0.0, 1.0, 0.0);
            major = m.x;
        }
        else if (m.y >= m.z)
        {
            face = n.y > 0.0 ? 2 : 3;
            s_axis = float3(1.0, 0.0, 0.0);
            t_axis = float3(0.0, 0.0, n.y > 0.0 ? -1.0 : 1.0);
            major = m.y;
        }
        else
        {
            face = n.z > 0.0 ? 4 : 5;
            s_axis = float3(n.z > 0.0 ? 1.0 : -1.0, 0.0, 0.0);
            t_axis = float3(0.0, 1.0, 0.0);
            major = m.z;
        }
        float2 gnomonic = float2(dot(n, s_axis), dot(n, t_axis)) / major;
        uv = 0.5 + (2.0 / PI) * float2(atan(gnomonic.x), atan(gnomonic.y));
    }
};

// Where a lookup lands: the pool slot of the finest resident tile at or above the level
// asked for, and the point's position within it in [0, 1]^2.
public struct TileHit
{
    public uint slot;
    public uint level;
    public float2 local;

    // `layer` is the layer's TileUniforms field.
    public __init(uint4 layer, FacePoint p, uint level_wanted)
    {
        uint level_asked = min(level_wanted, layer.x);
        uint side = 1u << level_asked;
        uint2 tile = min(uint2(p.uv * float(side)), side - 1);
        uint index = layer.y + p.face * entry_count(layer.x + 1) + entry_count(level_asked) + morton(tile);
        uint entry = earth_tile_indirection[index];
        slot = entry >> INDIRECTION_LEVEL_BITS;
        level = entry & ((1u << INDIRECTION_LEVEL_BITS) - 1);
        // Reading first spares the atomic when the bit is already set, as it is for all
        // but the first of the many lookups that want one tile.
        uint bit = 1u << (index & 31);
        if (level < level_asked && (earth_tile_requests[index >> 5] & bit) == 0)
            InterlockedOr(earth_tile_requests[index >> 5], bit);
        float resident_side = float(1u << level);
        local = p.uv * resident_side - float2(tile >> (level_asked - level));
    }

    // Continuous index into the bordered grid of a vertex-centred tile.
    float2 vertex_index()
    {
        return float(TILE_BORDER) + local * float(TILE_INTERVALS);
    }
};

// Entries of the levels above `level`: (4^level - 1) / 3.
public uint entry_count(uint level)
{
    return ((1u << (2 * level)) - 1) / 3;
}

// Bit k of x at bit 2k, bit k of y at bit 2k + 1.
public uint morton(uint2 tile)
{
    uint result = 0;
    for (uint bit = 0; bit < 16; bit++)
        result |= ((tile.x >> bit) & 1) << (2 * bit) | ((tile.y >> bit) & 1) << (2 * bit + 1);
    return result;
}

// Albedo at the finest resident level up to `level_wanted`.
public float3 tile_albedo(float3 n, uint level_wanted)
{
    TileHit hit = TileHit(earth_tiles.surface, FacePoint(n), level_wanted);
    float2 texel = float(TILE_BORDER) + hit.local * float(TILE_INTERVALS);
    return earth_tile_surface.SampleLevel(earth_tile_sampler, float3(texel / float(SURFACE_SIDE), float(hit.slot)), 0.0).rgb;
}

// Water fraction in [0, 1].
public float tile_water(float3 n, uint level_wanted)
{
    TileHit hit = TileHit(earth_tiles.height, FacePoint(n), level_wanted);
    float2 texel = hit.vertex_index() + 0.5;
    return earth_tile_water.SampleLevel(earth_tile_sampler, float3(texel / float(HEIGHT_SIDE), float(hit.slot)), 0.0);
}

// Height in metres, bilinear in f32 between the four nearest samples.
public float tile_height(float3 n, uint level_wanted)
{
    TileHit hit = TileHit(earth_tiles.height, FacePoint(n), level_wanted);
    float2 index = hit.vertex_index();
    int2 base = int2(floor(index));
    float2 f = index - float2(base);
    float h00 = float(earth_tile_heights.Load(int4(base, hit.slot, 0)));
    float h10 = float(earth_tile_heights.Load(int4(base + int2(1, 0), hit.slot, 0)));
    float h01 = float(earth_tile_heights.Load(int4(base + int2(0, 1), hit.slot, 0)));
    float h11 = float(earth_tile_heights.Load(int4(base + int2(1, 1), hit.slot, 0)));
    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
}
