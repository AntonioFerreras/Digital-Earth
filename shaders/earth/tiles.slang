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
// One bit per indirection entry: a lookup sets the bit of the tile it wanted, which the
// host reads back to stream it in or keep it.
public RWStructuredBuffer<uint> earth_tile_requests;
// Per height-layer indirection entry, the largest height in metres of the tile's samples,
// or of its nearest ancestor present where the tile is absent.
public StructuredBuffer<int> earth_tile_height_max;
// Per height slot, BLOCK_ENTRIES maxima in metres of the slot's tile split into blocks
// 1 to BLOCK_LEVELS levels below it, coarsest first, each level row by row.
public StructuredBuffer<int> earth_tile_block_max;
// The 10 m wind of src/assets/WindMap.zig over sphere_uv, or one cell of alpha 0
// without a wind map.
public Texture2D<float4> earth_wind;
public SamplerState earth_wind_sampler;

static const uint INDIRECTION_LEVEL_BITS = 4;
// Asks for the finest level a layer has; lookups clamp it to the layer's last level.
public static const uint TILE_LEVEL_FINEST = (1u << INDIRECTION_LEVEL_BITS) - 1;
// The arc of a cube face's middle row, which its tiles split 2^level ways.
public static const float FACE_SPAN = 0.5 * PI * PLANET_RADIUS;

// The level whose texels match a ray footprint `width` metres across.
public uint tile_level(float width)
{
    float level = ceil(log2(FACE_SPAN / (float(TILE_INTERVALS) * max(width, 1.0))));
    return uint(clamp(level, 0.0, float(TILE_LEVEL_FINEST)));
}
public static const uint TILE_INTERVALS = 192;
// Block levels below each height tile; the finest blocks span 3 intervals.
public static const uint BLOCK_LEVELS = 6;
static const uint BLOCK_ENTRIES = ((1u << (2 * (BLOCK_LEVELS + 1))) - 4) / 3;
static const uint TILE_BORDER = 2;
static const uint HEIGHT_SIDE = TILE_INTERVALS + 1 + 2 * TILE_BORDER;
static const uint SURFACE_SIDE = TILE_INTERVALS + 2 * TILE_BORDER;

// A cube face's outward normal and the axes its (s, t) coordinates grow along.
public struct FaceFrame
{
    public float3 normal;
    public float3 s_axis;
    public float3 t_axis;

    public __init(uint face)
    {
        float sign = (face & 1) == 0 ? 1.0 : -1.0;
        if (face < 2)
        {
            normal = float3(sign, 0.0, 0.0);
            s_axis = float3(0.0, 0.0, -sign);
            t_axis = float3(0.0, 1.0, 0.0);
        }
        else if (face < 4)
        {
            normal = float3(0.0, sign, 0.0);
            s_axis = float3(1.0, 0.0, 0.0);
            t_axis = float3(0.0, 0.0, -sign);
        }
        else
        {
            normal = float3(0.0, 0.0, sign);
            s_axis = float3(sign, 0.0, 0.0);
            t_axis = float3(0.0, 1.0, 0.0);
        }
    }

    // The plane through the planet centre holding the directions at face coordinate
    // `u` along `axis` (`s_axis` or `t_axis`): its points x satisfy dot(x, normal) = 0.
    public float3 edge_plane(float3 axis, float u)
    {
        return axis - tan(0.5 * PI * (u - 0.5)) * normal;
    }
};

// A point on a cube face: (u, v) in [0, 1] along the face's s and t axes.
public struct FacePoint
{
    public uint face;
    public float2 uv;

    // The face point of the unit direction `n`, through the equal-angle warp.
    public __init(float3 n)
    {
        float3 m = abs(n);
        if (m.x >= m.y && m.x >= m.z)
            face = n.x > 0.0 ? 0 : 1;
        else if (m.y >= m.z)
            face = n.y > 0.0 ? 2 : 3;
        else
            face = n.z > 0.0 ? 4 : 5;
        FaceFrame frame = FaceFrame(face);
        float2 gnomonic = float2(dot(n, frame.s_axis), dot(n, frame.t_axis)) / dot(n, frame.normal);
        uv = 0.5 + (2.0 / PI) * float2(atan(gnomonic.x), atan(gnomonic.y));
    }
};

// The unit direction at face coordinates `uv` of `face`, the inverse of FacePoint; uv
// may run past [0, 1] onto the face's continued warp, where tile borders sample.
public float3 face_direction(uint face, float2 uv)
{
    FaceFrame frame = FaceFrame(face);
    float2 gnomonic = tan(0.5 * PI * (uv - 0.5));
    return normalize(frame.normal + gnomonic.x * frame.s_axis + gnomonic.y * frame.t_axis);
}

// Where a lookup lands: the pool slot of the finest resident tile at or above the level
// asked for, and the point's position within it in [0, 1]^2.
public struct TileHit
{
    public uint face;
    public uint slot;
    public uint level;
    public float2 local;
    // The resident tile's position at its level, and its height-layer entry index.
    public uint2 tile;
    public uint index;

    // `layer` is the layer's TileUniforms field; with `request`, the lookup sets the
    // wanted tile's request bit, which loads it or, once resident, keeps it.
    public __init(uint4 layer, FacePoint p, uint level_wanted, bool request)
    {
        face = p.face;
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
        if (request && (earth_tile_requests[index >> 5] & bit) == 0)
            InterlockedOr(earth_tile_requests[index >> 5], bit);
        float resident_side = float(1u << level);
        this.tile = tile >> (level_asked - level);
        this.index = layer.y + p.face * entry_count(layer.x + 1) + entry_count(level) + morton(this.tile);
        local = p.uv * resident_side - float2(this.tile);
    }

    // Moves the lookup to `p`, a point of the same resident tile.
    [mutating]
    public void move_to(FacePoint p)
    {
        local = p.uv * float(1u << level) - float2(tile);
    }

    // The largest height in metres a lookup can return in tile `area` of `area_level`,
    // which holds this lookup's point: the indirection entry's maximum at or above the
    // resident level, a block maximum of the resident tile below it.
    public int area_max(uint area_level, uint2 area)
    {
        if (area_level <= level)
        {
            return earth_tile_height_max[earth_tiles.height.y + face * entry_count(earth_tiles.height.x + 1) +
                                         entry_count(area_level) + morton(area)];
        }
        uint below = area_level - level;
        uint2 block = area - (tile << below);
        return earth_tile_block_max[slot * BLOCK_ENTRIES + ((1u << (2 * below)) - 4) / 3 + (block.y << below) + block.x];
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

// Bit k of x at bit 2k, bit k of y at bit 2k + 1, for x and y below 2^16.
public uint morton(uint2 tile)
{
    uint2 v = tile & 0xFFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v.x | (v.y << 1);
}

// Albedo at the finest resident level up to `level_wanted`.
public float3 tile_albedo(float3 n, uint level_wanted, bool request)
{
    TileHit hit = TileHit(earth_tiles.surface, FacePoint(n), level_wanted, request);
    float2 texel = float(TILE_BORDER) + hit.local * float(TILE_INTERVALS);
    return earth_tile_surface.SampleLevel(earth_tile_sampler, float3(texel / float(SURFACE_SIDE), float(hit.slot)), 0.0).rgb;
}

// Water fraction in [0, 1].
public float tile_water(float3 n, uint level_wanted, bool request)
{
    TileHit hit = TileHit(earth_tiles.height, FacePoint(n), level_wanted, request);
    float2 texel = hit.vertex_index() + 0.5;
    return earth_tile_water.SampleLevel(earth_tile_sampler, float3(texel / float(HEIGHT_SIDE), float(hit.slot)), 0.0);
}

// Height in metres, bilinear in f32 between the four nearest samples.
public float tile_height(float3 n, uint level_wanted, bool request)
{
    return tile_height_at(TileHit(earth_tiles.height, FacePoint(n), level_wanted, request));
}

// The height at `hit`, a lookup of the height layer.
public float tile_height_at(TileHit hit)
{
    float2 index = hit.vertex_index();
    int2 base = int2(floor(index));
    float2 f = index - float2(base);
    float h00 = float(earth_tile_heights.Load(int4(base, hit.slot, 0)));
    float h10 = float(earth_tile_heights.Load(int4(base + int2(1, 0), hit.slot, 0)));
    float h01 = float(earth_tile_heights.Load(int4(base + int2(0, 1), hit.slot, 0)));
    float h11 = float(earth_tile_heights.Load(int4(base + int2(1, 1), hit.slot, 0)));
    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
}
