// The Earth map globals and their sampling. src/render/ModeKernel.zig binds
// these globals to every mode kernel that declares them, which importing this
// module (directly or through earth/surface.slang) does.
//
// Each map is a Texture2DArray of longitude tiles laid out by
// src/assets/TilePlan.zig, whose doc comment derives the tile lookup below.
// Texels are UNORM as stored (no sRGB decode), v = 0 is the south pole, and
// map coordinates come from earth/maps.slang `sphere_uv`.

module earth_bindings;

import shared.bindings;
import shared.uniforms;

public ConstantBuffer<EarthMapUniforms> earth_maps;
// Encoded sRGB colour; alpha is unused.
public Texture2DArray<float4> earth_albedo;
// 0 is sea level, 1 is the scene's land height scale.
public Texture2DArray<float> earth_topography;
// 1 over water.
public Texture2DArray<float> earth_ocean_mask;
// Cloud coverage.
public Texture2DArray<float> earth_clouds;
// 1 for land or shallow water, 0 for the deepest ocean.
public Texture2DArray<float> earth_bathymetry;
public Texture2DArray<float> earth_night_lights;
// Encoded sRGB colour of the sky, looked up by direction.
public Texture2DArray<float4> earth_stars;
// Coarse bounds of the topography (r) and the clouds (g), src/assets/MapBounds.zig.
public Texture2D<float4> earth_bounds;

// Tile of map column `x` (texel units) under `layout`, a field of `earth_maps`.
uint tile_of(float x, uint4 layout)
{
    // A float quotient instead of an integer division, which costs more than the
    // fetch. Where it rounds across a tile edge, the taps fall within the half texel
    // that the one-texel apron covers.
    return min(uint(x / float(layout.z)), (layout.x - 1) / layout.z);
}

// Bilinear sample at map coordinates `uv` with hardware filtering, u repeating
// and v clamped, LOD 0.
public T sample_tiled<T : ITexelElement>(Texture2DArray<T> map, uint4 layout, float2 uv)
{
    float x = frac(uv.x) * float(layout.x);
    uint tile = tile_of(x, layout);
    float layer_width = float(layout.z + 2 * layout.w);
    float u = (x - float(tile * layout.z) + float(layout.w)) / layer_width;
    return map.SampleLevel(linear_repeat_u, float3(u, uv.y, float(tile)), 0.0);
}

float texel(Texture2DArray<float> map, uint4 layout, uint column, uint row)
{
    uint tile = tile_of(float(column), layout);
    return map.Load(int4(int(column - tile * layout.z + layout.w), int(row), int(tile), 0));
}

// `sample_tiled` with f32 weights from four texel loads: hardware filtering
// quantizes sub-texel weights to 8 bits, which steps a sphere-traced
// heightfield and its finite-difference normals.
public float sample_tiled_exact(Texture2DArray<float> map, uint4 layout, float2 uv)
{
    float2 x = float2(frac(uv.x) * float(layout.x), saturate(uv.y) * float(layout.y)) - 0.5;
    float2 base = floor(x);
    float2 f = x - base;
    int2 first = int2(base);
    uint column0 = uint(first.x + int(layout.x)) % layout.x;
    uint column1 = (column0 + 1) % layout.x;
    uint row0 = uint(clamp(first.y, 0, int(layout.y) - 1));
    uint row1 = uint(clamp(first.y + 1, 0, int(layout.y) - 1));
    float south = lerp(texel(map, layout, column0, row0), texel(map, layout, column1, row0), f.x);
    float north = lerp(texel(map, layout, column0, row1), texel(map, layout, column1, row1), f.x);
    return lerp(south, north, f.y);
}
