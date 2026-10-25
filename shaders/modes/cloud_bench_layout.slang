// The occupancy and cell depths from which src/render/ProbeLayout.zig lays the cloud
// bench's probes out, measured on the GPU by src/render/LayoutDepths.zig.

import common.constants;

struct LayoutUniforms
{
    // xyz: voxels per axis; w: voxels per occupancy texel along an axis.
    uint4 voxels;
    // xyz: occupancy texels per axis.
    uint4 texels;
    // xyz: the layout's finest cells per axis.
    uint4 cells;
};

ConstantBuffer<LayoutUniforms> layout;
Texture3D<float> layout_density;
// The volume's distance inside the cloud per texel, which `mark_occupancy` turns into
// the occupancy: the distance where any density the march's trilinear filter reads
// from within the texel is above zero, else -1.
[format("r16f")]
RWTexture3D<float> layout_occupancy;
// Per finest cell along x, then y, then z: the first and last occupancy texel that
// overlaps it, which the host computes so that the cells match its own.
StructuredBuffer<uint2> layout_ranges;
// Per finest cell, x fastest: the least depth inside the cloud over it, zero where it
// also holds air, or -1 where it holds no cloud.
RWStructuredBuffer<float> layout_depths;

[shader("compute")]
[numthreads(4, 4, 4)]
void mark_occupancy(uint3 texel: SV_DispatchThreadID)
{
    if (any(texel >= layout.texels.xyz))
        return;
    // The voxels the trilinear filter reads from points in the texel: its own and one
    // more on either side.
    uint3 first = uint3(max(int3(texel * layout.voxels.w) - 1, 0));
    uint3 last = min(texel * layout.voxels.w + layout.voxels.w + 1, layout.voxels.xyz);
    for (uint z = first.z; z < last.z; z++)
        for (uint y = first.y; y < last.y; y++)
            for (uint x = first.x; x < last.x; x++)
                if (layout_density.Load(int4(x, y, z, 0)) > 0.0)
                    return;
    layout_occupancy[texel] = -1.0;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void measure_cells(uint3 cell: SV_DispatchThreadID)
{
    uint3 cells = layout.cells.xyz;
    if (any(cell >= cells))
        return;
    uint2 x = layout_ranges[cell.x];
    uint2 y = layout_ranges[cells.x + cell.y];
    uint2 z = layout_ranges[cells.x + cells.y + cell.z];
    bool occupied = false;
    float depth = FLOAT_MAX;
    for (uint tz = z.x; tz <= z.y; tz++)
        for (uint ty = y.x; ty <= y.y; ty++)
            for (uint tx = x.x; tx <= x.y; tx++)
            {
                float value = layout_occupancy[uint3(tx, ty, tz)];
                occupied = occupied || value >= 0.0;
                // Air is at depth zero.
                depth = min(depth, max(value, 0.0));
            }
    layout_depths[cell.x + cells.x * (cell.y + cells.y * cell.z)] = occupied ? depth : -1.0;
}
