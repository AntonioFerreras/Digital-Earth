//! Where the cloud bench's transfer probes lie (src/render/CloudBench.zig), and which
//! of them a point interpolates: the corners of the leaves of an octree of cells over
//! the volume's box, finest along the cloud's surface and coarser with depth
//! (docs/outlines/20261023-adaptive-cloud-probes.md), or of a dense grid. A point finds
//! its finest cell's leaf and interpolates that leaf's eight corners. The cells' depths
//! inside the cloud come from a `DepthSource`: src/render/LayoutDepths.zig measures them
//! on the GPU, and `Occupancy` on the CPU for the tests.

const ProbeLayout = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const CloudVolume = @import("../assets/CloudVolume.zig");
const gpu = @import("../gpu.zig");

/// Lattice point zero and the finest cells' edges, cloud units.
origin: [3]f32,
spacing: [3]f32,
/// Finest cells per axis.
cells: [3]u32,
/// Per probe, its position in cloud units; w unused.
positions: [][4]f32,
/// Per finest cell, x fastest: its leaf as `Leaf`, or `no_leaf` where it holds no cloud.
cell_leaves: []u32,
/// Per leaf, the probes at its corners: x in bit 0 of the corner's index, y in bit 1,
/// z in bit 2.
corners: [][8]u32,

pub const no_leaf = std.math.maxInt(u32);

/// A leaf of level `level` spans 2^level finest cells per axis, aligned to multiples of
/// that; `slot` is its index in `corners`.
pub const Leaf = packed struct(u32) { slot: u27, level: u5 };

/// Leaves span at most 2^level_max finest cells per axis.
pub const level_max = 6;

pub fn deinit(layout: *ProbeLayout, gpa: Allocator) void {
    gpa.free(layout.positions);
    gpa.free(layout.cell_leaves);
    gpa.free(layout.corners);
    layout.* = undefined;
}

/// The volume's occupancy grid, at its distance texture's resolution.
pub const Grid = struct {
    size: [3]u32,
    texel: f32,
    min_corner: [3]f32,
    /// The voxel grid's, cloud units.
    extent: [3]f32,

    pub fn init(volume: *const CloudVolume) Grid {
        var extent: [3]f32 = undefined;
        for (&extent, volume.size) |*e, s| e.* = @as(f32, @floatFromInt(s)) * volume.voxel;
        return .{ .size = volume.distance_size, .texel = CloudVolume.distance_downsample * volume.voxel, .min_corner = volume.min_corner, .extent = extent };
    }
};

/// Measures the `Depths` of the finest cells of edge `h0`, `cells` per axis.
pub const DepthSource = struct {
    context: *const anyopaque,
    measure: *const fn (context: *const anyopaque, gpa: Allocator, io: Io, h0: f32, cells: [3]u32) BuildError!Depths,
};

/// The occupancy on the CPU, the reference of LayoutDepths.zig's: per texel of `grid`,
/// the distance inside the cloud's surface, or -1 where none of the density that the
/// march's trilinear filter reads from within the texel is above zero.
pub const Occupancy = struct {
    grid: Grid,
    values: []f16,

    pub fn init(gpa: Allocator, volume: *const CloudVolume) Allocator.Error!Occupancy {
        const d = CloudVolume.distance_downsample;
        const size = volume.distance_size;
        const values = try gpa.alloc(f16, @as(usize, size[0]) * size[1] * size[2]);
        errdefer gpa.free(values);
        const plane_len = @as(usize, size[0]) * size[1];
        // Each voxel slice falls in the footprints of two texel slices; the ring keeps
        // the slices of the current footprint reduced to the texels' rows and columns.
        var ring: [d + 2][]bool = undefined;
        var ring_slice: [d + 2]?u32 = @splat(null);
        for (&ring, 0..) |*plane, i| {
            errdefer for (ring[0..i]) |done| gpa.free(done);
            plane.* = try gpa.alloc(bool, plane_len);
        }
        defer for (ring) |plane| gpa.free(plane);
        const row = try gpa.alloc(bool, @as(usize, size[0]) * volume.size[1]);
        defer gpa.free(row);

        for (0..size[2]) |tz| {
            const plane = values[tz * plane_len ..][0..plane_len];
            @memset(plane, -1);
            const z_window = footprint(@intCast(tz), volume.size[2]);
            for (z_window[0]..z_window[1]) |vz| {
                const slot = vz % ring.len;
                if (ring_slice[slot] != @as(u32, @intCast(vz))) {
                    reduceSlice(volume, @intCast(vz), size, row, ring[slot]);
                    ring_slice[slot] = @intCast(vz);
                }
                for (plane, ring[slot], volume.distance[tz * plane_len ..][0..plane_len]) |*value, occupied, distance| {
                    if (occupied) value.* = distance;
                }
            }
        }
        return .{ .grid = .init(volume), .values = values };
    }

    pub fn depthSource(occupancy: *const Occupancy) DepthSource {
        return .{ .context = occupancy, .measure = measure };
    }

    fn measure(context: *const anyopaque, gpa: Allocator, io: Io, h0: f32, cells: [3]u32) BuildError!Depths {
        return cellDepths(gpa, io, @ptrCast(@alignCast(context)), h0, cells);
    }

    pub fn deinit(occupancy: *Occupancy, gpa: Allocator) void {
        gpa.free(occupancy.values);
        occupancy.* = undefined;
    }

    /// Whether any voxel of slice `vz` within each texel column's footprint holds density.
    fn reduceSlice(volume: *const CloudVolume, vz: u32, size: [3]u32, row: []bool, out: []bool) void {
        const nx: usize = volume.size[0];
        const ny: usize = volume.size[1];
        const slice = volume.density[vz * nx * ny ..][0 .. nx * ny];
        for (0..ny) |y| {
            const voxels = slice[y * nx ..][0..nx];
            for (0..size[0]) |tx| {
                const window = footprint(@intCast(tx), volume.size[0]);
                var any = false;
                for (voxels[window[0]..window[1]]) |density| any = any or density > 0;
                row[y * size[0] + tx] = any;
            }
        }
        for (0..size[1]) |ty| {
            const window = footprint(@intCast(ty), volume.size[1]);
            for (0..size[0]) |tx| {
                var any = false;
                for (window[0]..window[1]) |y| any = any or row[y * size[0] + tx];
                out[ty * size[0] + tx] = any;
            }
        }
    }

    /// The voxels along an axis of `count` that the trilinear filter reads from points in
    /// texel `t`: its own and one more on either side.
    fn footprint(t: u32, count: u32) [2]u32 {
        const d = CloudVolume.distance_downsample;
        return .{ (d * t) -| 1, @min(d * t + d + 1, count) };
    }
};

/// `resolution` probes per axis, at the centres of as many equal cells of the box, x
/// fastest; a point beyond the outermost ones takes theirs.
pub fn dense(gpa: Allocator, grid: Grid, resolution: u32) Allocator.Error!ProbeLayout {
    var spacing: [3]f32 = undefined;
    var origin: [3]f32 = undefined;
    for (&spacing, &origin, grid.extent, grid.min_corner) |*s, *o, extent, min| {
        s.* = extent / @as(f32, @floatFromInt(resolution));
        o.* = min + 0.5 * s.*;
    }
    const n: usize = resolution;
    const cells: [3]u32 = @splat(resolution - 1);
    const positions = try gpa.alloc([4]f32, n * n * n);
    errdefer gpa.free(positions);
    for (positions, 0..) |*position, i| {
        const lattice = [3]usize{ i % n, i / n % n, i / (n * n) };
        for (0..3) |a| position[a] = origin[a] + @as(f32, @floatFromInt(lattice[a])) * spacing[a];
        position[3] = 0;
    }
    const m: usize = resolution - 1;
    const cell_leaves = try gpa.alloc(u32, m * m * m);
    errdefer gpa.free(cell_leaves);
    const corners = try gpa.alloc([8]u32, m * m * m);
    for (cell_leaves, corners, 0..) |*leaf, *corner, i| {
        leaf.* = @bitCast(Leaf{ .slot = @intCast(i), .level = 0 });
        const cell = [3]usize{ i % m, i / m % m, i / (m * m) };
        for (corner, 0..) |*probe, bits| {
            const x = cell[0] + (bits & 1);
            const y = cell[1] + (bits >> 1 & 1);
            const z = cell[2] + (bits >> 2);
            probe.* = @intCast(x + n * (y + n * z));
        }
    }
    return .{ .origin = origin, .spacing = spacing, .cells = cells, .positions = positions, .cell_leaves = cell_leaves, .corners = corners };
}

/// The adaptive layout with the most probes within `budget`: a cell may be of level L
/// where h0 2^L is at most `depth_ratio` times its least distance inside the cloud, and
/// h0, the finest cells' edge, is searched for. The count falls with h0 nearly as a
/// power of it, so the search takes secant steps in the logarithms, from the dense
/// grid's spacing, aiming a little under the budget.
pub fn adaptive(gpa: Allocator, io: Io, grid: Grid, depths: DepthSource, budget: u32, depth_ratio: f32) BuildError!ProbeLayout {
    var best: ?ProbeLayout = null;
    errdefer if (best) |*layout| layout.deinit(gpa);
    const extent_max = @reduce(.Max, @as(@Vector(3, f32), grid.extent));
    const target = @log(search_aim * @as(f64, @floatFromInt(budget)));
    var previous: ?[2]f64 = null;
    var log_h: f64 = @log(extent_max / 64);
    for (0..search_steps) |_| {
        // Below a texel, a texel could overlap three cells.
        const h0: f32 = @max(@as(f32, @floatCast(@exp(log_h))), grid.texel);
        var layout = try build(gpa, io, grid, depths, h0, depth_ratio);
        const count = layout.positions.len;
        if (count <= budget and (best == null or count > best.?.positions.len)) {
            if (best) |*old| old.deinit(gpa);
            best = layout;
        } else layout.deinit(gpa);
        if (best) |found| {
            if (@as(f64, @floatFromInt(found.positions.len)) >= search_tolerance * @as(f64, @floatFromInt(budget))) break;
            if (found.spacing[0] == grid.texel) break;
        }
        const point = [2]f64{ @log(@as(f64, h0)), @log(@as(f64, @floatFromInt(@max(count, 1)))) };
        var slope: f64 = search_slope;
        if (previous) |last| {
            const measured = (point[1] - last[1]) / (point[0] - last[0]);
            if (measured < search_slope_min) slope = measured;
        }
        previous = point;
        log_h = point[0] + (target - point[1]) / slope;
    }
    return best orelse build(gpa, io, grid, depths, extent_max / 4, depth_ratio);
}

pub const BuildError = Allocator.Error || Io.Cancelable || gpu.Error;

const search_steps = 8;
/// A layout within this share of the budget ends the search.
const search_tolerance = 0.97;
const search_aim = 0.985;
/// The count's power of h0 before two trials measure it: the surface's cells
/// dominate, about h0^-2, and the interior adds a little.
const search_slope = -2.3;
/// A secant flatter than this is noise in the count, not its trend.
const search_slope_min = -0.5;

/// The adaptive layout for finest cells of edge `h0`.
fn build(gpa: Allocator, io: Io, grid: Grid, source: DepthSource, h0: f32, depth_ratio: f32) BuildError!ProbeLayout {
    assert(h0 >= grid.texel);
    var cells: [3]u32 = undefined;
    for (&cells, grid.extent) |*n, extent| n.* = @intFromFloat(@ceil(extent / h0));
    var depths = try source.measure(source.context, gpa, io, h0, cells);
    defer depths.deinit(gpa);

    // Levels per finest cell, padded to whole roots, and their minima per block of
    // 2^l cells for each level l: a block is a leaf when no cell in it needs finer.
    const root = 1 << level_max;
    var padded: [3]u32 = undefined;
    for (&padded, cells) |*p, n| p.* = std.mem.alignForward(u32, n, root);
    var pyramid: [level_max + 1][]u8 = undefined;
    var built: usize = 0;
    defer for (pyramid[0..built]) |level| gpa.free(level);
    pyramid[0] = try gpa.alloc(u8, @as(usize, padded[0]) * padded[1] * padded[2]);
    built = 1;
    @memset(pyramid[0], not_needed);
    const log_ratio = std.math.log2(depth_ratio / h0);
    for (0..cells[2]) |z| for (0..cells[1]) |y| for (0..cells[0]) |x| {
        const cell = x + cells[0] * (y + cells[1] * z);
        const depth = depths.depth[cell];
        pyramid[0][x + padded[0] * (y + padded[1] * z)] = if (!depths.occupied[cell])
            not_needed
        else if (depth == 0)
            0
        else
            @intFromFloat(std.math.clamp(@floor(log_ratio + std.math.log2(depth)), 0, level_max));
    };
    for (1..level_max + 1) |l| {
        const size = blockCount(padded, l);
        const finer = blockCount(padded, l - 1);
        pyramid[l] = try gpa.alloc(u8, @as(usize, size[0]) * size[1] * size[2]);
        built += 1;
        for (0..size[2]) |z| for (0..size[1]) |y| for (0..size[0]) |x| {
            var least: u8 = not_needed;
            for (0..8) |bits| {
                const fx = 2 * x + (bits & 1);
                const fy = 2 * y + (bits >> 1 & 1);
                const fz = 2 * z + (bits >> 2);
                least = @min(least, pyramid[l - 1][fx + finer[0] * (fy + finer[1] * fz)]);
            }
            pyramid[l][x + size[0] * (y + size[1] * z)] = least;
        };
    }

    var builder: Builder = .{ .cells = cells, .points = .{ padded[0] + 1, padded[1] + 1, padded[2] + 1 } };
    errdefer builder.deinit(gpa);
    builder.cell_leaves = try gpa.alloc(u32, @as(usize, cells[0]) * cells[1] * cells[2]);
    @memset(builder.cell_leaves, no_leaf);
    builder.probes = try gpa.alloc(u32, @as(usize, builder.points[0]) * builder.points[1] * builder.points[2]);
    @memset(builder.probes, no_probe);
    const roots = blockCount(padded, level_max);
    for (0..roots[2]) |z| for (0..roots[1]) |y| for (0..roots[0]) |x| {
        try builder.visit(gpa, &pyramid, padded, level_max, .{ @intCast(x), @intCast(y), @intCast(z) });
    };

    const origin = grid.min_corner;
    const positions = try gpa.alloc([4]f32, builder.lattice.items.len);
    errdefer gpa.free(positions);
    for (positions, builder.lattice.items) |*position, point| {
        for (0..3) |a| position[a] = origin[a] + @as(f32, @floatFromInt(point[a])) * h0;
        position[3] = 0;
    }
    const corners = try builder.corners.toOwnedSlice(gpa);
    gpa.free(builder.probes);
    builder.lattice.deinit(gpa);
    return .{
        .origin = origin,
        .spacing = @splat(h0),
        .cells = cells,
        .positions = positions,
        .cell_leaves = builder.cell_leaves,
        .corners = corners,
    };
}

const not_needed = std.math.maxInt(u8);
const no_probe = std.math.maxInt(u32);

fn blockCount(padded: [3]u32, level: usize) [3]u32 {
    return .{ padded[0] >> @intCast(level), padded[1] >> @intCast(level), padded[2] >> @intCast(level) };
}

const Builder = struct {
    cells: [3]u32,
    /// Lattice points per axis, of the padded roots' cells.
    points: [3]u32,
    cell_leaves: []u32 = &.{},
    corners: std.ArrayList([8]u32) = .empty,
    /// Probe per lattice point, x fastest, numbered in the order the leaves reach them,
    /// and the lattice point per probe.
    probes: []u32 = &.{},
    lattice: std.ArrayList([3]u32) = .empty,

    fn deinit(builder: *Builder, gpa: Allocator) void {
        gpa.free(builder.cell_leaves);
        builder.corners.deinit(gpa);
        gpa.free(builder.probes);
        builder.lattice.deinit(gpa);
    }

    /// Block `block` of level `level`, in blocks of 2^level finest cells.
    fn visit(builder: *Builder, gpa: Allocator, pyramid: *const [level_max + 1][]u8, padded: [3]u32, level: u5, block: [3]u32) Allocator.Error!void {
        const size = blockCount(padded, level);
        const least = pyramid[level][block[0] + size[0] * (block[1] + size[1] * block[2])];
        if (least == not_needed) return;
        if (least >= level) return builder.leaf(gpa, level, block);
        for (0..8) |bits| {
            const child = [3]u32{ 2 * block[0] + @as(u32, @intCast(bits & 1)), 2 * block[1] + @as(u32, @intCast(bits >> 1 & 1)), 2 * block[2] + @as(u32, @intCast(bits >> 2)) };
            try builder.visit(gpa, pyramid, padded, level - 1, child);
        }
    }

    fn leaf(builder: *Builder, gpa: Allocator, level: u5, block: [3]u32) Allocator.Error!void {
        const slot: u27 = @intCast(builder.corners.items.len);
        const span = @as(u32, 1) << level;
        const first = [3]u32{ block[0] * span, block[1] * span, block[2] * span };
        var corner: [8]u32 = undefined;
        for (&corner, 0..) |*probe, bits| {
            const point = [3]u32{ first[0] + span * @as(u32, @intCast(bits & 1)), first[1] + span * @as(u32, @intCast(bits >> 1 & 1)), first[2] + span * @as(u32, @intCast(bits >> 2)) };
            const entry = &builder.probes[point[0] + builder.points[0] * (point[1] + builder.points[1] * point[2])];
            if (entry.* == no_probe) {
                entry.* = @intCast(builder.lattice.items.len);
                try builder.lattice.append(gpa, point);
            }
            probe.* = entry.*;
        }
        try builder.corners.append(gpa, corner);
        const encoded: u32 = @bitCast(Leaf{ .slot = slot, .level = level });
        const n = builder.cells;
        for (first[2]..@min(first[2] + span, n[2])) |z| for (first[1]..@min(first[1] + span, n[1])) |y| {
            const row = builder.cell_leaves[n[0] * (y + n[1] * z) ..][0..n[0]];
            @memset(row[@min(first[0], n[0])..@min(first[0] + span, n[0])], encoded);
        };
    }
};

/// Per cell of a grid, x fastest: whether it holds cloud, and the least distance
/// inside the cloud over it, zero where it also holds air.
pub const Depths = struct {
    occupied: []bool,
    depth: []f32,

    pub fn init(gpa: Allocator, size: [3]u32) Allocator.Error!Depths {
        const count = @as(usize, size[0]) * size[1] * size[2];
        const occupied = try gpa.alloc(bool, count);
        errdefer gpa.free(occupied);
        @memset(occupied, false);
        const depth = try gpa.alloc(f32, count);
        @memset(depth, std.math.inf(f32));
        return .{ .occupied = occupied, .depth = depth };
    }

    pub fn deinit(depths: *Depths, gpa: Allocator) void {
        gpa.free(depths.occupied);
        gpa.free(depths.depth);
        depths.* = undefined;
    }
};

/// `Depths` of the finest cells of edge `h0`, reduced from the occupancy one axis at a
/// time. A texel counts in every cell it overlaps, as points of each lie in it.
fn cellDepths(gpa: Allocator, io: Io, occupancy: *const Occupancy, h0: f32, cells: [3]u32) BuildError!Depths {
    const t = occupancy.grid.size;
    const texel = occupancy.grid.texel;
    var x_pass = try reduceAxis(true, gpa, io, occupancy, t, 0, cells[0], texel, h0);
    defer x_pass.deinit(gpa);
    var y_pass = try reduceAxis(false, gpa, io, &x_pass, .{ cells[0], t[1], t[2] }, 1, cells[1], texel, h0);
    defer y_pass.deinit(gpa);
    return reduceAxis(false, gpa, io, &y_pass, .{ cells[0], cells[1], t[2] }, 2, cells[2], texel, h0);
}

/// The first and last of `cell_count` cells of edge `h0` that texel `i` overlaps: a
/// texel no wider than a cell overlaps at most two.
fn cellsOfTexel(i: usize, texel: f32, h0: f32, cell_count: u32) [2]u32 {
    const start = @as(f32, @floatFromInt(i)) * texel;
    return .{
        @min(@as(u32, @intFromFloat(start / h0)), cell_count - 1),
        @min(@as(u32, @intFromFloat(@ceil((start + texel) / h0))) -| 1, cell_count - 1),
    };
}

/// Per cell of edge `h0` along x, then y, then z, the first and last texel of `grid`
/// that overlaps it, as `cellDepths` counts them.
pub fn texelRanges(gpa: Allocator, grid: Grid, h0: f32, cells: [3]u32) Allocator.Error![][2]u32 {
    const ranges = try gpa.alloc([2]u32, cells[0] + cells[1] + cells[2]);
    @memset(ranges, .{ std.math.maxInt(u32), 0 });
    var first: usize = 0;
    for (grid.size, cells) |texels, cell_count| {
        const axis = ranges[first..][0..cell_count];
        for (0..texels) |i| {
            const overlapped = cellsOfTexel(i, grid.texel, h0, cell_count);
            for (overlapped[0]..overlapped[1] + 1) |cell| {
                axis[cell][0] = @min(axis[cell][0], @as(u32, @intCast(i)));
                axis[cell][1] = @max(axis[cell][1], @as(u32, @intCast(i)));
            }
        }
        first += cell_count;
    }
    return ranges;
}

fn reduceAxis(
    comptime from_occupancy: bool,
    gpa: Allocator,
    io: Io,
    source: if (from_occupancy) *const Occupancy else *const Depths,
    size: [3]u32,
    axis: usize,
    cell_count: u32,
    texel: f32,
    h0: f32,
) BuildError!Depths {
    var out_size = size;
    out_size[axis] = cell_count;
    var out: Depths = try .init(gpa, out_size);
    errdefer out.deinit(gpa);
    const cells_of = try gpa.alloc([2]u32, size[axis]);
    defer gpa.free(cells_of);
    for (cells_of, 0..) |*cells, i| cells.* = cellsOfTexel(i, texel, h0, cell_count);
    // Bands of the outermost axis that is kept write cells of their own.
    const band_axis: usize = if (axis == 2) 1 else 2;
    const band = std.math.divCeil(u32, size[band_axis], reduce_bands) catch unreachable;
    var group: Io.Group = .init;
    var first: u32 = 0;
    while (first < size[band_axis]) : (first += band) {
        group.async(io, BandReducer(from_occupancy).run, .{ source, &out, size, out_size, axis, cells_of, band_axis, first, @min(first + band, size[band_axis]) });
    }
    try group.await(io);
    return out;
}

const reduce_bands = 64;

fn BandReducer(comptime from_occupancy: bool) type {
    return struct {
        fn run(
            source: if (from_occupancy) *const Occupancy else *const Depths,
            out: *Depths,
            size: [3]u32,
            out_size: [3]u32,
            axis: usize,
            cells_of: []const [2]u32,
            band_axis: usize,
            first: u32,
            end: u32,
        ) void {
            var lower: [3]u32 = @splat(0);
            var upper = size;
            lower[band_axis] = first;
            upper[band_axis] = end;
            for (lower[2]..upper[2]) |z| for (lower[1]..upper[1]) |y| for (0..size[0]) |x| {
                const index = x + size[0] * (y + size[1] * z);
                var occupied: bool = undefined;
                var depth: f32 = undefined;
                if (from_occupancy) {
                    const value: f32 = source.values[index];
                    occupied = value >= 0;
                    // Air is at depth zero.
                    depth = @max(value, 0);
                } else {
                    occupied = source.occupied[index];
                    depth = source.depth[index];
                }
                var coord = [3]usize{ x, y, z };
                const cells = cells_of[coord[axis]];
                for (cells[0]..cells[1] + 1) |cell| {
                    coord[axis] = cell;
                    const o = coord[0] + out_size[0] * (coord[1] + out_size[1] * coord[2]);
                    out.occupied[o] = out.occupied[o] or occupied;
                    out.depth[o] = @min(out.depth[o], depth);
                }
            };
        }
    };
}

const testing = std.testing;

/// A ball of cloud of `radius` voxels in a cube of `side` voxels, its distance texture
/// the exact distance to its sphere inside.
const Ball = struct {
    density: []f16,
    distance: []f16,
    volume: CloudVolume,

    fn init(gpa: Allocator, side: u32, radius: f32) !Ball {
        const d = CloudVolume.distance_downsample;
        const t = side / d;
        const density = try gpa.alloc(f16, side * side * side);
        errdefer gpa.free(density);
        const distance = try gpa.alloc(f16, t * t * t);
        const centre = @as(f32, @floatFromInt(side)) / 2;
        for (density, 0..) |*value, i| {
            const r = length(.{ @floatFromInt(i % side), @floatFromInt(i / side % side), @floatFromInt(i / (side * side)) }, 0.5, centre);
            value.* = if (r < radius) 1 else 0;
        }
        for (distance, 0..) |*value, i| {
            const r = length(.{ @floatFromInt(i % t), @floatFromInt(i / t % t), @floatFromInt(i / (t * t)) }, 0.5, centre / d) * d;
            value.* = @floatCast(@max(radius - r, 0));
        }
        return .{ .density = density, .distance = distance, .volume = .{
            .size = @splat(side),
            .cell = 8,
            .cells = @splat(side / 8),
            .distance_size = @splat(t),
            .voxel = 1,
            .min_corner = @splat(0),
            .density = density,
            .majorant = &.{},
            .distance = distance,
            .file = &.{},
        } };
    }

    fn length(index: [3]f32, offset: f32, centre: f32) f32 {
        var sum: f32 = 0;
        for (index) |i| sum += (i + offset - centre) * (i + offset - centre);
        return @sqrt(sum);
    }

    fn deinit(ball: *Ball, gpa: Allocator) void {
        gpa.free(ball.density);
        gpa.free(ball.distance);
    }
};

test "every cell that holds cloud has a leaf whose corners are its own" {
    const gpa = testing.allocator;
    var ball: Ball = try .init(gpa, 64, 24);
    defer ball.deinit(gpa);
    var occupancy: Occupancy = try .init(gpa, &ball.volume);
    defer occupancy.deinit(gpa);
    var layout = try build(gpa, testing.io, occupancy.grid, occupancy.depthSource(), 2, 2.5);
    defer layout.deinit(gpa);
    const n = layout.cells;
    var leaves_by_level: [level_max + 1]u32 = @splat(0);
    for (0..n[2]) |z| for (0..n[1]) |y| for (0..n[0]) |x| {
        const centre = [3]f32{ (@as(f32, @floatFromInt(x)) + 0.5) * 2, (@as(f32, @floatFromInt(y)) + 0.5) * 2, (@as(f32, @floatFromInt(z)) + 0.5) * 2 };
        const entry = layout.cell_leaves[x + n[0] * (y + n[1] * z)];
        if (Ball.length(centre, 0, 32) < 24) try testing.expect(entry != no_leaf);
        if (entry == no_leaf) continue;
        const leaf: Leaf = @bitCast(entry);
        if (x % (@as(u32, 1) << leaf.level) == 0 and y % (@as(u32, 1) << leaf.level) == 0 and z % (@as(u32, 1) << leaf.level) == 0)
            leaves_by_level[leaf.level] += 1;
        // The leaf's first corner lies at its aligned origin, its last a span further.
        const span = @as(f32, @floatFromInt(@as(u32, 1) << leaf.level)) * 2;
        const first = layout.positions[layout.corners[leaf.slot][0]];
        const last = layout.positions[layout.corners[leaf.slot][7]];
        for (0..3) |a| {
            try testing.expectApproxEqAbs(span, last[a] - first[a], 1e-4);
            try testing.expect(centre[a] >= first[a] and centre[a] <= last[a]);
        }
    };
    // The ball's core, deeper than its surface's cells, takes coarser leaves.
    try testing.expect(leaves_by_level[0] > 0 and leaves_by_level[1] > 0);
}

test "leaves sharing a corner share its probe" {
    const gpa = testing.allocator;
    var ball: Ball = try .init(gpa, 32, 12);
    defer ball.deinit(gpa);
    var occupancy: Occupancy = try .init(gpa, &ball.volume);
    defer occupancy.deinit(gpa);
    var layout = try build(gpa, testing.io, occupancy.grid, occupancy.depthSource(), 2, 2.5);
    defer layout.deinit(gpa);
    var seen: std.AutoHashMapUnmanaged([3]u32, void) = .empty;
    defer seen.deinit(gpa);
    for (layout.positions) |position| {
        const key: [3]u32 = .{ @bitCast(position[0]), @bitCast(position[1]), @bitCast(position[2]) };
        try testing.expect(!(try seen.getOrPut(gpa, key)).found_existing);
    }
}

test "the adaptive layout keeps within its budget, and a larger one refines it" {
    const gpa = testing.allocator;
    var ball: Ball = try .init(gpa, 64, 24);
    defer ball.deinit(gpa);
    var occupancy: Occupancy = try .init(gpa, &ball.volume);
    defer occupancy.deinit(gpa);
    var small = try adaptive(gpa, testing.io, occupancy.grid, occupancy.depthSource(), 2000, 2.5);
    defer small.deinit(gpa);
    var large = try adaptive(gpa, testing.io, occupancy.grid, occupancy.depthSource(), 8000, 2.5);
    defer large.deinit(gpa);
    try testing.expect(small.positions.len <= 2000);
    try testing.expect(large.positions.len <= 8000);
    try testing.expect(large.spacing[0] < small.spacing[0]);
}

test "the dense layout's probes are its cells' corners" {
    const gpa = testing.allocator;
    var ball: Ball = try .init(gpa, 16, 6);
    defer ball.deinit(gpa);
    var occupancy: Occupancy = try .init(gpa, &ball.volume);
    defer occupancy.deinit(gpa);
    var layout = try dense(gpa, occupancy.grid, 4);
    defer layout.deinit(gpa);
    try testing.expectEqual(64, layout.positions.len);
    try testing.expectEqual(27, layout.corners.len);
    try testing.expectEqualSlices(f32, &.{ 2, 2, 2, 0 }, &layout.positions[0]);
    try testing.expectEqualSlices(f32, &.{ 14, 14, 14, 0 }, &layout.positions[63]);
    try testing.expectEqual(@as(u32, 63), layout.corners[26][7]);
}

test "the GPU's depths, and so its layouts, are the CPU's" {
    const LayoutDepths = @import("LayoutDepths.zig");
    const uploadVolume = @import("CloudBench.zig").uploadVolume;
    const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;
    const device = try createTestDevice();
    defer device.deinit();
    const gpa = testing.allocator;
    var ball: Ball = try .init(gpa, 64, 24);
    defer ball.deinit(gpa);
    var occupancy: Occupancy = try .init(gpa, &ball.volume);
    defer occupancy.deinit(gpa);
    const density = try uploadVolume(device, ball.volume.size, ball.density, .{ .shader_resource = true }, "ball density");
    defer density.deinit();
    var layout_depths: LayoutDepths = try .init(device, &ball.volume, density);
    defer layout_depths.deinit();
    const cpu = occupancy.depthSource();
    const on_gpu = layout_depths.source();
    for ([_]f32{ 2, 2.6, 3.3, 8 }) |h0| {
        var cells: [3]u32 = undefined;
        for (&cells, occupancy.grid.extent) |*n, extent| n.* = @intFromFloat(@ceil(extent / h0));
        var expected = try cpu.measure(cpu.context, gpa, testing.io, h0, cells);
        defer expected.deinit(gpa);
        var measured = try on_gpu.measure(on_gpu.context, gpa, testing.io, h0, cells);
        defer measured.deinit(gpa);
        try testing.expectEqualSlices(bool, expected.occupied, measured.occupied);
        for (expected.occupied, expected.depth, measured.depth) |occupied, a, b| {
            if (occupied) try testing.expectEqual(a, b);
        }
        var from_cpu = try build(gpa, testing.io, occupancy.grid, cpu, h0, 2.5);
        defer from_cpu.deinit(gpa);
        var from_gpu = try build(gpa, testing.io, layout_depths.grid, on_gpu, h0, 2.5);
        defer from_gpu.deinit(gpa);
        try testing.expectEqualSlices([4]f32, from_cpu.positions, from_gpu.positions);
        try testing.expectEqualSlices([8]u32, from_cpu.corners, from_gpu.corners);
        try testing.expectEqualSlices(u32, from_cpu.cell_leaves, from_gpu.cell_leaves);
    }
}
