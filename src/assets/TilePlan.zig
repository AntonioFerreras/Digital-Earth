//! Splits an equirectangular map wider than the device's 2D texture limit into
//! equal longitude tiles stored as layers of a `Texture2DArray`.
//!
//! Each layer holds `tile_width` columns plus `apron` columns copied from each
//! neighbour, wrapping across the u = 0/1 seam, so bilinear filtering inside a
//! layer matches filtering the whole map. A map that fits is one tile with no
//! apron, and the sampler's REPEAT addressing in u wraps the seam instead.
//!
//! Shader mapping from map coordinate u (any real, repeating) to a layer:
//!
//!     x     = frac(u) * map_width                  // texel units, texel i spans [i, i+1)
//!     layer = min(floor(x / tile_width), tile_count - 1)
//!     u_layer = (x - layer * tile_width + apron) / layer_width
//!
//! v passes through unchanged. Sample with linear filtering, REPEAT in u and
//! CLAMP in v. Bilinear taps at u_layer reach at most half a texel past the
//! tile, which lands inside the one-texel apron. The last tile may run past
//! map_width; those columns wrap to the start of the map and are never the
//! nearest texel for any u, only filter taps across the seam.

const TilePlan = @This();

const std = @import("std");
const assert = std.debug.assert;

map_width: u32,
map_height: u32,
tile_count: u32,
/// Map columns owned by each tile; `tile_count * tile_width >= map_width`.
tile_width: u32,
/// Columns copied from each neighbouring tile on each side: 0 or 1.
apron: u32,
/// Texture array layer width, `tile_width + 2 * apron`, at most the device limit.
layer_width: u32,

pub const PlanError = error{MapTooTall};

/// Plans the fewest equal tiles whose layers fit in `dimension_max`, the
/// device's maximum 2D texture width and height.
pub fn plan(map_width: u32, map_height: u32, dimension_max: u32) PlanError!TilePlan {
    assert(map_width > 0);
    assert(map_height > 0);
    assert(dimension_max >= 3);
    if (map_height > dimension_max) return error.MapTooTall;

    const result: TilePlan = if (map_width <= dimension_max) .{
        .map_width = map_width,
        .map_height = map_height,
        .tile_count = 1,
        .tile_width = map_width,
        .apron = 0,
        .layer_width = map_width,
    } else blk: {
        const apron = 1;
        const tile_count = std.math.divCeil(u32, map_width, dimension_max - 2 * apron) catch unreachable;
        const tile_width = std.math.divCeil(u32, map_width, tile_count) catch unreachable;
        break :blk .{
            .map_width = map_width,
            .map_height = map_height,
            .tile_count = tile_count,
            .tile_width = tile_width,
            .apron = apron,
            .layer_width = tile_width + 2 * apron,
        };
    };
    assert(result.layer_width <= dimension_max);
    assert(result.tile_count * result.tile_width >= map_width);
    assert((result.tile_count - 1) * result.tile_width < map_width);
    return result;
}

/// Map column held by column 0 of `tile`'s layer.
pub fn sourceColumnFirst(tile_plan: TilePlan, tile: u32) u32 {
    assert(tile < tile_plan.tile_count);
    return (tile * tile_plan.tile_width + tile_plan.map_width - tile_plan.apron) % tile_plan.map_width;
}

/// Copies rows `row_first..` of `tile`'s layer out of a whole map with `channels`
/// bytes per texel, wrapping columns across the seam. `layer` holds whole rows of
/// `layer_width` texels, as many as fit, so a full layer is `layer_width x map_height`.
pub fn copyTile(tile_plan: TilePlan, tile: u32, channels: u32, map: []const u8, row_first: u32, layer: []u8) void {
    const map_stride = @as(usize, tile_plan.map_width) * channels;
    const layer_stride = @as(usize, tile_plan.layer_width) * channels;
    assert(map.len == map_stride * tile_plan.map_height);
    assert(layer.len % layer_stride == 0);
    const row_count = layer.len / layer_stride;
    assert(row_first + row_count <= tile_plan.map_height);

    const column_first = tile_plan.sourceColumnFirst(tile);
    for (0..row_count) |layer_row_index| {
        const map_row = map[(row_first + layer_row_index) * map_stride ..][0..map_stride];
        const layer_row = layer[layer_row_index * layer_stride ..][0..layer_stride];
        // Runs of contiguous map columns; a layer is narrower than the map
        // whenever it is split, so at most two runs cross the seam.
        var column = column_first;
        var layer_offset: usize = 0;
        for (0..2) |_| {
            if (layer_offset == layer_stride) break;
            const run_size = @min(layer_stride - layer_offset, map_stride - @as(usize, column) * channels);
            @memcpy(layer_row[layer_offset..][0..run_size], map_row[@as(usize, column) * channels ..][0..run_size]);
            layer_offset += run_size;
            column = 0;
        }
        assert(layer_offset == layer_stride);
    }
}

test "21600-wide maps split in two at 16384 and stay whole at 32768" {
    const split = try plan(21600, 10800, 16384);
    try std.testing.expectEqual(2, split.tile_count);
    try std.testing.expectEqual(10800, split.tile_width);
    try std.testing.expectEqual(1, split.apron);
    try std.testing.expectEqual(10802, split.layer_width);
    try std.testing.expectEqual(21599, split.sourceColumnFirst(0));
    try std.testing.expectEqual(10799, split.sourceColumnFirst(1));

    const whole = try plan(21600, 10800, 32768);
    try std.testing.expectEqual(1, whole.tile_count);
    try std.testing.expectEqual(0, whole.apron);
    try std.testing.expectEqual(21600, whole.layer_width);
    try std.testing.expectEqual(0, whole.sourceColumnFirst(0));
}

test "a map exactly at the limit is one tile, one texel over needs two" {
    try std.testing.expectEqual(1, (try plan(16384, 8192, 16384)).tile_count);
    const over = try plan(16385, 8192, 16384);
    try std.testing.expectEqual(2, over.tile_count);
    try std.testing.expectEqual(8193, over.tile_width);
}

test "odd widths round tiles up and every layer fits" {
    const widths = [_]u32{ 21601, 16383 * 3 + 1, 50000, 7, 99991 };
    for (widths) |width| {
        const tile_plan = try plan(width, 4, 16384);
        try std.testing.expect(tile_plan.layer_width <= 16384);
        try std.testing.expect(tile_plan.tile_count * tile_plan.tile_width >= width);
    }
    const prime = try plan(99991, 4, 16384);
    try std.testing.expectEqual(7, prime.tile_count);
    try std.testing.expectEqual(14285, prime.tile_width);
}

test "tall maps are rejected" {
    try std.testing.expectError(error.MapTooTall, plan(1024, 16385, 16384));
}

test "copyTile wraps aprons across the seam and copies row ranges" {
    // A 10-column map split into tiles of 4, 4 and 2 (+2 wrapped) columns by a limit of 6.
    const tile_plan = try plan(10, 2, 6);
    try std.testing.expectEqual(3, tile_plan.tile_count);
    try std.testing.expectEqual(4, tile_plan.tile_width);

    var map: [10 * 2]u8 = undefined;
    for (&map, 0..) |*texel, index| texel.* = @intCast((index / 10) * 100 + index % 10);
    var layer: [6 * 2]u8 = undefined;

    tile_plan.copyTile(0, 1, &map, 0, &layer);
    try std.testing.expectEqualSlices(u8, &.{ 9, 0, 1, 2, 3, 4, 109, 100, 101, 102, 103, 104 }, &layer);
    tile_plan.copyTile(2, 1, &map, 0, &layer);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 0, 1, 2, 107, 108, 109, 100, 101, 102 }, &layer);
    tile_plan.copyTile(2, 1, &map, 1, layer[0..6]);
    try std.testing.expectEqualSlices(u8, &.{ 107, 108, 109, 100, 101, 102 }, layer[0..6]);
}

test "copyTile copies multi-channel texels whole" {
    const tile_plan = try plan(4, 1, 3);
    try std.testing.expectEqual(4, tile_plan.tile_count);
    const map = [_]u8{ 0, 1, 10, 11, 20, 21, 30, 31 };
    var layer: [3 * 2]u8 = undefined;
    tile_plan.copyTile(0, 2, &map, 0, &layer);
    try std.testing.expectEqualSlices(u8, &.{ 30, 31, 0, 1, 10, 11 }, &layer);
    tile_plan.copyTile(3, 2, &map, 0, &layer);
    try std.testing.expectEqualSlices(u8, &.{ 20, 21, 30, 31, 0, 1 }, &layer);
}
