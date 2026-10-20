//! Coarse upper bounds of Earth maps, so that marches through the terrain and the
//! cloud shell skip regions where the full maps cannot produce a hit.
//!
//! A bound map is `width` x `height` cells in the cache layout (south row first,
//! columns west to east from u = 0). Cell (i, j) holds the largest texel of the source map that bilinear filtering can reach from any point of the
//! cells (i - 1 .. i + 1, j - 1 .. j + 1), the longitude wrapping and the latitude
//! clamped. Any point within the cell-sized angle of a query point lies in those
//! cells, which is what shaders/earth/map_bounds.slang relies on.

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const EarthMaps = @import("EarthMaps.zig");

/// The bound map's size, read by the shaders from `EarthMapUniforms.bounds`; every
/// source map must be at least this fine.
pub const width = 4096;
pub const height = 2048;

/// One bound map channel: the bounds of a one-channel map.
pub const Job = struct {
    header: EarthMaps.Header,
    pixels: []const u8,
    channel: u32,
};

/// Cell rows per task.
const band_height = 64;

/// Builds every job into its channel of the `channel_count`-channel cell array
/// `cells`, in bands of cell rows across the threads of `io`. `gpa` must be thread-safe.
pub fn build(
    gpa: std.mem.Allocator,
    io: Io,
    jobs: []const Job,
    cells: []u8,
    channel_count: u32,
) (error{OutOfMemory} || Io.Cancelable)!void {
    assert(cells.len == width * height * channel_count);
    for (jobs) |job| {
        assert(job.header.channels == 1);
        assert(job.header.width >= width and job.header.height >= height);
        assert(job.pixels.len == job.header.pixelsSize());
        assert(job.channel < channel_count);
    }
    // Each cell's own texels, plus the one-texel margins that bilinear filtering
    // reaches from the cell's points; then the neighbourhoods of those.
    const own = try gpa.alloc([width * height]u8, jobs.len);
    defer gpa.free(own);
    var reduced: Io.Group = .init;
    for (jobs, 0..) |job, job_index| {
        var j_first: usize = 0;
        while (j_first < height) : (j_first += band_height) {
            reduced.async(io, reduceBand, .{ job, &own[job_index], j_first, @min(j_first + band_height, height) });
        }
    }
    try reduced.await(io);
    var dilated: Io.Group = .init;
    for (jobs, 0..) |job, job_index| {
        var j_first: usize = 0;
        while (j_first < height) : (j_first += band_height) {
            dilated.async(io, dilateBand, .{ job, &own[job_index], cells, channel_count, j_first, @min(j_first + band_height, height) });
        }
    }
    try dilated.await(io);
}

/// Fills the cell rows `j_first` .. `j_end` of `own`.
fn reduceBand(job: Job, own: []u8, j_first: usize, j_end: usize) void {
    const header = job.header;
    @memset(own[j_first * width .. j_end * width], 0);
    var row_bound: [width]u8 = undefined;
    const row_first = (j_first * header.height / height) -| 1;
    const row_last = @min(j_end * header.height / height + 1, header.height - 1);
    for (row_first..row_last + 1) |row| {
        const texels = job.pixels[row * header.width ..][0..header.width];
        var first: usize = 0;
        for (&row_bound, 1..) |*value, i_next| {
            const next = i_next * header.width / width;
            // Texels first - 1 .. next + 1, wrapping across the seam.
            var bound = texels[if (first == 0) header.width - 1 else first - 1];
            for (texels[first..@min(next + 2, header.width)]) |texel| bound = @max(bound, texel);
            for (texels[0 .. (next + 2) -| header.width]) |texel| bound = @max(bound, texel);
            value.* = bound;
            first = next;
        }
        // A superset of the cell rows whose points filter this texel row.
        const j_low = @max(((row -| 1) * height / header.height) -| 1, j_first);
        const j_high: usize = @min((row + 2) * height / header.height + 1, j_end);
        for (j_low..j_high) |j| {
            if (!filters(j, row, header.height)) continue;
            for (own[j * width ..][0..width], &row_bound) |*value, bound| value.* = @max(value.*, bound);
        }
    }
}

/// Writes the cell rows `j_first` .. `j_end` of the job's channel of `cells` from `own`.
fn dilateBand(job: Job, own: []u8, cells: []u8, channel_count: u32, j_first: usize, j_end: usize) void {
    for (j_first..j_end) |j| {
        for (0..width) |i| {
            var value: u8 = 0;
            for (@max(j, 1) - 1..@min(j + 2, height)) |neighbour_j| {
                for ([_]usize{ (i + width - 1) % width, i, (i + 1) % width }) |neighbour_i| {
                    value = @max(value, own[neighbour_j * width + neighbour_i]);
                }
            }
            cells[(j * width + i) * channel_count + job.channel] = value;
        }
    }
}

/// Whether bilinear filtering reads texel row `row` of `texel_count` from a point of
/// cell row `j`, whose points lie in texel rows floor(j n / c) .. floor((j + 1) n / c),
/// the reach extending one row either side and clamping at the poles.
fn filters(j: usize, row: usize, texel_count: usize) bool {
    const first = j * texel_count / height;
    const next = (j + 1) * texel_count / height;
    const reach_first = first -| 1;
    const reach_last = @min(next + 1, texel_count - 1);
    // The clamped rows also stand in for the rows past the poles.
    return row >= reach_first and row <= reach_last;
}

const testing = std.testing;

test "a lone texel bounds its cell and the ring around it, wrapping in longitude" {
    const header: EarthMaps.Header = .init(2 * width, 2 * height, 1);
    const pixels = try testing.allocator.alloc(u8, header.pixelsSize());
    defer testing.allocator.free(pixels);
    @memset(pixels, 0);
    // Texel column 0 of row 2000: its filtering reaches cells across the seam.
    pixels[2000 * header.width] = 200;
    const cells = try testing.allocator.alloc(u8, width * height * 2);
    defer testing.allocator.free(cells);
    @memset(cells, 7);
    try build(testing.allocator, testing.io, &.{.{ .header = header, .pixels = pixels, .channel = 1 }}, cells, 2);

    var lit: usize = 0;
    for (0..height) |j| {
        for (0..width) |i| {
            const value = cells[(j * width + i) * 2 + 1];
            try testing.expectEqual(7, cells[(j * width + i) * 2]);
            if (value == 0) continue;
            try testing.expectEqual(200, value);
            lit += 1;
            try testing.expect(j >= 998 and j <= 1001);
            try testing.expect(i <= 1 or i >= width - 2);
        }
    }
    // Two cells of the texel's own span plus the one-cell ring, on both sides of the seam.
    try testing.expectEqual(4 * 4, lit);
}

test "the bound holds every bilinear sample within a cell of the query cell" {
    var prng: std.Random.DefaultPrng = .init(1);
    const random = prng.random();
    const header: EarthMaps.Header = .init(width + 517, height + 301, 1);
    const pixels = try testing.allocator.alloc(u8, header.pixelsSize());
    defer testing.allocator.free(pixels);
    for (pixels) |*pixel| pixel.* = if (random.uintLessThan(u8, 64) == 0) random.int(u8) else 0;
    const cells = try testing.allocator.alloc(u8, width * height);
    defer testing.allocator.free(cells);
    try build(testing.allocator, testing.io, &.{.{ .header = header, .pixels = pixels, .channel = 0 }}, cells, 1);

    const cell_size: [2]f64 = .{ 1.0 / @as(f64, width), 1.0 / @as(f64, height) };
    for (0..20000) |_| {
        const query: [2]f64 = .{ random.float(f64), random.float(f64) };
        // A sample anywhere in the query cell's neighbourhood.
        const u = query[0] + (random.float(f64) * 2 - 1) * cell_size[0];
        const v = std.math.clamp(query[1] + (random.float(f64) * 2 - 1) * cell_size[1], 0, 1);
        const i: usize = @intFromFloat(query[0] * width);
        const j: usize = @intFromFloat(query[1] * height);
        const bound = cells[j * width + i];
        const x = @mod(u, 1.0) * @as(f64, @floatFromInt(header.width)) - 0.5;
        const y = v * @as(f64, @floatFromInt(header.height)) - 0.5;
        const x0: i64 = @intFromFloat(@floor(x));
        const y0: i64 = @intFromFloat(@floor(y));
        for (0..2) |dy| {
            for (0..2) |dx| {
                const column: usize = @intCast(@mod(x0 + @as(i64, @intCast(dx)), header.width));
                const row: usize = @intCast(std.math.clamp(y0 + @as(i64, @intCast(dy)), 0, header.height - 1));
                try testing.expect(pixels[row * header.width + column] <= bound);
            }
        }
    }
}
