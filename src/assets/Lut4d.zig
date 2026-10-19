//! A dense table of the symmetric scene's linear Rec.709 radiance over uvwz, for the
//! `bruneton` render mode: the `.edgn` file `earth datagen --sampling grid` writes.
//! The format carries no shape; every record carries its grid vertex instead, so the
//! vertex count of an axis is where its coordinate first changes in record order, and
//! `parse` then requires each record to sit exactly on its `datagen_format.gridPoint`.
//! Any other `.edgn` file fails that check. Grid files only come from `earth datagen`,
//! so they hold the renderer's radiance scale.

const Lut4d = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const datagen_format = @import("datagen_format.zig");

const log = std.log.scoped(.assets);

/// Vertex counts along (u, v, w, z) = (nu, mu_s, mu, r), one texel each on the GPU.
shape: [4]u32,
/// Views `file`, in `datagen_format.gridPoint` order.
records: []const datagen_format.Record,
/// Owned: the whole file, which `records` views without a copy.
file: []align(@alignOf(datagen_format.Record)) const u8,

/// Where `earth` looks without `--lut`, relative to the data root.
pub const default_path = "cache/lut/scattering.edgn";
/// Guards against loading an unrelated huge file; a 64^4 grid is 470 MB.
const file_size_max = 1 << 32;

pub const ParseError = datagen_format.ParseError || error{ NotAGrid, TexelCentreGrid, NonFiniteRadiance };
pub const LoadError = Io.Dir.ReadFileAllocError || ParseError;

pub fn load(gpa: Allocator, io: Io, dir: Io.Dir, path: []const u8) LoadError!Lut4d {
    const file = try dir.readFileAllocOptions(io, path, gpa, .limited(file_size_max), .of(datagen_format.Record), null);
    errdefer gpa.free(file);
    return parse(file);
}

/// Validates `file` as a grid and takes ownership of it on success.
pub fn parse(file: []align(@alignOf(datagen_format.Record)) const u8) ParseError!Lut4d {
    const records = try datagen_format.parse(file);
    if (records.len == 0) return error.NotAGrid;
    if (isTexelCentreGrid(records[0].uvwz)) {
        log.err("this table samples texel centres, which the bruneton mode no longer reads; " ++
            "rebake it with `earth datagen --sampling grid`", .{});
        return error.TexelCentreGrid;
    }
    var shape: [4]u32 = undefined;
    var stride: u64 = 1;
    for (&shape, 0..) |*size, axis| {
        // Records step along this axis every `stride` records until the next axis moves.
        var count: u64 = 1;
        while (stride * count < records.len and
            std.mem.eql(f32, records[stride * count].uvwz[axis + 1 ..], records[0].uvwz[axis + 1 ..])) count += 1;
        if (count > std.math.maxInt(u32)) return error.NotAGrid;
        size.* = @intCast(count);
        stride *= count;
    }
    if (stride != records.len or !datagen_format.gridShapeValid(shape)) return error.NotAGrid;
    for (records, 0..) |record, index| {
        if (!std.mem.eql(f32, &record.uvwz, &datagen_format.gridPoint(shape, index))) return error.NotAGrid;
        for (record.rgb) |channel| {
            if (!std.math.isFinite(channel)) return error.NonFiniteRadiance;
        }
    }
    return .{ .shape = shape, .records = records, .file = file };
}

/// Whether `first` is the first record of a grid of texel centres, (0.5 / n) per axis,
/// which earlier `earth datagen` versions wrote.
fn isTexelCentreGrid(first: [4]f32) bool {
    for (first) |centre| {
        if (!(centre > 0 and centre <= 0.5)) return false;
        if (0.5 / @round(0.5 / centre) != centre) return false;
    }
    return true;
}

pub fn deinit(lut: *Lut4d, gpa: Allocator) void {
    gpa.free(lut.file);
    lut.* = undefined;
}

/// The radiance of texel `texel` (u, v, w, z).
pub fn at(lut: *const Lut4d, texel: [4]u32) [3]f32 {
    for (texel, lut.shape) |index, size| assert(index < size);
    const index = ((texel[3] * lut.shape[2] + texel[2]) * lut.shape[1] + texel[1]) * lut.shape[0] + texel[0];
    return lut.records[index].rgb;
}

/// A grid file of `shape` in `gpa` whose texel at index i has radiance (i, -i, 1 / (i + 1)).
pub fn testFile(gpa: Allocator, shape: [4]u32) Allocator.Error![]align(@alignOf(datagen_format.Record)) u8 {
    const record_count = shape[0] * shape[1] * shape[2] * shape[3];
    const file = try gpa.alignedAlloc(u8, .of(datagen_format.Record), @sizeOf(datagen_format.Header) +
        record_count * @sizeOf(datagen_format.Record));
    const header: *datagen_format.Header = @ptrCast(file[0..@sizeOf(datagen_format.Header)]);
    header.* = .{
        .magic = datagen_format.magic,
        .version = datagen_format.version,
        .record_count = record_count,
        .field_count = datagen_format.field_count,
    };
    const records: []datagen_format.Record = @alignCast(std.mem.bytesAsSlice(datagen_format.Record, file[@sizeOf(datagen_format.Header)..]));
    for (records, 0..) |*record, index| {
        const value: f32 = @floatFromInt(index);
        record.* = .{ .uvwz = datagen_format.gridPoint(shape, index), .rgb = .{ value, -value, 1 / (value + 1) } };
    }
    return file;
}

const testing = std.testing;

test "the shape comes from the record order, and at() indexes in grid order" {
    const shape = [4]u32{ 2, 3, 4, 7 };
    var lut: Lut4d = try .parse(try testFile(testing.allocator, shape));
    defer lut.deinit(testing.allocator);
    try testing.expectEqual(shape, lut.shape);
    try testing.expectEqual([3]f32{ 0, -0.0, 1 }, lut.at(.{ 0, 0, 0, 0 }));
    try testing.expectEqual(1, lut.at(.{ 1, 0, 0, 0 })[0]);
    try testing.expectEqual(2, lut.at(.{ 0, 1, 0, 0 })[0]);
    try testing.expectEqual(2 * 3, lut.at(.{ 0, 0, 1, 0 })[0]);
    try testing.expectEqual(2 * 3 * 4, lut.at(.{ 0, 0, 0, 1 })[0]);
    try testing.expectEqual(2 * 3 * 4 * 7 - 1, lut.at(.{ 1, 2, 3, 6 })[0]);
}

test "files that are not a complete grid are rejected" {
    const gpa = testing.allocator;
    const file = try testFile(gpa, .{ 2, 3, 4, 5 });
    defer gpa.free(file);
    const header: *datagen_format.Header = @ptrCast(file[0..@sizeOf(datagen_format.Header)]);
    const records: []datagen_format.Record = @alignCast(std.mem.bytesAsSlice(datagen_format.Record, file[@sizeOf(datagen_format.Header)..]));
    _ = try parse(file);

    // A truncated grid: one slice of z short.
    header.record_count -= 2 * 3 * 4;
    try testing.expectError(error.NotAGrid, parse(file[0 .. file.len - 2 * 3 * 4 * @sizeOf(datagen_format.Record)]));
    header.record_count += 2 * 3 * 4;
    // Two records swapped: the order is part of the format.
    std.mem.swap(datagen_format.Record, &records[3], &records[4]);
    try testing.expectError(error.NotAGrid, parse(file));
    std.mem.swap(datagen_format.Record, &records[3], &records[4]);
    // Uniformly sampled training data.
    records[0].uvwz = .{ 0.31, 0.77, 0.02, 0.5 };
    try testing.expectError(error.NotAGrid, parse(file));
    records[0].uvwz[0] = 0;
    try testing.expectError(error.NotAGrid, parse(file));
    records[0].uvwz = datagen_format.gridPoint(.{ 2, 3, 4, 5 }, 0);
    records[7].rgb[1] = std.math.nan(f32);
    try testing.expectError(error.NonFiniteRadiance, parse(file));
    records[7].rgb[1] = 0;
    // A bake from before the grid spanned the domain: texel centres (k + 0.5) / n.
    records[0].uvwz = .{ 0.25, 1.0 / 6.0, 0.125, 0.1 };
    try testing.expectError(error.TexelCentreGrid, parse(file));
    records[0].uvwz = datagen_format.gridPoint(.{ 2, 3, 4, 5 }, 0);
    header.magic[0] = 'X';
    try testing.expectError(error.InvalidMagic, parse(file));
}
