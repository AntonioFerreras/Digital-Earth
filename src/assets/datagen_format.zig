//! Binary MLP training data written by `earth datagen` and read by
//! `python/train.py` (`python/datagen_format.py` mirrors this layout).
//!
//! Little-endian, every field 4 bytes:
//!   magic "EDGN" | version u32 | record_count u32 | field_count u32 (= 7)
//!   record_count x f32[7]: u v w z R G B
//! A file may exceed 4 GiB; only the record count is 32-bit.
//!
//! RGB is in the renderer's radiance scale, the normalized spectral estimator's.
//! Text data from the original Taichi data_prep.py, and `.edgn` files converted
//! from it, are `Network.legacy_output_scale` times darker; the header does not
//! say which a file holds.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Io = std.Io;

comptime {
    // Records are written and viewed as raw host memory.
    assert(builtin.cpu.arch.endian() == .little);
    assert(@sizeOf(Header) == 16);
    assert(@sizeOf(Record) == field_count * @sizeOf(f32));
}

pub const magic: [4]u8 = "EDGN".*;
pub const version: u32 = 1;
pub const field_count: u32 = 7;
/// File extension `train.py` reads as this format.
pub const suffix = ".edgn";

pub const Header = extern struct {
    magic: [4]u8,
    version: u32,
    record_count: u32,
    field_count: u32,
};

pub const Record = extern struct {
    /// Bruneton-style coordinates of the MLP input domain, each in [0, 1].
    uvwz: [4]f32,
    /// Linear Rec.709 radiance, the training target.
    rgb: [3]f32,
};

pub const ParseError = error{
    Truncated,
    InvalidMagic,
    UnsupportedVersion,
    FieldCountMismatch,
    RecordCountMismatch,
};

/// Views the records of a whole file held in memory, without copying.
pub fn parse(bytes: []align(@alignOf(Record)) const u8) ParseError![]const Record {
    if (bytes.len < @sizeOf(Header)) return error.Truncated;
    const header: *const Header = @ptrCast(bytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &header.magic, &magic)) return error.InvalidMagic;
    if (header.version != version) return error.UnsupportedVersion;
    if (header.field_count != field_count) return error.FieldCountMismatch;
    const body = bytes[@sizeOf(Header)..];
    if (body.len != @as(u64, header.record_count) * @sizeOf(Record)) return error.RecordCountMismatch;

    const records: []const Record = @alignCast(std.mem.bytesAsSlice(Record, body));
    assert(records.len == header.record_count);
    return records;
}

/// Streams records into a file. The header's record count is written by
/// `finish`, so a producer need not know the total up front.
pub const Writer = struct {
    file_writer: Io.File.Writer,
    record_count: u32,

    pub const Error = Io.File.Writer.Error || Io.File.WritePositionalError || error{TooManyRecords};

    /// Writes a provisional header; `buffer` batches the record writes.
    pub fn init(io: Io, file: Io.File, buffer: []u8) Error!Writer {
        var writer: Writer = .{ .file_writer = file.writer(io, buffer), .record_count = 0 };
        const header = headerFor(0);
        writer.file_writer.interface.writeAll(std.mem.asBytes(&header)) catch
            return writer.file_writer.err.?;
        return writer;
    }

    pub fn write(writer: *Writer, records: []const Record) Error!void {
        if (records.len > std.math.maxInt(u32) - writer.record_count) return error.TooManyRecords;
        writer.file_writer.interface.writeAll(std.mem.sliceAsBytes(records)) catch
            return writer.file_writer.err.?;
        writer.record_count += @intCast(records.len);
    }

    /// Flushes the records and stores the final count in the header.
    pub fn finish(writer: *Writer) Error!void {
        try writer.file_writer.flush();
        assert(writer.file_writer.logicalPos() ==
            @sizeOf(Header) + @as(u64, writer.record_count) * @sizeOf(Record));
        const header = headerFor(writer.record_count);
        try writer.file_writer.file.writePositionalAll(writer.file_writer.io, std.mem.asBytes(&header), 0);
    }

    fn headerFor(record_count: u32) Header {
        return .{ .magic = magic, .version = version, .record_count = record_count, .field_count = field_count };
    }
};

// The uvwz texel counts of the MLP's mapping, SCATTERING_* in shaders/atmosphere/bruneton.slang.
const scattering_mu_s_size = 4096.0;
const scattering_mu_half_size = 2048.0;
const scattering_r_size = 4096.0;

/// The uvwz ranges that `scattering_uvwz` reaches at the ends of each domain, mirroring
/// the V_, W_, and Z_ constants of shaders/atmosphere/bruneton.slang: nu from -1 to 1,
/// mu_s from MU_S_MIN to 1, mu over rays that hit the ground (horizon to nadir) and
/// over the others (zenith to horizon), r from the ground to the top of the atmosphere.
pub const nu_range = [2]f32{ 0, 1 };
pub const mu_s_range = [2]f32{ 0.5 / scattering_mu_s_size, 1 - 0.5 / scattering_mu_s_size };
pub const mu_ground_range = [2]f32{ 0.25 / scattering_mu_half_size, 0.5 - 0.25 / scattering_mu_half_size };
pub const mu_sky_range = [2]f32{ 0.5 + 0.25 / scattering_mu_half_size, 1 - 0.25 / scattering_mu_half_size };
pub const r_range = [2]f32{ 0.5 / scattering_r_size, 1 - 0.5 / scattering_r_size };

/// Whether a grid of `shape` vertices along (u, v, w, z) has both ends of every range:
/// at least two vertices per axis and per mu half, with the halves equal.
pub fn gridShapeValid(shape: [4]u32) bool {
    var vertex_count: u64 = 1;
    for (shape) |size| {
        if (size < 2) return false;
        vertex_count *= size;
    }
    return shape[2] % 2 == 0 and shape[2] >= 4 and vertex_count <= std.math.maxInt(u32);
}

/// Vertex `index` of a grid of `shape` vertices in row-major `[z][w][v][u]` order: u
/// varies fastest. `earth datagen --sampling grid` writes its records so, and
/// shaders/atmosphere/lut4d.slang samples them. Vertex k of n on an axis sits at
/// fraction k / (n - 1) of the axis's range, so the first and last vertices are the
/// ends of the domain; mu holds the ground range then the sky range, n / 2 vertices each.
pub fn gridPoint(shape: [4]u32, index: u64) [4]f32 {
    assert(gridShapeValid(shape));
    var vertex: [4]u32 = undefined;
    var rest = index;
    for (&vertex, shape) |*coordinate, size| {
        coordinate.* = @intCast(rest % size);
        rest /= size;
    }
    assert(rest == 0);
    const mu_half = shape[2] / 2;
    const mu_range, const mu_vertex = if (vertex[2] < mu_half)
        .{ mu_ground_range, vertex[2] }
    else
        .{ mu_sky_range, vertex[2] - mu_half };
    return .{
        rangePoint(nu_range, vertex[0], shape[0]),
        rangePoint(mu_s_range, vertex[1], shape[1]),
        rangePoint(mu_range, mu_vertex, mu_half),
        rangePoint(r_range, vertex[3], shape[3]),
    };
}

/// Vertex `vertex` of `count` spanning `range`; the last one is the range's end exactly.
fn rangePoint(range: [2]f32, vertex: u32, count: u32) f32 {
    assert(count >= 2);
    assert(vertex < count);
    if (vertex == count - 1) return range[1];
    const fraction = @as(f32, @floatFromInt(vertex)) / @as(f32, @floatFromInt(count - 1));
    return range[0] + fraction * (range[1] - range[0]);
}

test "grid points run through u fastest, z slowest, and span every range" {
    const shape = [4]u32{ 2, 3, 4, 5 };
    const corner = [4]f32{ nu_range[0], mu_s_range[0], mu_ground_range[0], r_range[0] };
    try std.testing.expectEqual(corner, gridPoint(shape, 0));
    try std.testing.expectEqual([4]f32{ 1, corner[1], corner[2], corner[3] }, gridPoint(shape, 1));
    const mu_s_middle = gridPoint(shape, 2)[1];
    try std.testing.expectApproxEqAbs(0.5, mu_s_middle, 1e-7);
    try std.testing.expectEqual(mu_s_range[1], gridPoint(shape, 2 * 2)[1]);
    try std.testing.expectEqual(mu_ground_range[1], gridPoint(shape, 2 * 3)[2]);
    try std.testing.expectEqual(mu_sky_range[0], gridPoint(shape, 2 * 3 * 2)[2]);
    try std.testing.expectEqual(mu_sky_range[1], gridPoint(shape, 2 * 3 * 3)[2]);
    try std.testing.expectApproxEqAbs(0.25 * (r_range[1] - r_range[0]), gridPoint(shape, 2 * 3 * 4)[3] - r_range[0], 1e-7);
    try std.testing.expectEqual([4]f32{ 1, mu_s_range[1], mu_sky_range[1], r_range[1] }, gridPoint(shape, 2 * 3 * 4 * 5 - 1));

    try std.testing.expect(gridShapeValid(shape));
    try std.testing.expect(!gridShapeValid(.{ 1, 3, 4, 5 }));
    try std.testing.expect(!gridShapeValid(.{ 2, 3, 2, 5 }));
    try std.testing.expect(!gridShapeValid(.{ 2, 3, 5, 5 }));
    try std.testing.expect(!gridShapeValid(.{ 65536, 65536, 4, 2 }));
}

test "streamed records read back unchanged" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var expected: [5]Record = undefined;
    for (&expected, 0..) |*record, index| {
        const base: f32 = @floatFromInt(index * 10);
        record.* = .{ .uvwz = .{ base, base + 1, base + 2, base + 3 }, .rgb = .{ base + 4, base + 5, base + 6 } };
    }
    {
        const file = try tmp.dir.createFile(io, "samples.edgn", .{});
        defer file.close(io);
        // A buffer smaller than a record forces several drains.
        var buffer: [20]u8 = undefined;
        var writer: Writer = try .init(io, file, &buffer);
        try writer.write(expected[0..2]);
        try writer.write(expected[2..]);
        try writer.finish();
    }
    const bytes = try tmp.dir.readFileAllocOptions(io, "samples.edgn", std.testing.allocator, .unlimited, .of(Record), null);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualSlices(Record, &expected, try parse(bytes));
}

test "malformed files are rejected" {
    var words: [4 + 2 * field_count]u32 align(@alignOf(Record)) = @splat(0);
    const header: *Header = @ptrCast(&words);
    header.* = .{ .magic = magic, .version = version, .record_count = 2, .field_count = field_count };
    const bytes = std.mem.sliceAsBytes(&words);
    try std.testing.expectEqual(2, (try parse(bytes)).len);

    try std.testing.expectError(error.Truncated, parse(bytes[0..12]));
    try std.testing.expectError(error.RecordCountMismatch, parse(bytes[0 .. bytes.len - 4]));
    header.record_count = 3;
    try std.testing.expectError(error.RecordCountMismatch, parse(bytes));
    header.record_count = 2;
    header.field_count = 6;
    try std.testing.expectError(error.FieldCountMismatch, parse(bytes));
    header.field_count = field_count;
    header.version = version + 1;
    try std.testing.expectError(error.UnsupportedVersion, parse(bytes));
    header.version = version;
    header.magic[0] = 'X';
    try std.testing.expectError(error.InvalidMagic, parse(bytes));
}
