//! A dense cloud density grid for the `cloud_bench` render mode: the `.ecld` file that
//! python/clouds/convert_vdb.py writes from an OpenVDB grid.
//!
//! Layout, little-endian: a 64-byte header (`Header`), then the density as f16 per
//! voxel, x fastest, then the majorant of every cell of `cell` voxels per axis, then
//! the distance from inside the cloud to its edge per cell of `distance_downsample`
//! voxels per axis, both f16 in the same order. Positions and distances are in the
//! source's world units.

const CloudVolume = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// Voxels per axis, x first.
size: [3]u32,
/// Voxels per cell edge of the majorant grid.
cell: u32,
/// Cells per axis.
cells: [3]u32,
/// Distance texels per axis.
distance_size: [3]u32,
/// Voxel edge length.
voxel: f32,
/// Where voxel (0, 0, 0) starts.
min_corner: [3]f32,
/// View `file`.
density: []const f16,
majorant: []const f16,
distance: []const f16,
/// Owned: the whole file.
file: []align(@alignOf(f16)) const u8,

/// Where the volumes lie under the data root, named by the config's `cloud_volume`.
pub const directory_path = "cache/clouds";
pub const file_extension = ".ecld";
/// The Walt Disney Animation Studios cloud at half resolution.
pub const default_stem = "wdas_cloud_half";
pub const PathBuffer = [directory_path.len + 1 + Io.Dir.max_name_bytes]u8;
/// The half-resolution WDAS cloud is 1.6 GB and the widest altocumulus field 2.3 GB;
/// the full WDAS cloud, 8 GB, does not fit in a 3D texture.
const file_size_max = 1 << 32;
const magic = "ECLOUD02";
/// DISTANCE_DOWNSAMPLE in python/clouds/convert_vdb.py.
pub const distance_downsample = 2;

const Header = extern struct {
    magic: [8]u8,
    size: [3]u32,
    cell: u32,
    cells: [3]u32,
    voxel: f32,
    min_corner: [3]f32,
    distance_size: [3]u32,
};

comptime {
    assert(@sizeOf(Header) == 64);
}

/// The file of the volume `stem`, relative to the data root.
pub fn filePath(buffer: *PathBuffer, stem: []const u8) []const u8 {
    assert(stem.len + file_extension.len <= Io.Dir.max_name_bytes);
    return std.fmt.bufPrint(buffer, "{s}/{s}{s}", .{ directory_path, stem, file_extension }) catch unreachable;
}

pub const ParseError = error{InvalidCloudVolume};
pub const LoadError = Io.Dir.ReadFileAllocError || ParseError;

pub fn load(gpa: Allocator, io: Io, dir: Io.Dir, path: []const u8) LoadError!CloudVolume {
    const file = try dir.readFileAllocOptions(io, path, gpa, .limited(file_size_max), .of(f16), null);
    errdefer gpa.free(file);
    return parse(file);
}

/// Validates `file` and takes ownership of it on success.
pub fn parse(file: []align(@alignOf(f16)) const u8) ParseError!CloudVolume {
    if (file.len < @sizeOf(Header)) return error.InvalidCloudVolume;
    const header = std.mem.bytesToValue(Header, file[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &header.magic, magic)) return error.InvalidCloudVolume;
    if (!(header.voxel > 0) or header.cell == 0) return error.InvalidCloudVolume;
    var voxel_count: u64 = 1;
    var cell_count: u64 = 1;
    var distance_count: u64 = 1;
    for (header.size, header.cells, header.distance_size) |size, cells, distance_size| {
        if (size == 0 or cells != std.math.divCeil(u32, size, header.cell) catch unreachable) return error.InvalidCloudVolume;
        if (distance_size != std.math.divCeil(u32, size, distance_downsample) catch unreachable) return error.InvalidCloudVolume;
        voxel_count *= size;
        cell_count *= cells;
        distance_count *= distance_size;
    }
    if (file.len != @sizeOf(Header) + 2 * (voxel_count + cell_count + distance_count)) return error.InvalidCloudVolume;
    const values: []const f16 = @ptrCast(file[@sizeOf(Header)..]);
    return .{
        .size = header.size,
        .cell = header.cell,
        .cells = header.cells,
        .distance_size = header.distance_size,
        .voxel = header.voxel,
        .min_corner = header.min_corner,
        .density = values[0..@intCast(voxel_count)],
        .majorant = values[@intCast(voxel_count)..][0..@intCast(cell_count)],
        .distance = values[@intCast(voxel_count + cell_count)..],
        .file = file,
    };
}

pub fn deinit(volume: *CloudVolume, gpa: Allocator) void {
    gpa.free(volume.file);
    volume.* = undefined;
}

const testing = std.testing;

test "parse reads the header and splits density, majorants, and distances" {
    const size = [3]u32{ 3, 2, 1 };
    const cells = [3]u32{ 2, 1, 1 };
    const distance_size = [3]u32{ 2, 1, 1 };
    const header: Header = .{
        .magic = magic.*,
        .size = size,
        .cell = 2,
        .cells = cells,
        .voxel = 0.5,
        .min_corner = .{ -1, 0, 1 },
        .distance_size = distance_size,
    };
    var bytes: [@sizeOf(Header) + 2 * (6 + 2 + 2)]u8 align(2) = undefined;
    @memcpy(bytes[0..@sizeOf(Header)], std.mem.asBytes(&header));
    const values: *[10]f16 = @ptrCast(bytes[@sizeOf(Header)..]);
    values.* = .{ 0, 0.25, 0.5, 0.75, 1, 0.5, 1, 0.5, 0, 2 };
    const volume = try parse(&bytes);
    try testing.expectEqual(size, volume.size);
    try testing.expectEqual(6, volume.density.len);
    try testing.expectEqual(@as(f16, 0.75), volume.density[3]);
    try testing.expectEqualSlices(f16, &.{ 1, 0.5 }, volume.majorant);
    try testing.expectEqualSlices(f16, &.{ 0, 2 }, volume.distance);

    try testing.expectError(error.InvalidCloudVolume, parse(bytes[0 .. bytes.len - 2]));
    bytes[0] = 'X';
    try testing.expectError(error.InvalidCloudVolume, parse(&bytes));
}
