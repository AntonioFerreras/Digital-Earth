//! The cloud bench's baked probes on disk (src/render/CloudBench.zig): the walks' raw
//! sums, under <data>/cache/probes, one file per `Key`, so that a bake resumes where
//! the last one with the same settings stopped, or in a file of the caller's, as
//! validation keeps its fields (TESTING.md). Fits and lighting follow from the sums
//! and are not stored.

const std = @import("std");
const Io = std.Io;
const gpu = @import("../gpu.zig");

/// Everything the walks' sums depend on.
pub const Key = extern struct {
    /// Bumped whenever the bake or the sums' layout changes.
    format: u32 = format_version,
    /// The layout (src/render/ProbeLayout.zig): the dense grid's probes per axis, or
    /// zero for the adaptive layout, whose budget and depth ratio are zero otherwise.
    dense_resolution: u32,
    probe_budget: u32,
    depth_ratio: f32,
    /// The volume's grid, as `uniforms.CloudBenchUniforms` carries it.
    grid_size: [4]u32,
    grid_min: [4]f32,
    extinction: f32,
    bounce_limit: u32,
    options: Options,

    pub const Options = packed struct(u32) {
        ratio_tracking: bool,
        photons: bool,
        leaf_lengths: bool = false,
        _: u29 = 0,
    };
};

/// Fewer walks are not worth a file: a settings slider dragged through its range would
/// leave one per value it passed.
pub const walks_min = 64;

const format_version = 3;
const magic = "EPROBE01";
const directory = "cache/probes";

/// Padding-free, so that its bytes hash and compare as its values.
const Header = extern struct {
    magic: [magic.len]u8,
    key: Key,
    walks: u32,
    _: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Key) == 15 * 4);
    std.debug.assert(@sizeOf(Header) == magic.len + @sizeOf(Key) + 8);
}

pub const PathBuffer = [directory.len + "/".len + 16 + ".eprb".len]u8;

/// The cache's file for `key`.
pub fn cachePath(buffer: *PathBuffer, key: Key) []const u8 {
    const hash = std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
    return std.fmt.bufPrint(buffer, directory ++ "/{x:0>16}.eprb", .{hash}) catch unreachable;
}

/// Uploads the sums stored for `key` at `sub_path` into `transfer` and `sun_bins` and
/// returns their walks, or null if there is no file.
pub fn load(
    gpa: std.mem.Allocator,
    io: Io,
    dir: Io.Dir,
    sub_path: []const u8,
    device: gpu.Device,
    key: Key,
    transfer: gpu.Buffer,
    sun_bins: gpu.Buffer,
) !?u32 {
    const size = @sizeOf(Header) + transfer.size + sun_bins.size;
    const bytes = dir.readFileAlloc(io, sub_path, gpa, .limited(size + 1)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer gpa.free(bytes);
    if (bytes.len != size) return error.InvalidProbeCache;
    const header = std.mem.bytesToValue(Header, bytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &header.magic, magic) or !std.meta.eql(header.key, key)) return error.InvalidProbeCache;
    const sums = bytes[@sizeOf(Header)..];
    try transfer.upload(device, 0, sums[0..transfer.size]);
    try sun_bins.upload(device, 0, sums[transfer.size..]);
    return header.walks;
}

/// Reads `transfer` and `sun_bins` back, blocking until the GPU is done with them, and
/// stores them for `key` at `sub_path`, replacing what was there.
pub fn save(
    gpa: std.mem.Allocator,
    io: Io,
    dir: Io.Dir,
    sub_path: []const u8,
    device: gpu.Device,
    key: Key,
    walks: u32,
    transfer: gpu.Buffer,
    sun_bins: gpu.Buffer,
) !void {
    const bytes = try gpa.alloc(u8, @sizeOf(Header) + transfer.size + sun_bins.size);
    defer gpa.free(bytes);
    const header: Header = .{ .magic = magic.*, .key = key, .walks = walks };
    @memcpy(bytes[0..@sizeOf(Header)], std.mem.asBytes(&header));
    const sums = bytes[@sizeOf(Header)..];
    try transfer.read(device, 0, sums[0..transfer.size]);
    try sun_bins.read(device, 0, sums[transfer.size..]);
    if (std.fs.path.dirname(sub_path)) |parent| try dir.createDirPath(io, parent);
    // Written aside and renamed, so that an interrupted save leaves the old file whole.
    const part = try std.mem.concat(gpa, u8, &.{ sub_path, ".part" });
    defer gpa.free(part);
    try dir.writeFile(io, .{ .sub_path = part, .data = bytes });
    try dir.rename(part, dir, sub_path, io);
}

/// Deletes the file at `sub_path`, if there is one.
pub fn delete(io: Io, dir: Io.Dir, sub_path: []const u8) !void {
    dir.deleteFile(io, sub_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

const testing = std.testing;

test "each key has a file of its own" {
    const walks: Key = .{
        .dense_resolution = 64,
        .probe_budget = 0,
        .depth_ratio = 0,
        .grid_size = .{ 994, 676, 1225, 8 },
        .grid_min = .{ -200, -140, -250, 0.4 },
        .extinction = 0.1,
        .bounce_limit = 4096,
        .options = .{ .ratio_tracking = false, .photons = false },
    };
    var photons = walks;
    photons.options.photons = true;
    var a: PathBuffer = undefined;
    var b: PathBuffer = undefined;
    try testing.expectEqualStrings(cachePath(&a, walks), cachePath(&b, walks));
    try testing.expect(!std.mem.eql(u8, cachePath(&a, walks), cachePath(&b, photons)));
}
