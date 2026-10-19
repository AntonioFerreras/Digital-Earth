//! NASA Earth maps: the texture set of each quality preset, the bake from the
//! source PNGs into raw caches, and memory-mapped cache readers.
//!
//! A cache file is `earth/<quality>/<role>.raw` under the cache directory:
//!   magic "EMAP" | version u32 | width u32 | height u32 | channels u32
//!   u8[height][width][channels], south row first, columns west to east from u = 0
//! South-first rows put v = 0 at the south pole, as the Taichi renderer loaded them.

const EarthMaps = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Image = @import("Image.zig");
const stb = @import("stb");

maps: std.EnumArray(Role, Map),

comptime {
    // Headers are read and written as raw host memory.
    assert(builtin.cpu.arch.endian() == .little);
    assert(@sizeOf(Header) == 20);
}

pub const Quality = enum { q1, q2 };

pub const Role = enum {
    albedo,
    /// 0 is sea level, 255 is 7800 m.
    topography,
    /// 255 is water.
    ocean_mask,
    /// Cloud coverage.
    clouds,
    /// 255 is land or shallow water, 0 is deep ocean.
    bathymetry,
    night_lights,
    stars,
};

pub const Source = struct {
    /// PNG file name inside the textures directory.
    file_name: []const u8,
    width: u32,
    height: u32,
};

pub const sources: std.EnumArray(Quality, std.EnumArray(Role, Source)) = .init(.{
    .q1 = .init(.{
        .albedo = .{ .file_name = "earth_color_10K.png", .width = 10800, .height = 5400 },
        .topography = .{ .file_name = "topography_10K.png", .width = 10800, .height = 5400 },
        .ocean_mask = .{ .file_name = "earth_landocean_8K.png", .width = 8100, .height = 4050 },
        .clouds = .{ .file_name = "earth_clouds_8K.png", .width = 8100, .height = 4050 },
        .bathymetry = .{ .file_name = "earth_bathymetry_10k.png", .width = 10800, .height = 5400 },
        .night_lights = .{ .file_name = "earth_nightlights_10K.png", .width = 10800, .height = 5400 },
        .stars = .{ .file_name = "stars_16K.png", .width = 16200, .height = 8100 },
    }),
    .q2 = .init(.{
        .albedo = .{ .file_name = "earth_color_21K.png", .width = 21600, .height = 10800 },
        .topography = .{ .file_name = "topography_21K.png", .width = 21600, .height = 10800 },
        .ocean_mask = .{ .file_name = "earth_landocean_16K.png", .width = 16200, .height = 8100 },
        .clouds = .{ .file_name = "earth_clouds_21K.png", .width = 21600, .height = 10800 },
        .bathymetry = .{ .file_name = "earth_bathymetry_21k.png", .width = 21600, .height = 10800 },
        .night_lights = .{ .file_name = "earth_nightlights_21K.png", .width = 21600, .height = 10800 },
        .stars = .{ .file_name = "stars_16K.png", .width = 16200, .height = 8100 },
    }),
});

/// Colour maps are cached as RGBA (alpha 255) for direct rgba8 upload; the
/// rest keep only channel 0 of whatever the PNG stores, as Taichi sampled them.
pub fn channelCount(role: Role) u32 {
    return switch (role) {
        .albedo, .stars => 4,
        .topography, .ocean_mask, .clouds, .bathymetry, .night_lights => 1,
    };
}

pub const Header = extern struct {
    magic: [4]u8,
    version: u32,
    width: u32,
    height: u32,
    channels: u32,

    pub const magic_expected: [4]u8 = "EMAP".*;
    pub const version_expected: u32 = 1;

    pub fn init(width: u32, height: u32, channels: u32) Header {
        return .{ .magic = magic_expected, .version = version_expected, .width = width, .height = height, .channels = channels };
    }

    pub fn of(quality: Quality, role: Role) Header {
        const source = sources.get(quality).get(role);
        return .init(source.width, source.height, channelCount(role));
    }

    pub fn pixelsSize(header: Header) u64 {
        return @as(u64, header.width) * header.height * header.channels;
    }
};

pub const cache_path_max = 64;

pub fn cachePath(buffer: *[cache_path_max]u8, quality: Quality, role: Role) []const u8 {
    return std.fmt.bufPrint(buffer, "earth/{t}/{t}.raw", .{ quality, role }) catch unreachable;
}

/// One memory-mapped cache file.
pub const Map = struct {
    header: Header,
    /// `header.pixelsSize()` bytes in the cache layout.
    pixels: []const u8,
    file: Io.File,
    memory_map: Io.File.MemoryMap,

    pub const OpenError = error{ CacheHeaderMismatch, CacheSizeMismatch } ||
        Io.File.OpenError || Io.File.StatError || Io.File.MemoryMap.CreateError;

    /// Maps the cache at `cache_path`, which must hold exactly `expected`.
    pub fn open(io: Io, cache_dir: Io.Dir, cache_path: []const u8, expected: Header) OpenError!Map {
        const file = try cache_dir.openFile(io, cache_path, .{});
        errdefer file.close(io);
        const size = (try file.stat(io)).size;
        if (size != @sizeOf(Header) + expected.pixelsSize()) return error.CacheSizeMismatch;

        var memory_map = try file.createMemoryMap(io, .{
            .len = @intCast(size),
            .protection = .{ .read = true },
            .populate = false,
        });
        errdefer memory_map.destroy(io);
        const header = std.mem.bytesToValue(Header, memory_map.memory[0..@sizeOf(Header)]);
        if (!std.meta.eql(header, expected)) return error.CacheHeaderMismatch;
        return .{
            .header = header,
            .pixels = memory_map.memory[@sizeOf(Header)..],
            .file = file,
            .memory_map = memory_map,
        };
    }

    pub fn close(map: *Map, io: Io) void {
        map.memory_map.destroy(io);
        map.file.close(io);
        map.* = undefined;
    }
};

/// Maps every cache of `quality`; `bake` must have run first.
pub fn open(io: Io, cache_dir: Io.Dir, quality: Quality) Map.OpenError!EarthMaps {
    var maps: std.EnumArray(Role, Map) = .initUndefined();
    var opened_count: usize = 0;
    errdefer for (std.enums.values(Role)[0..opened_count]) |role| maps.getPtr(role).close(io);
    for (std.enums.values(Role)) |role| {
        var path_buffer: [cache_path_max]u8 = undefined;
        maps.set(role, try .open(io, cache_dir, cachePath(&path_buffer, quality, role), .of(quality, role)));
        opened_count += 1;
    }
    return .{ .maps = maps };
}

pub fn close(earth_maps: *EarthMaps, io: Io) void {
    for (&earth_maps.maps.values) |*map| map.close(io);
    earth_maps.* = undefined;
}

pub const BakeMapError = error{ SourceDimensionsMismatch, ImageTooLarge, ImageDecode } ||
    Allocator.Error || Io.Dir.StatFileError || Io.Dir.ReadFileAllocError ||
    Io.Dir.CreateFileAtomicError || Io.File.Writer.Error || Io.File.Atomic.ReplaceError;

pub const BakeOutcome = enum { baked, fresh };

/// Bakes every map of `quality` from the PNGs in `source_dir`, one concurrent
/// task per map, so peak memory exceeds the decoded size of the whole set.
/// `gpa` must be thread-safe.
pub fn bake(gpa: Allocator, io: Io, source_dir: Io.Dir, cache_dir: Io.Dir, quality: Quality) (BakeMapError || Io.Cancelable)!void {
    var results: std.EnumArray(Role, BakeMapError!BakeOutcome) = .initUndefined();
    var group: Io.Group = .init;
    for (std.enums.values(Role)) |role| {
        group.async(io, bakeTask, .{ gpa, io, source_dir, cache_dir, quality, role, results.getPtr(role) });
    }
    try group.await(io);

    for (std.enums.values(Role)) |role| {
        _ = try results.get(role);
    }
}

fn bakeTask(
    gpa: Allocator,
    io: Io,
    source_dir: Io.Dir,
    cache_dir: Io.Dir,
    quality: Quality,
    role: Role,
    result: *BakeMapError!BakeOutcome,
) void {
    const log = std.log.scoped(.assets);
    const source = sources.get(quality).get(role);
    var path_buffer: [cache_path_max]u8 = undefined;
    const cache_path = cachePath(&path_buffer, quality, role);
    const start = Io.Clock.awake.now(io);
    result.* = bakeMap(gpa, io, source_dir, source.file_name, cache_dir, cache_path, .of(quality, role));
    const elapsed_ms = start.untilNow(io, .awake).toMilliseconds();
    if (result.*) |outcome| {
        log.info("{s} -> {s}: {t} in {d} ms", .{ source.file_name, cache_path, outcome, elapsed_ms });
    } else |err| switch (err) {
        error.ImageDecode => log.err("{s}: {t}: {s}", .{ source.file_name, err, stb.stbi_failure_reason() }),
        else => log.err("{s} -> {s}: {t}", .{ source.file_name, cache_path, err }),
    }
}

/// Decodes `source_file` into the cache at `cache_path` unless that cache is
/// at least as new as the source and already holds `expected`.
pub fn bakeMap(
    gpa: Allocator,
    io: Io,
    source_dir: Io.Dir,
    source_file: []const u8,
    cache_dir: Io.Dir,
    cache_path: []const u8,
    expected: Header,
) BakeMapError!BakeOutcome {
    assert(expected.channels == 1 or expected.channels == 4);
    const source_stat = try source_dir.statFile(io, source_file, .{});
    if (isCacheFresh(io, cache_dir, cache_path, expected, source_stat.mtime)) return .fresh;

    const encoded = try source_dir.readFileAlloc(io, source_file, gpa, .limited(1 << 30));
    var image = image: {
        defer gpa.free(encoded);
        break :image try Image.decode(encoded, if (expected.channels == 4) 4 else null);
    };
    defer image.deinit();
    if (image.width != expected.width or image.height != expected.height) return error.SourceDimensionsMismatch;

    var atomic_file = try cache_dir.createFileAtomic(io, cache_path, .{ .make_path = true, .replace = true });
    defer atomic_file.deinit(io);
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    var file_writer = atomic_file.file.writer(io, buffer);
    const writer = &file_writer.interface;

    const channel_row = try gpa.alloc(u8, image.width);
    defer gpa.free(channel_row);

    writer.writeAll(std.mem.asBytes(&expected)) catch return file_writer.err.?;
    const image_stride = @as(usize, image.width) * image.channels;
    for (0..image.height) |row_index| {
        // PNG rows run north to south.
        const row = image.pixels[(image.height - 1 - row_index) * image_stride ..][0..image_stride];
        if (image.channels == expected.channels) {
            writer.writeAll(row) catch return file_writer.err.?;
        } else {
            assert(expected.channels == 1);
            for (channel_row, 0..) |*value, column| value.* = row[column * image.channels];
            writer.writeAll(channel_row) catch return file_writer.err.?;
        }
    }
    try file_writer.flush();
    assert(file_writer.logicalPos() == @sizeOf(Header) + expected.pixelsSize());
    try atomic_file.replace(io);
    return .baked;
}

/// A cache that cannot be read counts as stale, so the bake rewrites it. Equal
/// mtimes count as fresh, as in make: file systems stamp with coarse ticks.
fn isCacheFresh(io: Io, cache_dir: Io.Dir, cache_path: []const u8, expected: Header, source_mtime: Io.Timestamp) bool {
    const file = cache_dir.openFile(io, cache_path, .{}) catch return false;
    defer file.close(io);
    const stat = file.stat(io) catch return false;
    if (stat.mtime.compare(.lt, source_mtime)) return false;
    if (stat.size != @sizeOf(Header) + expected.pixelsSize()) return false;
    var header: Header = undefined;
    const read_size = file.readPositionalAll(io, std.mem.asBytes(&header), 0) catch return false;
    return read_size == @sizeOf(Header) and std.meta.eql(header, expected);
}

test "bakeMap stores rows south-first with channel 0 or RGBA, then reuses the cache" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // A 3x2 RGB image: north row red ramp, south row green ramp.
    const rgb = [_]u8{ 10, 0, 0, 20, 0, 0, 30, 0, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0 };
    try Image.writePng(io, tmp.dir, "map.png", 3, 2, 3, &rgb);

    try std.testing.expectEqual(.baked, try bakeMap(gpa, io, tmp.dir, "map.png", tmp.dir, "earth/test/gray.raw", .init(3, 2, 1)));
    try std.testing.expectEqual(.baked, try bakeMap(gpa, io, tmp.dir, "map.png", tmp.dir, "earth/test/rgba.raw", .init(3, 2, 4)));
    try std.testing.expectEqual(.fresh, try bakeMap(gpa, io, tmp.dir, "map.png", tmp.dir, "earth/test/gray.raw", .init(3, 2, 1)));
    try std.testing.expectError(
        error.SourceDimensionsMismatch,
        bakeMap(gpa, io, tmp.dir, "map.png", tmp.dir, "earth/test/wrong.raw", .init(2, 3, 1)),
    );

    var gray: Map = try .open(io, tmp.dir, "earth/test/gray.raw", .init(3, 2, 1));
    defer gray.close(io);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 10, 20, 30 }, gray.pixels);

    var rgba: Map = try .open(io, tmp.dir, "earth/test/rgba.raw", .init(3, 2, 4));
    defer rgba.close(io);
    try std.testing.expectEqualSlices(u8, &.{ 0, 40, 0, 255 }, rgba.pixels[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 30, 0, 0, 255 }, rgba.pixels[20..24]);

    try std.testing.expectError(error.CacheSizeMismatch, Map.open(io, tmp.dir, "earth/test/gray.raw", .init(3, 2, 4)));
    try std.testing.expectError(error.CacheHeaderMismatch, Map.open(io, tmp.dir, "earth/test/gray.raw", .init(2, 3, 1)));
}

test "sources are 2:1 PNGs and cache paths name quality and role" {
    var path_buffer: [cache_path_max]u8 = undefined;
    try std.testing.expectEqualStrings("earth/q2/night_lights.raw", cachePath(&path_buffer, .q2, .night_lights));
    for (std.enums.values(Quality)) |quality| {
        for (std.enums.values(Role)) |role| {
            const source = sources.get(quality).get(role);
            try std.testing.expectEqual(source.width, 2 * source.height);
            try std.testing.expect(std.mem.endsWith(u8, source.file_name, ".png"));
        }
    }
}
