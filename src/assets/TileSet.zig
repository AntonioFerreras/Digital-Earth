//! One `.tiles` file: a layer of one cube face as a quadtree of tiles, written by
//! python/tiles/bake.py in the format of docs/plans/20261020-streamed-earth-tiles.md.
//! The file is memory-mapped; entries index payloads, which decode into the tile
//! layouts below.

const TileSet = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Io = std.Io;
const stb = @import("stb");
const zstd = @import("zstd");

header: Header,
/// `entryCount(header.last_level)` entries, level by level, each level in Morton order.
entries: []align(1) const Entry,
/// The whole file; payload offsets count from its start.
bytes: []const u8,
file: Io.File,
memory_map: Io.File.MemoryMap,

comptime {
    // Headers and entries are read as raw host memory.
    assert(builtin.cpu.arch.endian() == .little);
    assert(@sizeOf(Header) == 32);
    assert(@sizeOf(Entry) == 16);
}

pub const Layer = enum(u32) { height, surface };
pub const Codec = enum(u32) { height_planes_zstd, jpeg };

pub const Header = extern struct {
    magic: [4]u8,
    version: u32,
    layer: Layer,
    face: u32,
    /// Intervals across a tile; a height tile has one more sample per row.
    intervals: u32,
    border: u32,
    last_level: u32,
    codec: Codec,

    pub const magic_expected: [4]u8 = "ETIL".*;
    pub const version_expected = 1;
};

pub const Entry = extern struct {
    offset: u64,
    /// 0: the tile is absent and its parent stands in.
    size: u32,
    /// Height range of the tile in metres, border included; 0 for surface tiles.
    height_min: i16,
    height_max: i16,
};

pub const Key = struct {
    level: u32,
    x: u32,
    y: u32,

    /// Position of the key's entry in the file.
    pub fn index(key: Key) usize {
        assert(key.level < 16);
        assert(key.x >> @intCast(key.level) == 0 and key.y >> @intCast(key.level) == 0);
        return entryCount(key.level) + morton(key.x, key.y);
    }
};

/// The tile layouts of the plan: height tiles are vertex-centred, surface tiles
/// pixel-centred, both with `border` extra samples on each side.
pub const intervals = 192;
pub const border = 2;
pub const height_side = intervals + 1 + 2 * border;
pub const surface_side = intervals + 2 * border;

/// Entries of the levels below `level`: (4^level - 1) / 3.
pub fn entryCount(level: u32) usize {
    return ((@as(usize, 1) << @intCast(2 * level)) - 1) / 3;
}

/// Bit k of `x` at bit 2k, bit k of `y` at bit 2k + 1.
fn morton(x: u32, y: u32) usize {
    var result: usize = 0;
    for (0..16) |bit| {
        result |= @as(usize, (x >> @intCast(bit)) & 1) << @intCast(2 * bit);
        result |= @as(usize, (y >> @intCast(bit)) & 1) << @intCast(2 * bit + 1);
    }
    return result;
}

pub const OpenError = error{TileSetInvalid} || Io.File.OpenError || Io.File.StatError ||
    Io.File.MemoryMap.CreateError;

/// Maps the tile file at `path` under `dir`, which must hold `layer`.
pub fn open(io: Io, dir: Io.Dir, path: []const u8, layer: Layer) OpenError!TileSet {
    const file = try dir.openFile(io, path, .{});
    errdefer file.close(io);
    const size = (try file.stat(io)).size;
    if (size < @sizeOf(Header)) return error.TileSetInvalid;
    var memory_map = try file.createMemoryMap(io, .{
        .len = @intCast(size),
        .protection = .{ .read = true },
        .populate = false,
    });
    errdefer memory_map.destroy(io);
    const bytes = memory_map.memory;
    const header = std.mem.bytesToValue(Header, bytes[0..@sizeOf(Header)]);
    const valid_header = std.mem.eql(u8, &header.magic, &Header.magic_expected) and
        header.version == Header.version_expected and header.layer == layer and header.face < 6 and
        header.intervals == intervals and header.border == border and header.last_level < 16 and
        header.codec == switch (layer) {
            .height => Codec.height_planes_zstd,
            .surface => Codec.jpeg,
        };
    if (!valid_header) return error.TileSetInvalid;
    const entry_count = entryCount(header.last_level + 1);
    const table_end = @sizeOf(Header) + entry_count * @sizeOf(Entry);
    if (table_end > bytes.len) return error.TileSetInvalid;
    const entries = std.mem.bytesAsSlice(Entry, bytes[@sizeOf(Header)..table_end]);
    for (entries) |tile| {
        if (tile.size != 0 and (tile.offset < table_end or tile.offset + tile.size > bytes.len)) {
            return error.TileSetInvalid;
        }
    }
    return .{ .header = header, .entries = entries, .bytes = bytes, .file = file, .memory_map = memory_map };
}

pub fn close(set: *TileSet, io: Io) void {
    set.memory_map.destroy(io);
    set.file.close(io);
    set.* = undefined;
}

/// The entry of `key`, or null where the file holds no tile there.
pub fn entry(set: TileSet, key: Key) ?Entry {
    if (key.level > set.header.last_level) return null;
    const result = set.entries[key.index()];
    return if (result.size == 0) null else result;
}

pub fn payload(set: TileSet, tile: Entry) []const u8 {
    return set.bytes[@intCast(tile.offset)..][0..tile.size];
}

/// A decoded height tile, rows from the smallest v, columns along u.
pub const HeightTile = struct {
    height: [height_side * height_side]i16,
    /// 255 is all water.
    water: [height_side * height_side]u8,
};

pub const DecodeError = error{TileCorrupt};

/// Decodes a height payload; `planes` is scratch space for the decompressed planes.
pub fn decodeHeight(bytes: []const u8, planes: *[3 * height_side * height_side]u8, tile: *HeightTile) DecodeError!void {
    const written = zstd.ZSTD_decompress(planes, planes.len, bytes.ptr, bytes.len);
    if (zstd.ZSTD_isError(written) != 0 or written != planes.len) return error.TileCorrupt;
    const count = height_side * height_side;
    const low = planes[0..count];
    const high = planes[count .. 2 * count];
    @memcpy(&tile.water, planes[2 * count ..]);
    for (0..height_side) |j| {
        for (0..height_side) |i| {
            const at = j * height_side + i;
            const a: i32 = if (i > 0) tile.height[at - 1] else 0;
            const b: i32 = if (j > 0) tile.height[at - height_side] else 0;
            const c: i32 = if (i > 0 and j > 0) tile.height[at - height_side - 1] else 0;
            const prediction = if (c >= @max(a, b)) @min(a, b) else if (c <= @min(a, b)) @max(a, b) else a + b - c;
            const z: u16 = @as(u16, high[at]) << 8 | low[at];
            const d: i16 = @bitCast((z >> 1) ^ (0 -% (z & 1)));
            tile.height[at] = @truncate(prediction +% d);
        }
    }
}

/// Decodes a surface payload into `rgb`, rows from the smallest v.
pub fn decodeSurface(bytes: []const u8, rgb: *[surface_side * surface_side * 3]u8) DecodeError!void {
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;
    const pixels = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &width, &height, &channels, 3) orelse
        return error.TileCorrupt;
    defer stb.stbi_image_free(pixels);
    if (width != surface_side or height != surface_side) return error.TileCorrupt;
    @memcpy(rgb, pixels[0..rgb.len]);
}

const testing = std.testing;

test "entry indices put each level after the ones above it, in Morton order" {
    try testing.expectEqual(0, (Key{ .level = 0, .x = 0, .y = 0 }).index());
    try testing.expectEqual(1, (Key{ .level = 1, .x = 0, .y = 0 }).index());
    try testing.expectEqual(2, (Key{ .level = 1, .x = 1, .y = 0 }).index());
    try testing.expectEqual(3, (Key{ .level = 1, .x = 0, .y = 1 }).index());
    try testing.expectEqual(5 + 0b1110, (Key{ .level = 2, .x = 2, .y = 3 }).index());
    try testing.expectEqual(entryCount(10) - 1, (Key{ .level = 9, .x = 511, .y = 511 }).index());
}

test "height planes decode the plan's prediction, wrapping as it does" {
    var tile: HeightTile = undefined;
    var expected: [height_side * height_side]i16 = undefined;
    var prng: std.Random.DefaultPrng = .init(3);
    for (&expected, 0..) |*h, at| h.* = if (at % 97 == 0) prng.random().int(i16) else @intCast(@as(i32, @intCast(at % height_side)) * 7 - 300);
    // Encode as the plan specifies, in the opposite direction of the decoder.
    var planes: [3 * height_side * height_side]u8 = undefined;
    const count = height_side * height_side;
    for (0..height_side) |j| {
        for (0..height_side) |i| {
            const at = j * height_side + i;
            const a: i32 = if (i > 0) expected[at - 1] else 0;
            const b: i32 = if (j > 0) expected[at - height_side] else 0;
            const c: i32 = if (i > 0 and j > 0) expected[at - height_side - 1] else 0;
            const prediction = if (c >= @max(a, b)) @min(a, b) else if (c <= @min(a, b)) @max(a, b) else a + b - c;
            const d: i16 = @truncate(@as(i32, expected[at]) -% prediction);
            const z: u16 = @bitCast((d << 1) ^ (d >> 15));
            planes[at] = @truncate(z);
            planes[count + at] = @truncate(z >> 8);
            planes[2 * count + at] = @intCast(at % 256);
        }
    }
    // Only the decoder is built, so the planes go into a frame of one raw block (RFC 8878):
    // magic, a descriptor with a single segment and a 4-byte content size, the size,
    // then a last raw block's 3-byte header.
    comptime assert(planes.len < 128 << 10);
    var frame: [13 + planes.len]u8 = undefined;
    std.mem.writeInt(u32, frame[0..4], 0xFD2FB528, .little);
    frame[4] = 0b1010_0000;
    std.mem.writeInt(u32, frame[5..9], planes.len, .little);
    std.mem.writeInt(u24, frame[9..12], 1 | planes.len << 3, .little);
    @memcpy(frame[12..][0..planes.len], &planes);
    const size = 12 + planes.len;
    var scratch: [planes.len]u8 = undefined;
    try decodeHeight(frame[0..size], &scratch, &tile);
    try testing.expectEqualSlices(i16, &expected, &tile.height);
    try testing.expectEqual(@as(u8, 5), tile.water[5]);
    try testing.expectError(error.TileCorrupt, decodeHeight(frame[0 .. size / 2], &scratch, &tile));
}
