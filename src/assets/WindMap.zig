//! The 10 m wind of one hour that python/tiles/wind.py bakes into `<data>/tiles/wind.raw`,
//! memory-mapped: RGBA8 cells, south row first, columns west to east from 180 degrees
//! west; R the speed over `speed_max`, G and B the eastward and northward components
//! as (component / speed_max + 1) / 2.

const WindMap = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Io = std.Io;

width: u32,
height: u32,
cells: []const [4]u8,
file: Io.File,
memory_map: Io.File.MemoryMap,

pub const file_name = "wind.raw";

const Header = extern struct {
    magic: [4]u8,
    version: u32,
    width: u32,
    height: u32,

    const magic_expected = "EWND".*;
    const version_expected = 1;
};

comptime {
    // The header is read as raw host memory.
    assert(builtin.cpu.arch.endian() == .little);
}

pub const OpenError = Io.File.OpenError || Io.File.StatError || Io.File.MemoryMap.CreateError || error{WindMapInvalid};

pub fn open(io: Io, dir: Io.Dir) OpenError!WindMap {
    const file = try dir.openFile(io, file_name, .{});
    errdefer file.close(io);
    const size = (try file.stat(io)).size;
    if (size < @sizeOf(Header)) return error.WindMapInvalid;
    var memory_map = try file.createMemoryMap(io, .{
        .len = @intCast(size),
        .protection = .{ .read = true },
        .populate = false,
    });
    errdefer memory_map.destroy(io);
    const bytes = memory_map.memory;
    const header = std.mem.bytesToValue(Header, bytes[0..@sizeOf(Header)]);
    const valid = std.mem.eql(u8, &header.magic, &Header.magic_expected) and header.version == Header.version_expected and
        header.width > 0 and header.height > 0 and bytes.len == @sizeOf(Header) + @as(u64, header.width) * header.height * 4;
    if (!valid) return error.WindMapInvalid;
    return .{
        .width = header.width,
        .height = header.height,
        .cells = std.mem.bytesAsSlice([4]u8, bytes[@sizeOf(Header)..]),
        .file = file,
        .memory_map = memory_map,
    };
}

pub fn close(map: *WindMap, io: Io) void {
    map.memory_map.destroy(io);
    map.file.close(io);
    map.* = undefined;
}
