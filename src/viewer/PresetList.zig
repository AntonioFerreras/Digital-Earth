//! The configs in <data>/presets, by name: what the panel's picker loads, and the name
//! its save field edits.

const std = @import("std");
const Io = std.Io;
const DataRoot = @import("../assets/DataRoot.zig");
const Panel = @import("Panel.zig");

const PresetList = @This();

/// Owned file stems, sorted.
stems: [][]u8,
/// `stems`, each NUL-terminated, as `ImGui_Combo` takes them.
labels: [:0]u8,
selected: usize,
/// The save field's text, NUL-terminated.
name: [name_max:0]u8,

pub const name_max = 64;
pub const extension = ".zon";
/// Holds any file name in the presets directory, not only the save field's.
pub const PathBuffer = [DataRoot.Subdir.presets.path().len + 1 + Io.Dir.max_name_bytes]u8;

/// The file of the preset `name`, relative to the data root.
pub fn path(buffer: *PathBuffer, name: []const u8) []const u8 {
    std.debug.assert(name.len + extension.len <= Io.Dir.max_name_bytes);
    return std.fmt.bufPrint(buffer, "{s}/{s}{s}", .{ DataRoot.Subdir.presets.path(), name, extension }) catch unreachable;
}

pub fn init(gpa: std.mem.Allocator, io: Io, data_root: DataRoot) error{OutOfMemory}!PresetList {
    const stems = try data_root.listStems(gpa, io, DataRoot.Subdir.presets.path(), extension);
    errdefer DataRoot.freeStems(gpa, stems);
    return .{ .stems = stems, .labels = try Panel.joinLabels(gpa, stems), .selected = 0, .name = @splat(0) };
}

pub fn deinit(list: *PresetList, gpa: std.mem.Allocator) void {
    DataRoot.freeStems(gpa, list.stems);
    gpa.free(list.labels);
    list.* = undefined;
}

/// Lists the directory afresh, keeping the name, and selects `stem` if it is listed.
pub fn rescan(list: *PresetList, gpa: std.mem.Allocator, io: Io, data_root: DataRoot, stem: []const u8) error{OutOfMemory}!void {
    var fresh: PresetList = try .init(gpa, io, data_root);
    fresh.name = list.name;
    for (fresh.stems, 0..) |listed, index| {
        if (std.mem.eql(u8, listed, stem)) fresh.selected = index;
    }
    list.deinit(gpa);
    list.* = fresh;
}

/// Fills the save field, cut to `name_max` bytes.
pub fn setName(list: *PresetList, name: []const u8) void {
    const len = @min(name.len, name_max);
    list.name = @splat(0);
    @memcpy(list.name[0..len], name[0..len]);
}

/// The save field's name, or null when it is empty or would leave the presets directory.
pub fn savedName(list: *const PresetList) ?[]const u8 {
    const name = std.mem.trim(u8, std.mem.sliceTo(&list.name, 0), " ");
    if (name.len == 0 or name[0] == '.' or std.mem.findAny(u8, name, "/\\:") != null) return null;
    return name;
}

const testing = std.testing;

test "presets list by stem, and the save field rejects names that leave the directory" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "presets");
    for ([_][]const u8{ "presets/b.zon", "presets/a.zon", "presets/notes.txt" }) |file| {
        try tmp.dir.writeFile(testing.io, .{ .sub_path = file, .data = "" });
    }
    const root: DataRoot = .{ .dir = tmp.dir, .path = "data" };
    var list: PresetList = try .init(testing.allocator, testing.io, root);
    defer list.deinit(testing.allocator);
    try testing.expectEqual(2, list.stems.len);
    try testing.expectEqualStrings("a\x00b\x00", list.labels);

    list.setName("c");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "presets/c.zon", .data = "" });
    try list.rescan(testing.allocator, testing.io, root, "c");
    try testing.expectEqual(2, list.selected);
    try testing.expectEqualStrings("c", list.savedName().?);
    var buffer: PathBuffer = undefined;
    try testing.expectEqualStrings("presets/c.zon", path(&buffer, "c"));

    for ([_][]const u8{ "", "  ", "../x", "a/b", "a\\b", ".hidden", "c:x" }) |bad| {
        list.setName(bad);
        try testing.expectEqual(null, list.savedName());
    }
    list.setName(" sunset ");
    try testing.expectEqualStrings("sunset", list.savedName().?);
}
