//! The files the viewer offers a mode: every file of one kind under the data root by
//! name, and the one it started with when that lies elsewhere. The panel picks
//! `selected`; the viewer hands it to the renderer and records it as `active` once
//! the renderer accepts it.

const std = @import("std");
const Io = std.Io;
const DataRoot = @import("../assets/DataRoot.zig");
const Renderer = @import("../render/Renderer.zig");

const InputList = @This();

/// Paths owned.
files: []Renderer.InputFile,
/// The labels, each NUL-terminated, as `ImGui_Combo` takes them: a directory entry's
/// file stem, a startup file's path as given. Empty without files.
labels: [:0]u8,
active: usize,
selected: usize,

/// Lists <data>/`directory_path`/*`file_extension` by name, and makes `startup`
/// active: listed among them when it is one of them, else first. A missing or
/// unreadable directory lists nothing from it.
pub fn init(
    gpa: std.mem.Allocator,
    io: Io,
    data_root: DataRoot,
    directory_path: []const u8,
    file_extension: []const u8,
    startup: ?Renderer.InputFile,
) error{OutOfMemory}!InputList {
    var files: std.ArrayList(Renderer.InputFile) = .empty;
    var labels: std.ArrayList(u8) = .empty;
    defer labels.deinit(gpa);
    errdefer {
        for (files.items) |file| gpa.free(file.path);
        files.deinit(gpa);
    }
    var active: ?usize = null;
    const stems = try data_root.listStems(gpa, io, directory_path, file_extension);
    defer DataRoot.freeStems(gpa, stems);
    for (stems) |stem| {
        const path = try std.mem.concat(gpa, u8, &.{ directory_path, std.fs.path.sep_str, stem, file_extension });
        errdefer gpa.free(path);
        if (startup) |file| {
            if (file.dir.handle == data_root.dir.handle and std.mem.eql(u8, file.path, path)) active = files.items.len;
        }
        try files.append(gpa, .{ .io = io, .dir = data_root.dir, .path = path });
        try labels.appendSlice(gpa, stem);
        try labels.append(gpa, 0);
    }
    if (startup) |file| if (active == null) {
        const path = try gpa.dupe(u8, file.path);
        errdefer gpa.free(path);
        try files.insert(gpa, 0, .{ .io = io, .dir = file.dir, .path = path });
        try labels.insert(gpa, 0, 0);
        try labels.insertSlice(gpa, 0, file.path);
        active = 0;
    };
    const owned_labels = try labels.toOwnedSliceSentinel(gpa, 0);
    errdefer gpa.free(owned_labels);
    return .{ .files = try files.toOwnedSlice(gpa), .labels = owned_labels, .active = active orelse 0, .selected = active orelse 0 };
}

pub fn deinit(list: *InputList, gpa: std.mem.Allocator) void {
    for (list.files) |file| gpa.free(file.path);
    gpa.free(list.files);
    gpa.free(list.labels);
    list.* = undefined;
}

/// The label of entry `index`.
pub fn label(list: *const InputList, index: usize) []const u8 {
    var labels = std.mem.splitScalar(u8, list.labels, 0);
    for (0..index) |_| _ = labels.next();
    return labels.next().?;
}

/// Selects the entry labelled `wanted`; returns whether there is one.
pub fn select(list: *InputList, wanted: []const u8) bool {
    for (0..list.files.len) |index| {
        if (std.mem.eql(u8, list.label(index), wanted)) {
            list.selected = index;
            return true;
        }
    }
    return false;
}

const testing = std.testing;

test "the directory's files list by name, and the startup file joins them or comes first" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "cache/neural");
    for ([_][]const u8{ "b.emlp", "a.emlp", "notes.txt", "default.emlp" }) |name| {
        const path = try std.fs.path.join(testing.allocator, &.{ "cache/neural", name });
        defer testing.allocator.free(path);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = path, .data = "" });
    }
    const root: DataRoot = .{ .dir = tmp.dir, .path = "data" };
    const listed: Renderer.InputFile = .{ .io = testing.io, .dir = tmp.dir, .path = "cache/neural/default.emlp" };
    var list: InputList = try .init(testing.allocator, testing.io, root, "cache/neural", ".emlp", listed);
    defer list.deinit(testing.allocator);
    try testing.expectEqual(3, list.files.len);
    try testing.expectEqual(2, list.active);
    try testing.expectEqualStrings("a", list.label(0));
    try testing.expectEqualStrings("default", list.label(2));
    try testing.expectEqualStrings("cache/neural/b.emlp", list.files[1].path);
    try testing.expect(list.select("b"));
    try testing.expectEqual(1, list.selected);
    try testing.expect(!list.select("c"));
    try testing.expectEqual(1, list.selected);

    const elsewhere: Renderer.InputFile = .{ .io = testing.io, .dir = tmp.dir, .path = "trained/x.emlp" };
    var first: InputList = try .init(testing.allocator, testing.io, root, "cache/neural", ".emlp", elsewhere);
    defer first.deinit(testing.allocator);
    try testing.expectEqual(4, first.files.len);
    try testing.expectEqual(0, first.active);
    try testing.expectEqualStrings("trained/x.emlp", first.label(0));
    try testing.expectEqualStrings("a", first.label(1));

    var empty: InputList = try .init(testing.allocator, testing.io, root, "cache/clouds", ".ecld", null);
    defer empty.deinit(testing.allocator);
    try testing.expectEqual(0, empty.files.len);
}
