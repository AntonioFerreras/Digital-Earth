//! The networks the viewer offers the neural mode: the one it started with
//! (`--weights`, else the default network), then every network file in
//! <data>/cache/neural by name. The panel picks `selected`; the viewer hands it to
//! the renderer and records it as `active` once the renderer accepts it.

const std = @import("std");
const Io = std.Io;
const DataRoot = @import("../assets/DataRoot.zig");
const Network = @import("../assets/Network.zig");
const Renderer = @import("../render/Renderer.zig");

const NetworkList = @This();

/// Paths owned.
files: []Renderer.InputFile,
/// The labels, each NUL-terminated, as `ImGui_Combo` takes them: a directory entry's
/// file stem, the startup network's path as given. Empty without networks.
labels: [:0]u8,
active: usize,
selected: usize,

const log = std.log.scoped(.viewer);

/// Lists `startup` first, if given, and makes it active. A missing or unreadable
/// directory lists nothing from it.
pub fn init(gpa: std.mem.Allocator, io: Io, data_root: DataRoot, startup: ?Renderer.InputFile) error{OutOfMemory}!NetworkList {
    var files: std.ArrayList(Renderer.InputFile) = .empty;
    var labels: std.ArrayList(u8) = .empty;
    defer labels.deinit(gpa);
    errdefer {
        for (files.items) |file| gpa.free(file.path);
        files.deinit(gpa);
    }
    if (startup) |file| {
        try files.append(gpa, .{ .io = io, .dir = file.dir, .path = try gpa.dupe(u8, file.path) });
        try appendLabel(gpa, &labels, file.path);
    }

    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |name| gpa.free(name);
        names.deinit(gpa);
    }
    listDirectory(gpa, io, data_root, &names) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.FileNotFound => {},
        else => data_root.logFailure(Network.directory_path, err),
    };
    std.mem.sort([]const u8, names.items, {}, lessThan);
    for (names.items) |name| {
        const path = try std.fs.path.join(gpa, &.{ Network.directory_path, name });
        errdefer gpa.free(path);
        if (startup) |file| {
            if (file.dir.handle == data_root.dir.handle and std.mem.eql(u8, file.path, path)) {
                gpa.free(path);
                continue;
            }
        }
        try files.append(gpa, .{ .io = io, .dir = data_root.dir, .path = path });
        try appendLabel(gpa, &labels, std.mem.cutSuffix(u8, name, Network.file_extension).?);
    }
    const owned_labels = try labels.toOwnedSliceSentinel(gpa, 0);
    errdefer gpa.free(owned_labels);
    return .{ .files = try files.toOwnedSlice(gpa), .labels = owned_labels, .active = 0, .selected = 0 };
}

pub fn deinit(list: *NetworkList, gpa: std.mem.Allocator) void {
    for (list.files) |file| gpa.free(file.path);
    gpa.free(list.files);
    gpa.free(list.labels);
    list.* = undefined;
}

fn listDirectory(gpa: std.mem.Allocator, io: Io, data_root: DataRoot, names: *std.ArrayList([]const u8)) !void {
    var dir = try data_root.dir.openDir(io, Network.directory_path, .{ .iterate = true });
    defer dir.close(io);
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| {
        if (entry.kind == .directory or !std.mem.endsWith(u8, entry.name, Network.file_extension)) continue;
        try names.append(gpa, try gpa.dupe(u8, entry.name));
    }
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn appendLabel(gpa: std.mem.Allocator, labels: *std.ArrayList(u8), text: []const u8) error{OutOfMemory}!void {
    try labels.appendSlice(gpa, text);
    try labels.append(gpa, 0);
}

/// The label of entry `index`.
pub fn label(list: *const NetworkList, index: usize) []const u8 {
    var labels = std.mem.splitScalar(u8, list.labels, 0);
    for (0..index) |_| _ = labels.next();
    return labels.next().?;
}

const testing = std.testing;

test "the startup network comes first, then the directory's networks by name" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, Network.directory_path);
    for ([_][]const u8{ "b.emlp", "a.emlp", "notes.txt", "default.emlp" }) |name| {
        const path = try std.fs.path.join(testing.allocator, &.{ Network.directory_path, name });
        defer testing.allocator.free(path);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = path, .data = "" });
    }
    const root: DataRoot = .{ .dir = tmp.dir, .path = "data" };
    const startup: Renderer.InputFile = .{ .io = testing.io, .dir = tmp.dir, .path = Network.directory_path ++ "/default.emlp" };
    var list: NetworkList = try .init(testing.allocator, testing.io, root, startup);
    defer list.deinit(testing.allocator);
    try testing.expectEqual(3, list.files.len);
    try testing.expectEqualStrings(Network.directory_path ++ "/default.emlp", list.label(0));
    try testing.expectEqualStrings("a", list.label(1));
    try testing.expectEqualStrings("b", list.label(2));
    try testing.expectEqualStrings(Network.directory_path ++ "/b.emlp", list.files[2].path);

    var empty: NetworkList = try .init(testing.allocator, testing.io, .{ .dir = tmp.dir, .path = "data" }, null);
    defer empty.deinit(testing.allocator);
    try testing.expectEqual(3, empty.files.len);
    try testing.expectEqualStrings("default", empty.label(2));
}
