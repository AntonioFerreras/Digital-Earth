//! The directory every runtime file lives under (`--data`, the working directory by
//! default), and the one place that names the paths inside it.

const DataRoot = @This();

const std = @import("std");
const Io = std.Io;

dir: Io.Dir,
/// As given; only for messages.
path: []const u8,

pub const Subdir = enum {
    /// Spectral and camera-response tables (`Luts`).
    lut,
    /// The NASA source PNGs that `bake-textures` reads.
    textures,
    /// Baked Earth maps.
    cache,
    screenshots,
    /// Shipped and user configs, loaded with `--config`.
    presets,

    pub fn path(subdir: Subdir) []const u8 {
        return switch (subdir) {
            .lut => "LUT",
            .textures => "textures",
            .cache => "cache",
            .screenshots => "screenshots",
            .presets => "presets",
        };
    }
};

/// What the viewer's I and O keys save and load.
pub const config_path = "config.zon";

const log = std.log.scoped(.assets);

pub fn open(io: Io, path: []const u8) Io.Dir.OpenError!DataRoot {
    const dir = Io.Dir.cwd().openDir(io, path, .{}) catch |err| {
        log.err("data root '{s}': {t}", .{ path, err });
        return err;
    };
    return .{ .dir = dir, .path = path };
}

pub fn close(root: DataRoot, io: Io) void {
    root.dir.close(io);
}

/// Opens `subdir`, logging its full path when it cannot.
pub fn openSubdir(root: DataRoot, io: Io, subdir: Subdir, options: Io.Dir.OpenOptions) Io.Dir.OpenError!Io.Dir {
    return root.dir.openDir(io, subdir.path(), options) catch |err| {
        root.logFailure(subdir.path(), err);
        return err;
    };
}

/// Opens `subdir`, creating it first if it is missing.
pub fn createSubdir(root: DataRoot, io: Io, subdir: Subdir) Io.Dir.CreateDirPathOpenError!Io.Dir {
    return root.dir.createDirPathOpen(io, subdir.path(), .{}) catch |err| {
        root.logFailure(subdir.path(), err);
        return err;
    };
}

/// Logs that `sub_path` under the root failed with `err`, naming the path as the user
/// would type it; an absolute `sub_path` already is.
pub fn logFailure(root: DataRoot, sub_path: []const u8, err: anyerror) void {
    if (std.fs.path.isAbsolute(sub_path)) {
        log.err("{s}: {t}", .{ sub_path, err });
    } else {
        log.err("{s}{c}{s}: {t}", .{ root.path, std.fs.path.sep, sub_path, err });
    }
}

const testing = std.testing;

test "subdirectories resolve under the root, and creation is limited to createSubdir" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "data/LUT/camera_response_functions");

    const root: DataRoot = .{ .dir = try tmp.dir.openDir(testing.io, "data", .{}), .path = "data" };
    defer root.close(testing.io);
    var lut = try root.openSubdir(testing.io, .lut, .{});
    defer lut.close(testing.io);
    var crf = try lut.openDir(testing.io, "camera_response_functions", .{});
    crf.close(testing.io);

    try testing.expectError(error.FileNotFound, root.openSubdir(testing.io, .cache, .{}));
    var screenshots = try root.createSubdir(testing.io, .screenshots);
    screenshots.close(testing.io);
    var reopened = try root.openSubdir(testing.io, .screenshots, .{});
    reopened.close(testing.io);
}

test "a missing root is an error" {
    try testing.expectError(error.FileNotFound, open(testing.io, "no such data root"));
}
