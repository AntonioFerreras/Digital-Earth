//! The viewer's persistent state, saved and loaded as ZON. Missing fields keep their
//! defaults and unknown fields are ignored, so files survive the struct evolving.
//!
//! A config from `load` or `clone` owns its strings and is released with `deinit`; a
//! default-initialized one borrows static strings and must be cloned before `deinit`.

const std = @import("std");
const Io = std.Io;
const Camera = @import("Camera.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const Renderer = @import("../render/Renderer.zig");
const View = @import("../render/View.zig");
const Post = @import("../render/Post.zig");
const scene = @import("../render/scene.zig");

const Config = @This();

camera: Camera = .{},
sun: View.Sun = .{},
mode: Renderer.ModeKind = .debug,
preset: scene.Preset = .textured,
/// The original's default; the upload falls back to Q1 where Q2 does not fit.
texture_quality: EarthMaps.Quality = .q2,
exposure: f64 = 2.5,
/// OpenDRT settings: a mild chroma boost over OpenDRT's look, contrast 0, the
/// DaGrinchi tonescale.
saturation: f64 = 1.2,
contrast: f64 = 0,
tonescale: Post.TonescalePreset = .dagrinchi_tone,
gamma: f64 = 1,
/// The camera response's .rf file stem in LUT/camera_response_functions; a name, not
/// a position, because directory order differs between platforms.
camera_response: []const u8 = "Neutral",
/// Off removes the cloud shell from presets that have one.
clouds: bool = true,
gui_visible: bool = true,

const log = std.log.scoped(.config);

/// Borrows `config`'s strings.
pub fn tone(config: Config) Post.Tone {
    return .{
        .exposure = config.exposure,
        .saturation = config.saturation,
        .contrast = config.contrast,
        .tonescale = config.tonescale,
        .camera_response = config.camera_response,
        .gamma = config.gamma,
    };
}

pub fn clone(config: Config, gpa: std.mem.Allocator) error{OutOfMemory}!Config {
    var copy = config;
    copy.camera_response = try gpa.dupe(u8, config.camera_response);
    return copy;
}

pub fn deinit(config: *Config, gpa: std.mem.Allocator) void {
    gpa.free(config.camera_response);
    config.* = undefined;
}

pub fn setCameraResponse(config: *Config, gpa: std.mem.Allocator, name: []const u8) error{OutOfMemory}!void {
    const copy = try gpa.dupe(u8, name);
    gpa.free(config.camera_response);
    config.camera_response = copy;
}

/// Far above any real config; guards against loading an unrelated file.
const file_size_max = 64 * 1024;

pub const LoadError = Io.Dir.ReadFileAllocError || error{InvalidConfig};
pub const SaveError = Io.Dir.WriteFileError || error{OutOfMemory};

pub fn load(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, path: []const u8) LoadError!Config {
    const source = try dir.readFileAllocOptions(io, path, gpa, .limited(file_size_max), .of(u8), 0);
    defer gpa.free(source);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var diagnostics: std.zon.parse.Diagnostics = undefined;
    var config = std.zon.parse.fromSlice(Config, .{
        .gpa = gpa,
        .arena = arena.allocator(),
        .source = source,
        .diagnostics = &diagnostics,
        .ignore_unknown_fields = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseZon => {
            diagnostics.log(path);
            return error.InvalidConfig;
        },
    };
    try config.validate(path);
    return config.clone(gpa);
}

pub fn save(config: Config, gpa: std.mem.Allocator, io: Io, dir: Io.Dir, path: []const u8) SaveError!void {
    var text: Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    std.zon.stringify.serialize(config, .{}, &text.writer) catch return error.OutOfMemory;
    text.writer.writeByte('\n') catch return error.OutOfMemory;
    try dir.writeFile(io, .{ .sub_path = path, .data = text.written() });
}

/// Rejects values the camera cannot represent and renormalizes hand-edited directions.
/// Checks are written as `!(valid)` so NaN fails them.
fn validate(config: *Config, path: []const u8) error{InvalidConfig}!void {
    const camera = &config.camera;
    const radius_m = @sqrt(@reduce(.Add, camera.position * camera.position));
    if (!(radius_m >= Camera.planet_radius_m)) {
        log.err("{s}: camera position is inside the planet", .{path});
        return error.InvalidConfig;
    }
    if (!(camera.vertical_fov_degrees >= View.vertical_fov_degrees_min and
        camera.vertical_fov_degrees <= View.vertical_fov_degrees_max))
    {
        log.err("{s}: vertical_fov_degrees outside [{d}, {d}]", .{
            path, View.vertical_fov_degrees_min, View.vertical_fov_degrees_max,
        });
        return error.InvalidConfig;
    }
    if (!(camera.aspect_scale > 0)) {
        log.err("{s}: aspect_scale must be positive", .{path});
        return error.InvalidConfig;
    }
    for ([_]*Camera.Vec3{ &camera.forward, &camera.up }) |direction| {
        const direction_length = @sqrt(@reduce(.Add, direction.* * direction.*));
        if (!(direction_length > 0 and std.math.isFinite(direction_length))) {
            log.err("{s}: camera forward and up must be finite and nonzero", .{path});
            return error.InvalidConfig;
        }
        direction.* /= @splat(direction_length);
    }
}

const testing = std.testing;

test "save then load round-trips every field" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const saved: Config = .{
        .camera = .{
            .position = .{ 1.25e7, -3.5e6, 6371100.5 },
            .forward = .{ 0, 0, 1 },
            .up = .{ 0, 1, 0 },
            .vertical_fov_degrees = 41.5,
            .aspect_scale = 0.9,
        },
        .mode = .debug,
        .sun = .{ .angle_degrees = 123.25, .path_rotation_degrees = 17 },
        .preset = .symmetric,
        .texture_quality = .q1,
        .exposure = -0.5,
        .saturation = 0.8,
        .contrast = -0.25,
        .tonescale = .aces_2,
        .gamma = 2.2,
        .camera_response = "FP2900ZG",
        .gui_visible = false,
    };
    try saved.save(testing.allocator, testing.io, tmp.dir, "viewer.zon");
    var loaded = try load(testing.allocator, testing.io, tmp.dir, "viewer.zon");
    defer loaded.deinit(testing.allocator);
    try testing.expectEqualDeep(saved, loaded);
}

test "missing fields keep defaults, unknown fields are ignored, directions renormalize" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "partial.zon", .data =
        \\.{
        \\    .camera = .{ .forward = .{ 0, 0, -2 }, .roll = 1.0 },
        \\    .exposure = 4,
        \\    .from_a_future_version = .{ .x = 1 },
        \\}
    });
    var loaded = try load(testing.allocator, testing.io, tmp.dir, "partial.zon");
    defer loaded.deinit(testing.allocator);
    const defaults: Config = .{};
    try testing.expectEqual(@as(Camera.Vec3, .{ 0, 0, -1 }), loaded.camera.forward);
    try testing.expectEqual(defaults.camera.position, loaded.camera.position);
    try testing.expectEqual(4, loaded.exposure);
    try testing.expectEqual(defaults.gamma, loaded.gamma);
    try testing.expectEqualStrings(defaults.camera_response, loaded.camera_response);
}

test "invalid files are rejected" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const cases = [_][]const u8{
        ".{ .camera = .{ .position = .{ 0, 0, 1 } } }",
        ".{ .camera = .{ .up = .{ 0, 0, 0 } } }",
        ".{ .camera = .{ .vertical_fov_degrees = 0 } }",
        ".{ .gamma = \"bright\" }",
        "not zon",
    };
    for (cases) |source| {
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "bad.zon", .data = source });
        try testing.expectError(error.InvalidConfig, load(testing.allocator, testing.io, tmp.dir, "bad.zon"));
    }
    try testing.expectError(error.FileNotFound, load(testing.allocator, testing.io, tmp.dir, "absent.zon"));
}

test "shipped presets load, name their camera response, and survive a save" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var presets = try Io.Dir.cwd().openDir(testing.io, "presets", .{ .iterate = true });
    defer presets.close(testing.io);
    var iterator = presets.iterate();
    const camera_responses = std.StaticStringMap([]const u8).initComptime(.{
        .{ "apollo-11.zon", "FP2900ZG" },
        .{ "florida.zon", "Neutral" },
        .{ "sunset-hurricane.zon", "Neutral" },
    });
    var count: u32 = 0;
    while (try iterator.next(testing.io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".zon")) continue;
        var preset = try load(testing.allocator, testing.io, presets, entry.name);
        defer preset.deinit(testing.allocator);
        try testing.expectEqualStrings(camera_responses.get(entry.name).?, preset.camera_response);
        try preset.save(testing.allocator, testing.io, tmp.dir, entry.name);
        var reloaded = try load(testing.allocator, testing.io, tmp.dir, entry.name);
        defer reloaded.deinit(testing.allocator);
        try testing.expectEqualDeep(preset, reloaded);
        count += 1;
    }
    try testing.expectEqual(camera_responses.keys().len, count);
}
