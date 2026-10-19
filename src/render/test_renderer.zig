//! End-to-end renders without Earth maps: accumulation resets and image orientation
//! through the display pass with the debug mode, the path tracer, the neural mode's
//! networks, and the bruneton mode's table, and that only a mode sampling the maps
//! loads them. Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const Luts = @import("../assets/Luts.zig");
const Network = @import("../assets/Network.zig");
const Lut4d = @import("../assets/Lut4d.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;
const Post = @import("Post.zig");
const Renderer = @import("Renderer.zig");

const width = 64;
const height = 48;
const tone: Post.Tone = .{
    .exposure = 2.5,
    .saturation = 1.5,
    .contrast = 0,
    .tonescale = .dagrinchi_tone,
    .camera_response = "Neutral",
    .gamma = 1,
};

/// The whole planet in view from 40 000 km on -X, the sun at +X.
const view: Renderer.View = .{
    .position = .{ -40e6, 0, 0 },
    .forward = .{ 1, 0, 0 },
    .right = .{ 0, 0, 1 },
    .up = .{ 0, 1, 0 },
    .vertical_fov_degrees = 27,
    .aspect_scale = 1,
    .sun = .{ .angle_degrees = 90, .path_rotation_degrees = 0 },
};

/// `view` with the sun at +Y (the original's angle 0 with its path tilted by -90
/// degrees), lighting the planet's northern half.
const overhead: Renderer.View = init: {
    var lit = view;
    lit.sun = .{ .angle_degrees = 0, .path_rotation_degrees = -90 };
    break :init lit;
};

fn initRenderer(device: @import("../gpu.zig").Device) !Renderer {
    return initRendererWith(device, .{ .size = .{ width, height }, .mode = .debug, .preset = .symmetric, .texture_quality = .q1 });
}

fn initRendererWith(device: @import("../gpu.zig").Device, options: Renderer.Options) !Renderer {
    var lut_dir = try std.Io.Dir.cwd().openDir(testing.io, "LUT", .{});
    defer lut_dir.close(testing.io);
    var luts: Luts = try .load(testing.allocator, testing.io, lut_dir);
    defer luts.deinit(testing.allocator);
    return .init(testing.allocator, device, &luts, options);
}

test "view changes reset accumulation, tone changes do not" {
    const device = try createTestDevice();
    defer device.deinit();
    var renderer = try initRenderer(device);
    defer renderer.deinit();

    for (0..3) |_| try renderer.render(view, tone);
    try testing.expectEqual(3, renderer.sampleCount());
    try renderer.render(view, .{
        .exposure = 5,
        .saturation = 0.5,
        .contrast = 0.5,
        .tonescale = .aces_1,
        .camera_response = "FP2900ZG",
        .gamma = 2,
    });
    try testing.expectEqual(4, renderer.sampleCount());

    var moved = view;
    moved.vertical_fov_degrees = 30;
    try renderer.render(moved, tone);
    try testing.expectEqual(1, renderer.sampleCount());
    moved.sun.angle_degrees = 91;
    try renderer.render(moved, tone);
    try testing.expectEqual(1, renderer.sampleCount());
    // Without Earth maps the textured preset is refused and the accumulation kept.
    try testing.expectError(error.EarthMapsMissing, renderer.setPreset(.textured));
    try renderer.render(moved, tone);
    try testing.expectEqual(2, renderer.sampleCount());
    try renderer.resize(.{ 32, 32 });
    try renderer.render(moved, tone);
    try testing.expectEqual(1, renderer.sampleCount());
    // Without a map cache a texture quality change has nothing to reload.
    try renderer.setTextureQuality(.q2);
    try renderer.render(moved, tone);
    try testing.expectEqual(2, renderer.sampleCount());
    renderer.restart();
    try renderer.render(moved, tone);
    try testing.expectEqual(1, renderer.sampleCount());
}

test "camera responses are chosen by name, and an unknown name is an error" {
    const device = try createTestDevice();
    defer device.deinit();
    var renderer = try initRenderer(device);
    defer renderer.deinit();
    var film = tone;
    film.camera_response = "FP2900ZG";
    try renderer.render(view, film);
    film.camera_response = "Velvia";
    try testing.expectError(error.UnknownCameraResponse, renderer.render(view, film));
}

test "a sun overhead lights the top of the display image" {
    // The northern half of the planet must land in the first rows of the image.
    const device = try createTestDevice();
    defer device.deinit();
    var renderer = try initRenderer(device);
    defer renderer.deinit();

    for (0..8) |_| try renderer.render(overhead, tone);
    var pixels: [width * height][4]u8 = undefined;
    try renderer.image.read(device, 0, std.mem.sliceAsBytes(&pixels));

    const column = width / 2;
    const top = pixels[(height / 2 - 6) * width + column];
    const bottom = pixels[(height / 2 + 6) * width + column];
    const corner = pixels[0];
    try testing.expect(top[1] > 64);
    try testing.expectEqual(0, bottom[1]);
    try testing.expectEqual(0, corner[1]);
    try testing.expectEqual(255, corner[3]);
}

test "the path tracer lights the day side of the symmetric planet without Earth maps" {
    const device = try createTestDevice();
    defer device.deinit();
    var renderer = try initRenderer(device);
    defer renderer.deinit();
    try renderer.setMode(.path_tracer);

    for (0..64) |_| try renderer.render(overhead, tone);
    var pixels: [width * height][4]u8 = undefined;
    try renderer.image.read(device, 0, std.mem.sliceAsBytes(&pixels));

    const column = width / 2;
    const day = pixels[(height / 2 - 6) * width + column];
    const night = pixels[(height / 2 + 6) * width + column];
    try testing.expect(day[1] > 32);
    try testing.expect(night[1] < day[1] / 4);
}

/// `path` under the source tree, where the tests run.
fn sourceFile(path: []const u8) Renderer.InputFile {
    return .{ .io = testing.io, .dir = .cwd(), .path = path };
}

test "the neural mode needs a network, lights the atmosphere only, and keeps it when a new one fails" {
    const device = try createTestDevice();
    defer device.deinit();
    var without = try initRenderer(device);
    defer without.deinit();
    try testing.expectError(error.NetworkMissing, without.setMode(.neural));
    try testing.expectEqual(.debug, std.meta.activeTag(without.mode));

    var renderer = try initRendererWith(device, .{
        .size = .{ width, height },
        .mode = .neural,
        .preset = .symmetric,
        .texture_quality = .q1,
        .network = sourceFile("tests/goldens/mlp.emlp"),
    });
    defer renderer.deinit();
    for (0..2) |_| try renderer.render(view, tone);
    var pixels: [width * height][4]u8 = undefined;
    try renderer.image.read(device, 0, std.mem.sliceAsBytes(&pixels));
    const centre = pixels[(height / 2) * width + width / 2];
    try testing.expect(centre[0] > 16 and centre[1] > 16 and centre[2] > 16);
    try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, pixels[0]);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "broken.emlp", .data = "EMLP\x03\x00\x00\x00" });
    try testing.expectError(error.Truncated, renderer.setNetwork(.{ .io = testing.io, .dir = tmp.dir, .path = "broken.emlp" }));
    try testing.expectError(error.FileNotFound, renderer.setNetwork(sourceFile("tests/goldens/absent.emlp")));
    try testing.expectEqualStrings("tests/goldens/mlp.emlp", renderer.shared.network.?.path);
    try renderer.render(view, tone);

    // A second architecture compiles once; returning to the first reuses its kernel.
    try renderer.setNetwork(sourceFile("tests/goldens/neural/k_planes.emlp"));
    try renderer.render(view, tone);
    try renderer.setNetwork(sourceFile("tests/goldens/mlp.emlp"));
    try testing.expectEqual(2, renderer.mode.neural.variants.items.len);
    try testing.expectEqual(0, renderer.sampleCount());
}

test "the bruneton mode needs a table and lights the atmosphere only" {
    const device = try createTestDevice();
    defer device.deinit();
    var without = try initRenderer(device);
    defer without.deinit();
    try testing.expectError(error.Lut4dMissing, without.setMode(.bruneton));
    try testing.expectEqual(.debug, std.meta.activeTag(without.mode));

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const table = try Lut4d.testFile(testing.allocator, .{ 2, 3, 4, 5 });
    defer testing.allocator.free(table);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "table.edgn", .data = table });
    var renderer = try initRendererWith(device, .{
        .size = .{ width, height },
        .mode = .bruneton,
        .preset = .symmetric,
        .texture_quality = .q1,
        .lut4d = .{ .io = testing.io, .dir = tmp.dir, .path = "table.edgn" },
    });
    defer renderer.deinit();
    for (0..2) |_| try renderer.render(view, tone);
    var texels: [width * height][4]f32 = undefined;
    try renderer.accum.read(device, 0, std.mem.sliceAsBytes(&texels));

    // The test table's red channel is its texel index: positive except at texel 0.
    try testing.expect(texels[(height / 2) * width + width / 2][0] > 0);
    try testing.expectEqual([4]f32{ 0, 0, 0, 1 }, texels[0]);
}

/// Mean linear Rec.709 samples of `frame_count` neural-mode frames of `view`.
fn renderNeural(device: @import("../gpu.zig").Device, network: Renderer.InputFile, frame_count: u32) ![width * height][4]f32 {
    var renderer = try initRendererWith(device, .{
        .size = .{ width, height },
        .mode = .neural,
        .preset = .symmetric,
        .texture_quality = .q1,
        .network = network,
    });
    defer renderer.deinit();
    for (0..frame_count) |_| try renderer.render(view, tone);
    var texels: [width * height][4]f32 = undefined;
    try renderer.accum.read(device, 0, std.mem.sliceAsBytes(&texels));
    return texels;
}

test "the neural mode scales legacy networks to the renderer's radiance and new ones by their header" {
    const device = try createTestDevice();
    defer device.deinit();
    // tests/goldens/mlp.emlp is a version 1 export; the same network as version 2,
    // which carries its output scale, here 1.
    const legacy = try std.Io.Dir.cwd().readFileAlloc(testing.io, "tests/goldens/mlp.emlp", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(legacy);
    const one: f32 = 1;
    const unscaled = try std.mem.concat(testing.allocator, u8, &.{ legacy[0..4], std.mem.asBytes(&@as(u32, 2)), legacy[8..12], std.mem.asBytes(&one), legacy[12..] });
    defer testing.allocator.free(unscaled);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "unscaled.emlp", .data = unscaled });

    const scaled_image = try renderNeural(device, sourceFile("tests/goldens/mlp.emlp"), 2);
    const unscaled_image = try renderNeural(device, .{ .io = testing.io, .dir = tmp.dir, .path = "unscaled.emlp" }, 2);
    const centre = (height / 2) * width + width / 2;
    try testing.expect(unscaled_image[centre][1] > 0);
    for (scaled_image, unscaled_image) |scaled, plain| {
        for (scaled[0..3], plain[0..3]) |value, plain_value| {
            try testing.expectApproxEqRel(Network.legacy_output_scale * plain_value, value, 1e-6);
        }
    }
}

test "the Earth maps load only when the active mode samples them" {
    // An empty cache fails every load, so a successful call proves nothing was loaded.
    const device = try createTestDevice();
    defer device.deinit();
    var empty_cache = testing.tmpDir(.{});
    defer empty_cache.cleanup();
    var renderer = try initRendererWith(device, .{
        .size = .{ width, height },
        .mode = .debug,
        .preset = .textured,
        .texture_quality = .q1,
        .earth_map_cache = .{ .io = testing.io, .dir = empty_cache.dir },
    });
    defer renderer.deinit();
    try renderer.setTextureQuality(.q2);
    try renderer.render(view, tone);
    try testing.expectEqual(null, renderer.shared.earth_maps.quality);

    try testing.expectError(error.FileNotFound, renderer.setMode(.path_tracer));
    try testing.expectEqual(.debug, std.meta.activeTag(renderer.mode));
    try renderer.setPreset(.symmetric);
    try renderer.setMode(.path_tracer);
    try testing.expectError(error.FileNotFound, renderer.setPreset(.textured));
    try testing.expectEqual(.symmetric, renderer.preset);
}
