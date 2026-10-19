//! GPU round trips through the wrappers. Every test skips when no device can be
//! created, e.g. while libearth_native is a stub or on a machine without a GPU.

const std = @import("std");
const gpu = @import("../gpu.zig");
const FrameUniforms = @import("../render/uniforms.zig").FrameUniforms;

pub fn createTestDevice() !gpu.Device {
    return gpu.Device.init(.{
        .backend = .default,
        .validation = true,
        // Tests run with the build root as the working directory.
        .shader_search_paths = &.{"shaders"},
        .shader_cache_path = null,
    }) catch |err| switch (err) {
        error.Unsupported => error.SkipZigTest,
        else => err,
    };
}

test "buffer upload and read round trip" {
    const device = try createTestDevice();
    defer device.deinit();

    const values = [_]u32{ 1, 2, 3, 0xdeadbeef };
    const buffer: gpu.Buffer = try .init(device, .{
        .size = @sizeOf(@TypeOf(values)),
        .element_size = @sizeOf(u32),
        .usage = .{ .shader_resource = true, .copy_source = true, .copy_destination = true },
        .memory = .device,
        .label = "round trip",
    }, null);
    defer buffer.deinit();

    try buffer.upload(device, 0, std.mem.asBytes(&values));
    var result: [values.len]u32 = undefined;
    try buffer.read(device, 0, std.mem.asBytes(&result));
    try std.testing.expectEqualSlices(u32, &values, &result);
}

test "texture region upload and layer read round trip" {
    const device = try createTestDevice();
    defer device.deinit();

    const texture: gpu.Texture = try .init(device, .{
        .kind = .@"2d_array",
        .format = .rgba8_unorm,
        .width = 4,
        .height = 2,
        .depth_or_layers = 2,
        .usage = .{ .shader_resource = true, .copy_source = true, .copy_destination = true },
        .label = "round trip",
    });
    defer texture.deinit();

    var texels: [4 * 2 * 4]u8 = undefined;
    for (&texels, 0..) |*texel, index| texel.* = @intCast(index);
    try texture.upload(device, .{ .x = 0, .y = 0, .z = 1, .width = 4, .height = 2, .depth = 1 }, &texels, 4 * 4);
    var result: [texels.len]u8 = undefined;
    try texture.read(device, 1, &result);
    try std.testing.expectEqualSlices(u8, &texels, &result);
}

test "kernel layout check, data binding, and dispatch" {
    const device = try createTestDevice();
    defer device.deinit();

    const kernel: gpu.Kernel = try .init(device, "tests/host_gpu", "echo_frame", &.{});
    defer kernel.deinit();
    try kernel.checkLayout(FrameUniforms, "frame");

    const output: gpu.Buffer = try .init(device, .{
        .size = 4 * @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "echo output",
    }, null);
    defer output.deinit();

    var frame = std.mem.zeroes(FrameUniforms);
    frame.frame = .{ 10, 20, 30, 40 };
    const bindings = [_]gpu.CommandList.Binding{
        .{ .param = try kernel.param("frame"), .resource = .{ .data = std.mem.asBytes(&frame) } },
        .{ .param = try kernel.param("output"), .resource = .{ .buffer = output } },
    };
    const commands: gpu.CommandList = try .begin(device);
    const dispatched = commands.dispatch(kernel, &bindings, kernel.groupCount(.{ 4, 1, 1 }));
    try commands.submit();
    try dispatched;

    var result: [4]u32 = undefined;
    try output.read(device, 0, std.mem.asBytes(&result));
    try std.testing.expectEqualSlices(u32, &.{ 10, 21, 32, 43 }, &result);
}
