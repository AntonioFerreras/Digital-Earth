//! The `debug` render mode (shaders/modes/debug.slang): a sun-lit Lambertian planet
//! without atmosphere. It proves the camera, accumulation, and post end to end and is
//! the smallest example of the mode contract in `Renderer.zig`.

const std = @import("std");
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const DebugMode = @This();

kernel: ModeKernel,

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) gpu.Kernel.LayoutError!DebugMode {
    _ = gpa;
    return .{ .kernel = try .init(shared.device, "modes/debug", "debug_main", &.{}) };
}

pub fn deinit(mode: *DebugMode, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.kernel.deinit();
    mode.* = undefined;
}

pub fn record(
    mode: *DebugMode,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    try mode.kernel.dispatch(command_list, shared, frame, &.{});
}

/// Nothing is baked from the scene: the kernel reads `scene` on every dispatch.
pub fn invalidate(mode: *DebugMode, scene: *const uniforms.SceneUniforms) void {
    _ = mode;
    _ = scene;
}
