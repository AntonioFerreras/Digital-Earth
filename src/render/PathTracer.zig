//! The `path_tracer` render mode (shaders/modes/pathtrace.slang): spectral volumetric
//! path tracing of either scene preset, one path per pixel per frame. Everything it
//! reads is a shared global, so it adds no bindings of its own.

const std = @import("std");
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const PathTracer = @This();

kernel: ModeKernel,

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) gpu.Kernel.LayoutError!PathTracer {
    _ = gpa;
    return .{ .kernel = try .init(shared.device, "modes/pathtrace", "pathtrace_main", &.{}) };
}

pub fn deinit(mode: *PathTracer, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.kernel.deinit();
    mode.* = undefined;
}

pub fn record(
    mode: *PathTracer,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    try mode.kernel.dispatch(command_list, shared, frame, &.{});
}

/// Nothing is baked from the scene: the kernel reads `scene` on every dispatch.
pub fn invalidate(mode: *PathTracer, scene: *const uniforms.SceneUniforms) void {
    _ = mode;
    _ = scene;
}
