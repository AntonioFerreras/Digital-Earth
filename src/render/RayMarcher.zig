//! The `raymarch` render mode (shaders/modes/raymarch.slang): a few marched segments
//! per pixel against two lookup tables that it bakes on the GPU
//! (shaders/modes/raymarch_lut.slang) whenever the scene preset changes. The tables and
//! their parameterizations are documented in shaders/atmosphere/scattering_luts.slang.

const std = @import("std");
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const RayMarcher = @This();

kernel: ModeKernel,
multiple_scattering_lut_param: gpu.Kernel.Param,
bake_multiple_scattering: gpu.Kernel,
multiple_scattering: gpu.Texture,
/// Set by `invalidate`; the next `record` rebakes the table first.
stale: bool,

/// MULTIPLE_SCATTERING_LUT_SIZE in scattering_luts.slang.
pub const multiple_scattering_size = [3]u32{ 32, 32, 128 };

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) !RayMarcher {
    _ = gpa;
    const device = shared.device;
    const kernel: ModeKernel = try .init(device, "modes/raymarch", "raymarch_main", &.{});
    errdefer kernel.deinit();
    const bake_multiple_scattering: gpu.Kernel = try .init(device, "modes/raymarch_lut", "bake_multiple_scattering", &.{});
    errdefer bake_multiple_scattering.deinit();

    const multiple_scattering_lut_param = try kernel.kernel.param("multiple_scattering_lut");

    // Bundles are interpolated in hardware, so this table needs linear filtering.
    const format: gpu.Texture.Format = if (device.info().rgba32f_filterable != 0) .rgba32_float else .rgba16_float;
    const multiple_scattering: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = format,
        .width = multiple_scattering_size[0],
        .height = multiple_scattering_size[1],
        .depth_or_layers = multiple_scattering_size[2],
        .usage = lut_usage,
        .label = "multiple scattering LUT",
    });
    return .{
        .kernel = kernel,
        .multiple_scattering_lut_param = multiple_scattering_lut_param,
        .bake_multiple_scattering = bake_multiple_scattering,
        .multiple_scattering = multiple_scattering,
        .stale = true,
    };
}

pub fn deinit(mode: *RayMarcher, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.multiple_scattering.deinit();
    mode.bake_multiple_scattering.deinit();
    mode.kernel.deinit();
    mode.* = undefined;
}

pub fn record(
    mode: *RayMarcher,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    if (mode.stale) {
        try mode.bake(command_list, shared, &frame.scene);
        mode.stale = false;
    }
    try mode.kernel.dispatch(command_list, shared, frame, &.{
        .{ .param = mode.multiple_scattering_lut_param, .resource = .{ .texture = mode.multiple_scattering } },
    });
}

pub fn invalidate(mode: *RayMarcher, scene: *const uniforms.SceneUniforms) void {
    _ = scene;
    mode.stale = true;
}

/// Records the multiple-scattering bake for `scene`.
pub fn bake(
    mode: *const RayMarcher,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    scene: *const uniforms.SceneUniforms,
) gpu.Error!void {
    const scene_data: gpu.CommandList.Binding.Resource = .{ .data = std.mem.asBytes(scene) };
    const ozone_cross_section: gpu.CommandList.Binding.Resource = .{ .buffer = shared.luts.ozone_cross_section };
    const ozone_profile: gpu.CommandList.Binding.Resource = .{ .buffer = shared.luts.ozone_profile };

    const multiple_scattering = mode.bake_multiple_scattering;
    try command_list.dispatch(multiple_scattering, &.{
        .{ .param = try multiple_scattering.param("scene"), .resource = scene_data },
        .{ .param = try multiple_scattering.param("ozone_cross_section"), .resource = ozone_cross_section },
        .{ .param = try multiple_scattering.param("ozone_profile"), .resource = ozone_profile },
        .{ .param = try multiple_scattering.param("cie"), .resource = .{ .buffer = shared.luts.cie } },
        .{ .param = try multiple_scattering.param("column_density_lut"), .resource = .{ .texture = shared.luts.column_density } },
        .{
            .param = try multiple_scattering.param("multiple_scattering_target"),
            .resource = .{ .texture = mode.multiple_scattering },
        },
    }, multiple_scattering.groupCount(multiple_scattering_size));
}

// Copies out serve the tests.
const lut_usage: gpu.Texture.Usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true };
