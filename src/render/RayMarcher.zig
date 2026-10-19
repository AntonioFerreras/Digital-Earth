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
column_density_lut_param: gpu.Kernel.Param,
multiple_scattering_lut_param: gpu.Kernel.Param,
bake_column_density: gpu.Kernel,
bake_multiple_scattering: gpu.Kernel,
column_density: gpu.Texture,
multiple_scattering: gpu.Texture,
/// Set by `invalidate`; the next `record` rebakes both tables first.
stale: bool,

/// COLUMN_DENSITY_LUT_SIZE and MULTIPLE_SCATTERING_LUT_SIZE in scattering_luts.slang.
pub const column_density_size = [2]u32{ 256, 256 };
pub const multiple_scattering_size = [3]u32{ 32, 32, 128 };

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) !RayMarcher {
    _ = gpa;
    const device = shared.device;
    const kernel: ModeKernel = try .init(device, "modes/raymarch", "raymarch_main", &.{});
    errdefer kernel.deinit();
    const bake_column_density: gpu.Kernel = try .init(device, "modes/raymarch_lut", "bake_column_density", &.{});
    errdefer bake_column_density.deinit();
    const bake_multiple_scattering: gpu.Kernel = try .init(device, "modes/raymarch_lut", "bake_multiple_scattering", &.{});
    errdefer bake_multiple_scattering.deinit();

    const column_density_lut_param = try kernel.kernel.param("column_density_lut");
    const multiple_scattering_lut_param = try kernel.kernel.param("multiple_scattering_lut");

    // The shader filters the column densities itself, in f32.
    const column_density: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = column_density_size[0],
        .height = column_density_size[1],
        .depth_or_layers = 1,
        .usage = lut_usage,
        .label = "column density LUT",
    });
    errdefer column_density.deinit();
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
        .column_density_lut_param = column_density_lut_param,
        .multiple_scattering_lut_param = multiple_scattering_lut_param,
        .bake_column_density = bake_column_density,
        .bake_multiple_scattering = bake_multiple_scattering,
        .column_density = column_density,
        .multiple_scattering = multiple_scattering,
        .stale = true,
    };
}

pub fn deinit(mode: *RayMarcher, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.multiple_scattering.deinit();
    mode.column_density.deinit();
    mode.bake_multiple_scattering.deinit();
    mode.bake_column_density.deinit();
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
        .{ .param = mode.column_density_lut_param, .resource = .{ .texture = mode.column_density } },
        .{ .param = mode.multiple_scattering_lut_param, .resource = .{ .texture = mode.multiple_scattering } },
    });
}

pub fn invalidate(mode: *RayMarcher, scene: *const uniforms.SceneUniforms) void {
    _ = scene;
    mode.stale = true;
}

/// Records both bakes for `scene`, column densities first since the second reads them.
pub fn bake(
    mode: *const RayMarcher,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    scene: *const uniforms.SceneUniforms,
) gpu.Error!void {
    const scene_data: gpu.CommandList.Binding.Resource = .{ .data = std.mem.asBytes(scene) };
    const ozone_cross_section: gpu.CommandList.Binding.Resource = .{ .buffer = shared.luts.ozone_cross_section };
    const ozone_profile: gpu.CommandList.Binding.Resource = .{ .buffer = shared.luts.ozone_profile };

    const column_density = mode.bake_column_density;
    try command_list.dispatch(column_density, &.{
        .{ .param = try column_density.param("scene"), .resource = scene_data },
        .{ .param = try column_density.param("ozone_cross_section"), .resource = ozone_cross_section },
        .{ .param = try column_density.param("ozone_profile"), .resource = ozone_profile },
        .{ .param = try column_density.param("column_density_target"), .resource = .{ .texture = mode.column_density } },
    }, column_density.groupCount(.{ column_density_size[0], column_density_size[1], 1 }));

    const multiple_scattering = mode.bake_multiple_scattering;
    try command_list.dispatch(multiple_scattering, &.{
        .{ .param = try multiple_scattering.param("scene"), .resource = scene_data },
        .{ .param = try multiple_scattering.param("ozone_cross_section"), .resource = ozone_cross_section },
        .{ .param = try multiple_scattering.param("ozone_profile"), .resource = ozone_profile },
        .{ .param = try multiple_scattering.param("cie"), .resource = .{ .buffer = shared.luts.cie } },
        .{ .param = try multiple_scattering.param("column_density_lut"), .resource = .{ .texture = mode.column_density } },
        .{
            .param = try multiple_scattering.param("multiple_scattering_target"),
            .resource = .{ .texture = mode.multiple_scattering },
        },
    }, multiple_scattering.groupCount(multiple_scattering_size));
}

// Copies out serve the tests.
const lut_usage: gpu.Texture.Usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true };
