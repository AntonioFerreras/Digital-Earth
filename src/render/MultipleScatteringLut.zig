//! The atmosphere's multiple-scattering table (atmosphere/scattering_luts.slang),
//! baked on the GPU by shaders/modes/raymarch_lut.slang from the scene's media, for
//! the modes that march the atmosphere.

const std = @import("std");
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const Renderer = @import("Renderer.zig");

const MultipleScatteringLut = @This();

kernel: gpu.Kernel,
texture: gpu.Texture,

/// MULTIPLE_SCATTERING_LUT_SIZE in scattering_luts.slang.
pub const size = [3]u32{ 32, 32, 128 };

pub fn init(device: gpu.Device) gpu.Error!MultipleScatteringLut {
    const kernel: gpu.Kernel = try .init(device, "modes/raymarch_lut", "bake_multiple_scattering", &.{});
    errdefer kernel.deinit();
    // Bundles are interpolated in hardware, so this table needs linear filtering.
    const format: gpu.Texture.Format = if (device.info().rgba32f_filterable != 0) .rgba32_float else .rgba16_float;
    const texture: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = format,
        .width = size[0],
        .height = size[1],
        .depth_or_layers = size[2],
        // Copies out serve the tests.
        .usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true },
        .label = "multiple scattering LUT",
    });
    return .{ .kernel = kernel, .texture = texture };
}

pub fn deinit(lut: MultipleScatteringLut) void {
    lut.texture.deinit();
    lut.kernel.deinit();
}

/// Records the bake for `scene`.
pub fn bake(
    lut: MultipleScatteringLut,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    scene: *const uniforms.SceneUniforms,
) gpu.Error!void {
    const kernel = lut.kernel;
    try command_list.dispatch(kernel, &.{
        .{ .param = try kernel.param("scene"), .resource = .{ .data = std.mem.asBytes(scene) } },
        .{ .param = try kernel.param("ozone_cross_section"), .resource = .{ .buffer = shared.luts.ozone_cross_section } },
        .{ .param = try kernel.param("ozone_profile"), .resource = .{ .buffer = shared.luts.ozone_profile } },
        .{ .param = try kernel.param("cie"), .resource = .{ .buffer = shared.luts.cie } },
        .{ .param = try kernel.param("column_density_lut"), .resource = .{ .texture = shared.luts.column_density } },
        .{ .param = try kernel.param("multiple_scattering_target"), .resource = .{ .texture = lut.texture } },
    }, kernel.groupCount(size));
}
