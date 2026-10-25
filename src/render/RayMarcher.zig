//! The `raymarch` render mode (shaders/modes/raymarch.slang): a few marched segments
//! per pixel against two lookup tables that it bakes on the GPU
//! (shaders/modes/raymarch_lut.slang) whenever the scene preset changes, and the
//! clouds' shadow map (shaders/earth/cloud_shadow.slang), baked whenever the sun or
//! the clouds change. The tables and their parameterizations are documented in
//! shaders/atmosphere/scattering_luts.slang.

const std = @import("std");
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const ModeKernel = @import("ModeKernel.zig");
const MultipleScatteringLut = @import("MultipleScatteringLut.zig");
const Renderer = @import("Renderer.zig");

const RayMarcher = @This();

kernel: ModeKernel,
multiple_scattering_lut_param: gpu.Kernel.Param,
multiple_scattering: MultipleScatteringLut,
/// Set by `invalidate`; the next `record` rebakes the table first.
stale: bool,
cloud_shadow_map_param: gpu.Kernel.Param,
bake_cloud_shadow: ModeKernel,
cloud_shadow_target_param: gpu.Kernel.Param,
cloud_shadow: gpu.Texture,
/// What the shadow map was baked for; null before the first bake.
cloud_shadow_key: ?CloudShadowKey,

/// CLOUD_SHADOW_WIDTH, CLOUD_SHADOW_HEIGHT, and CLOUD_SHADOW_SLICES in cloud_shadow.slang.
pub const cloud_shadow_size = [3]u32{ 4096, 2048 + 128, 8 };

/// The inputs of the cloud shadow map.
const CloudShadowKey = struct {
    sun_direction: [4]f32,
    clouds: bool,
    /// Cloud extinction and density.
    media: [2]f32,
    /// The quality of the cloud coverage map; null for the placeholder.
    coverage: ?EarthMaps.Quality,

    fn init(shared: *const Renderer.Shared, frame: *const Renderer.Frame) CloudShadowKey {
        const flags: uniforms.SceneFlags = @bitCast(frame.scene.config[0]);
        return .{
            .sun_direction = frame.uniforms.sun_direction,
            .clouds = flags.clouds,
            .media = frame.scene.media[1..3].*,
            .coverage = shared.earth_maps.quality,
        };
    }
};

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) !RayMarcher {
    _ = gpa;
    const device = shared.device;
    const kernel: ModeKernel = try .init(device, "modes/raymarch", "raymarch_main", &.{});
    errdefer kernel.deinit();

    const multiple_scattering_lut_param = try kernel.kernel.param("multiple_scattering_lut");

    const bake_cloud_shadow: ModeKernel = try .init(device, "modes/cloud_shadow_bake", "bake_cloud_shadow", &.{});
    errdefer bake_cloud_shadow.deinit();
    const cloud_shadow: gpu.Texture = try .init(device, .{
        .kind = .@"2d_array",
        .format = .r16_float,
        .width = cloud_shadow_size[0],
        .height = cloud_shadow_size[1],
        .depth_or_layers = cloud_shadow_size[2],
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud shadow map",
    });
    errdefer cloud_shadow.deinit();

    const multiple_scattering: MultipleScatteringLut = try .init(device);
    return .{
        .kernel = kernel,
        .multiple_scattering_lut_param = multiple_scattering_lut_param,
        .multiple_scattering = multiple_scattering,
        .stale = true,
        .cloud_shadow_map_param = try kernel.kernel.param("cloud_shadow_map"),
        .bake_cloud_shadow = bake_cloud_shadow,
        .cloud_shadow_target_param = try bake_cloud_shadow.kernel.param("cloud_shadow_target"),
        .cloud_shadow = cloud_shadow,
        .cloud_shadow_key = null,
    };
}

pub fn deinit(mode: *RayMarcher, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.cloud_shadow.deinit();
    mode.bake_cloud_shadow.deinit();
    mode.multiple_scattering.deinit();
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
        try mode.multiple_scattering.bake(command_list, shared, &frame.scene);
        mode.stale = false;
    }
    const cloud_shadow_key: CloudShadowKey = .init(shared, frame);
    const flags: uniforms.SceneFlags = @bitCast(frame.scene.config[0]);
    if (!flags.march_cloud_shadows and (mode.cloud_shadow_key == null or !std.meta.eql(mode.cloud_shadow_key.?, cloud_shadow_key))) {
        // The map is read only where the clouds are on.
        if (cloud_shadow_key.clouds) {
            try mode.bake_cloud_shadow.dispatchGroups(command_list, shared, frame, &.{
                .{ .param = mode.cloud_shadow_target_param, .resource = .{ .texture = mode.cloud_shadow } },
            }, mode.bake_cloud_shadow.kernel.groupCount(.{ cloud_shadow_size[0], cloud_shadow_size[1], 1 }));
        }
        mode.cloud_shadow_key = cloud_shadow_key;
    }
    try mode.kernel.dispatch(command_list, shared, frame, &.{
        .{ .param = mode.multiple_scattering_lut_param, .resource = .{ .texture = mode.multiple_scattering.texture } },
        .{ .param = mode.cloud_shadow_map_param, .resource = .{ .texture = mode.cloud_shadow } },
    });
}

pub fn invalidate(mode: *RayMarcher, scene: *const uniforms.SceneUniforms) void {
    _ = scene;
    mode.stale = true;
}
