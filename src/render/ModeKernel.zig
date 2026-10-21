//! A compute kernel on the globals of shaders/shared/bindings.slang, located once at
//! creation and bound on every dispatch next to the kernel's own parameters. The
//! Earth-map globals of shaders/shared/earth_bindings.slang and the tile globals of
//! shaders/earth/tiles.slang are bound too when the kernel declares them. Render modes run one thread per pixel in 8 x 8 groups
//! (`dispatch`); `Datagen` picks its own grid (`dispatchGroups`).

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const Renderer = @import("Renderer.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");

const ModeKernel = @This();

kernel: gpu.Kernel,
shared_params: std.EnumArray(SharedParam, gpu.Kernel.Param),
/// Null when the kernel does not import shared/earth_bindings.slang.
earth_params: ?EarthParams,
/// Null when the kernel does not import earth/tiles.slang.
tile_params: ?std.EnumArray(TileParam, gpu.Kernel.Param),

/// `earth_maps`, the map textures `earth_<role>`, and `earth_bounds`.
const EarthParams = struct {
    layouts: gpu.Kernel.Param,
    maps: std.EnumArray(EarthMaps.Role, gpu.Kernel.Param),
    bounds: gpu.Kernel.Param,
};

/// The globals of shaders/earth/tiles.slang, named `earth_<tag>`.
const TileParam = enum { tiles, tile_indirection, tile_heights, tile_water, tile_surface, tile_sampler, tile_requests, tile_height_max, tile_block_max, wind, wind_sampler };

/// The globals of shaders/shared/bindings.slang, named exactly as declared there.
pub const SharedParam = enum {
    frame,
    scene,
    accum,
    cie,
    srgb_to_spectrum_basis,
    ozone_cross_section,
    ozone_profile,
    column_density_lut,
    linear_clamp,
    linear_repeat_u,
};

/// MODE_GROUP_SIZE in shaders/shared/bindings.slang.
pub const group_size = 8;

/// Bindings of the Earth-map globals, for kernels that declare them.
const earth_binding_count = 2 + std.enums.values(EarthMaps.Role).len;

/// Compiles `entry` of `module`, which must import `shared.bindings`, specialized with
/// `generic_args` as `gpu.Kernel.init` describes.
pub fn init(device: gpu.Device, module: [*:0]const u8, entry: [*:0]const u8, generic_args: []const [*:0]const u8) gpu.Kernel.LayoutError!ModeKernel {
    const kernel: gpu.Kernel = try .init(device, module, entry, generic_args);
    errdefer kernel.deinit();
    try kernel.checkLayout(uniforms.FrameUniforms, "frame");
    try kernel.checkLayout(uniforms.SceneUniforms, "scene");

    var shared_params: std.EnumArray(SharedParam, gpu.Kernel.Param) = undefined;
    inline for (comptime std.enums.values(SharedParam)) |name| {
        shared_params.set(name, try kernel.param(@tagName(name)));
    }
    return .{
        .kernel = kernel,
        .shared_params = shared_params,
        .earth_params = try findEarthParams(kernel),
        .tile_params = try findTileParams(kernel),
    };
}

fn findEarthParams(kernel: gpu.Kernel) gpu.Kernel.LayoutError!?EarthParams {
    const layouts = try kernel.paramOptional("earth_maps") orelse return null;
    try kernel.checkLayout(uniforms.EarthMapUniforms, "earth_maps");
    var maps: std.EnumArray(EarthMaps.Role, gpu.Kernel.Param) = undefined;
    inline for (comptime std.enums.values(EarthMaps.Role)) |role| {
        maps.set(role, try kernel.param("earth_" ++ @tagName(role)));
    }
    return .{ .layouts = layouts, .maps = maps, .bounds = try kernel.param("earth_bounds") };
}

fn findTileParams(kernel: gpu.Kernel) gpu.Kernel.LayoutError!?std.EnumArray(TileParam, gpu.Kernel.Param) {
    _ = try kernel.paramOptional("earth_tiles") orelse return null;
    try kernel.checkLayout(uniforms.TileUniforms, "earth_tiles");
    var tiles: std.EnumArray(TileParam, gpu.Kernel.Param) = undefined;
    inline for (comptime std.enums.values(TileParam)) |name| {
        tiles.set(name, try kernel.param("earth_" ++ @tagName(name)));
    }
    return tiles;
}

pub fn deinit(mode_kernel: ModeKernel) void {
    mode_kernel.kernel.deinit();
}

/// Records one dispatch over every pixel of `frame` with the shared globals bound,
/// followed by `mode_bindings` (parameters looked up through `mode_kernel.kernel.param`).
/// The kernel must declare `[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]`.
pub fn dispatch(
    mode_kernel: ModeKernel,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
    mode_bindings: []const gpu.CommandList.Binding,
) gpu.Error!void {
    assert(std.mem.eql(u32, &mode_kernel.kernel.thread_group_size, &.{ group_size, group_size, 1 }));
    const size = frame.uniforms.frame[1..3];
    const groups = mode_kernel.kernel.groupCount(.{ size[0], size[1], 1 });
    try mode_kernel.dispatchGroups(command_list, shared, frame, mode_bindings, groups);
}

/// `dispatch` over `groups` thread groups, whatever the kernel's group size.
pub fn dispatchGroups(
    mode_kernel: ModeKernel,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
    mode_bindings: []const gpu.CommandList.Binding,
    groups: [3]u32,
) gpu.Error!void {
    const earth_count = if (mode_kernel.earth_params != null) earth_binding_count else 0;
    const tile_count = if (mode_kernel.tile_params != null) std.enums.values(TileParam).len else 0;
    assert(std.enums.values(SharedParam).len + earth_count + tile_count + mode_bindings.len <= gpu.CommandList.bindings_max);
    var buffer: [gpu.CommandList.bindings_max]gpu.CommandList.Binding = undefined;
    var bindings: std.ArrayList(gpu.CommandList.Binding) = .initBuffer(&buffer);
    inline for (comptime std.enums.values(SharedParam)) |name| {
        const resource: gpu.CommandList.Binding.Resource = switch (name) {
            .frame => .{ .data = std.mem.asBytes(&frame.uniforms) },
            .scene => .{ .data = std.mem.asBytes(&frame.scene) },
            .accum => .{ .texture = frame.accum },
            .cie => .{ .buffer = shared.luts.cie },
            .srgb_to_spectrum_basis => .{ .buffer = shared.luts.srgb_to_spectrum },
            .ozone_cross_section => .{ .buffer = shared.luts.ozone_cross_section },
            .ozone_profile => .{ .buffer = shared.luts.ozone_profile },
            .column_density_lut => .{ .texture = shared.luts.column_density },
            .linear_clamp => .{ .sampler = shared.luts.linear_clamp },
            .linear_repeat_u => .{ .sampler = shared.luts.linear_repeat_u },
        };
        bindings.appendAssumeCapacity(.{ .param = mode_kernel.shared_params.get(name), .resource = resource });
    }
    if (mode_kernel.earth_params) |earth_params| {
        const earth_maps = &shared.earth_maps;
        bindings.appendAssumeCapacity(.{ .param = earth_params.layouts, .resource = .{ .data = std.mem.asBytes(&earth_maps.layouts) } });
        for (std.enums.values(EarthMaps.Role)) |role| {
            bindings.appendAssumeCapacity(.{ .param = earth_params.maps.get(role), .resource = .{ .texture = earth_maps.textures.get(role) } });
        }
        bindings.appendAssumeCapacity(.{ .param = earth_params.bounds, .resource = .{ .texture = earth_maps.bounds } });
    }
    if (mode_kernel.tile_params) |tile_params| {
        const tiles = &shared.tiles;
        inline for (comptime std.enums.values(TileParam)) |name| {
            const resource: gpu.CommandList.Binding.Resource = switch (name) {
                .tiles => .{ .data = std.mem.asBytes(&tiles.layout) },
                .tile_indirection => .{ .buffer = tiles.indirection },
                .tile_heights => .{ .texture = tiles.heights },
                .tile_water => .{ .texture = tiles.water },
                .tile_surface => .{ .texture = tiles.surface },
                .tile_sampler => .{ .sampler = shared.luts.linear_clamp },
                .tile_requests => .{ .buffer = tiles.requests },
                .tile_height_max => .{ .buffer = tiles.height_max },
                .tile_block_max => .{ .buffer = tiles.block_max },
                .wind => .{ .texture = tiles.wind },
                .wind_sampler => .{ .sampler = shared.luts.linear_repeat_u },
            };
            bindings.appendAssumeCapacity(.{ .param = tile_params.get(name), .resource = resource });
        }
    }
    bindings.appendSliceAssumeCapacity(mode_bindings);
    try command_list.dispatch(mode_kernel.kernel, bindings.items, groups);
}
