//! The GPU side of the cube-face tile quadtrees (shaders/earth/tiles.slang): one
//! texture-array pool per tile layout, the indirection buffer of both layers, and their
//! residency. Levels 0 to `base_last_level` of every face stay resident, so every
//! lookup lands on a tile.

const TilePool = @This();

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const gpu = @import("../gpu.zig");
const TileSet = @import("../assets/TileSet.zig");
pub const TileResidency = @import("TileResidency.zig");
const uniforms = @import("uniforms.zig");

/// `TileSet.height_side`² int16 heights per slot.
heights: gpu.Texture,
/// The height tiles' water fraction, one byte per height sample.
water: gpu.Texture,
/// `TileSet.surface_side`² RGBA8 per slot.
surface: gpu.Texture,
/// Both layers' `TileResidency.indirection`, the height layer first.
indirection: gpu.Buffer,
residency: std.EnumArray(TileSet.Layer, TileResidency),
/// The `earth_tiles` uniforms.
layout: uniforms.TileUniforms,

/// Levels resident from load to exit.
pub const base_last_level = 3;
/// The array layer limit every Vulkan device supports.
pub const slot_count = 2048;

/// The tile pools for layers whose last levels are `last_levels`, all slots free.
pub fn init(gpa: std.mem.Allocator, device: gpu.Device, last_levels: std.EnumArray(TileSet.Layer, u32)) (gpu.Error || error{OutOfMemory})!TilePool {
    var residency: std.EnumArray(TileSet.Layer, TileResidency) = .initUndefined();
    residency.set(.height, try .init(gpa, last_levels.get(.height), slot_count));
    errdefer residency.getPtr(.height).deinit(gpa);
    residency.set(.surface, try .init(gpa, last_levels.get(.surface), slot_count));
    errdefer residency.getPtr(.surface).deinit(gpa);

    const heights = try initArray(device, .r16_sint, TileSet.height_side, "tile heights");
    errdefer heights.deinit();
    const water = try initArray(device, .r8_unorm, TileSet.height_side, "tile water");
    errdefer water.deinit();
    const surface = try initArray(device, .rgba8_unorm, TileSet.surface_side, "tile surface");
    errdefer surface.deinit();
    const height_entries = residency.get(.height).indirection.len;
    const indirection: gpu.Buffer = try .init(device, .{
        .size = (height_entries + residency.get(.surface).indirection.len) * @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = "tile indirection",
    }, null);
    return .{
        .heights = heights,
        .water = water,
        .surface = surface,
        .indirection = indirection,
        .residency = residency,
        .layout = .{
            .height = .{ last_levels.get(.height), 0, 0, 0 },
            .surface = .{ last_levels.get(.surface), @intCast(height_entries), 0, 0 },
        },
    };
}

pub fn deinit(pool: *TilePool, gpa: std.mem.Allocator) void {
    pool.indirection.deinit();
    pool.surface.deinit();
    pool.water.deinit();
    pool.heights.deinit();
    for (&pool.residency.values) |*residency| residency.deinit(gpa);
    pool.* = undefined;
}

fn initArray(device: gpu.Device, format: gpu.Texture.Format, side: u32, label: [*:0]const u8) gpu.Error!gpu.Texture {
    return .init(device, .{
        .kind = .@"2d_array",
        .format = format,
        .width = side,
        .height = side,
        .depth_or_layers = slot_count,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .label = label,
    });
}

/// A decoded tile of either layer.
pub const TileData = union(TileSet.Layer) {
    height: *const TileSet.HeightTile,
    /// `TileSet.surface_side`² RGBA8, rows from the smallest v.
    surface: []const u8,
};

/// Uploads `data` for `tile` into `slot` and makes it resident; the indirection reaches
/// the GPU with `uploadIndirection`. Blocks the host like any `Texture.upload`.
pub fn insert(pool: *TilePool, device: gpu.Device, slot: u32, tile: TileResidency.Tile, data: TileData) gpu.Error!void {
    const region: gpu.Texture.Region = .{ .x = 0, .y = 0, .z = slot, .width = 0, .height = 0, .depth = 1 };
    switch (data) {
        .height => |height| {
            var at = region;
            at.width = TileSet.height_side;
            at.height = TileSet.height_side;
            try pool.heights.upload(device, at, std.mem.sliceAsBytes(&height.height), TileSet.height_side * @sizeOf(i16));
            try pool.water.upload(device, at, &height.water, TileSet.height_side);
        },
        .surface => |rgba| {
            var at = region;
            at.width = TileSet.surface_side;
            at.height = TileSet.surface_side;
            assert(rgba.len == TileSet.surface_side * TileSet.surface_side * 4);
            try pool.surface.upload(device, at, rgba, TileSet.surface_side * 4);
        },
    }
    pool.residency.getPtr(std.meta.activeTag(data)).insert(slot, tile, {}, ignoreRange);
}

fn ignoreRange(_: void, _: TileResidency.Range) void {}

/// Uploads both layers' indirection whole.
pub fn uploadIndirection(pool: TilePool, device: gpu.Device) gpu.Error!void {
    const height = pool.residency.get(.height).indirection;
    try pool.indirection.upload(device, 0, std.mem.sliceAsBytes(height));
    try pool.indirection.upload(device, height.len * @sizeOf(u32), std.mem.sliceAsBytes(pool.residency.get(.surface).indirection));
}
