//! The Earth maps of `assets.EarthMaps` on the GPU: one `Texture2DArray` per role,
//! split into longitude tiles by `assets.TilePlan` for the device's 2D limit, and the
//! tile layouts that shaders/shared/earth_bindings.slang reads from `earth_maps`.
//!
//! Until a mode samples the maps, and always without a map cache, the renderer binds
//! 1x1 placeholders instead, so kernels that declare the maps still bind.

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const gpu = @import("../gpu.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const TilePlan = @import("../assets/TilePlan.zig");
const uniforms = @import("uniforms.zig");

const GpuEarthMaps = @This();

textures: std.EnumArray(EarthMaps.Role, gpu.Texture),
/// The `earth_maps` uniforms.
layouts: uniforms.EarthMapUniforms,
/// The set `load` was asked for, also after a fallback to Q1; null for the placeholders.
quality: ?EarthMaps.Quality,

const log = std.log.scoped(.render);

/// Where `load` reads the baked maps from.
pub const Cache = struct {
    io: Io,
    /// The directory `earth bake-textures` writes, holding `earth/<quality>/<role>.raw`.
    dir: Io.Dir,
};

pub const LoadError = EarthMaps.Map.OpenError || TilePlan.PlanError ||
    gpu.Error || std.mem.Allocator.Error || error{TooManyLayers};

/// Host bytes staged per upload; bounds both the host buffer and slang-rhi's staging heap.
const upload_chunk_size = 64 << 20;

/// Uploads the maps of `quality`. Q2 falls back to Q1, with a log line, when its
/// tiles exceed the device's limits or memory.
pub fn load(gpa: std.mem.Allocator, device: gpu.Device, cache: Cache, quality: EarthMaps.Quality) LoadError!GpuEarthMaps {
    return upload(gpa, device, cache, quality) catch |err| switch (err) {
        error.MapTooTall, error.TooManyLayers, error.OutOfMemory => {
            if (quality == .q1) return err;
            log.warn("Earth maps q2 do not fit this device ({t}); falling back to q1", .{err});
            var fallback = try upload(gpa, device, cache, .q1);
            fallback.quality = quality;
            return fallback;
        },
        else => return err,
    };
}

/// One black texel per role.
pub fn placeholder(device: gpu.Device) gpu.Error!GpuEarthMaps {
    var textures: std.EnumArray(EarthMaps.Role, gpu.Texture) = .initUndefined();
    var created_count: usize = 0;
    errdefer for (std.enums.values(EarthMaps.Role)[0..created_count]) |role| textures.get(role).deinit();
    var layouts: uniforms.EarthMapUniforms = undefined;
    for (std.enums.values(EarthMaps.Role)) |role| {
        const texture = try initTexture(device, role, 1, 1, 1);
        textures.set(role, texture);
        created_count += 1;
        const zeros: [4]u8 = @splat(0);
        try texture.upload(device, .{ .x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 1 }, &zeros, zeros.len);
        layoutField(&layouts, role).* = .{ 1, 1, 1, 0 };
    }
    return .{ .textures = textures, .layouts = layouts, .quality = null };
}

pub fn deinit(earth_maps: *GpuEarthMaps) void {
    for (earth_maps.textures.values) |texture| texture.deinit();
    earth_maps.* = undefined;
}

fn upload(gpa: std.mem.Allocator, device: gpu.Device, cache: Cache, quality: EarthMaps.Quality) LoadError!GpuEarthMaps {
    const start = Io.Clock.awake.now(cache.io);
    var maps = EarthMaps.open(cache.io, cache.dir, quality) catch |err| {
        log.err("Earth map cache for {t} unreadable ({t}); bake it with `earth bake-textures --texture-quality {t}`", .{ quality, err, quality });
        return err;
    };
    defer maps.close(cache.io);
    const info = device.info();
    const limits: Limits = .{ .dimension_max = info.max_texture_dimension_2d, .layers_max = info.max_texture_array_layers };

    const chunk = try gpa.alloc(u8, upload_chunk_size);
    defer gpa.free(chunk);
    var textures: std.EnumArray(EarthMaps.Role, gpu.Texture) = .initUndefined();
    var created_count: usize = 0;
    errdefer for (std.enums.values(EarthMaps.Role)[0..created_count]) |role| textures.get(role).deinit();
    var layouts: uniforms.EarthMapUniforms = undefined;
    for (std.enums.values(EarthMaps.Role)) |role| {
        const map = maps.maps.get(role);
        const tiled = try uploadMap(device, role, map.header, map.pixels, limits, chunk);
        textures.set(role, tiled.texture);
        created_count += 1;
        layoutField(&layouts, role).* = tiled.layout;
    }
    // Releases the staging memory of the uploads.
    try device.waitIdle();
    log.info("Earth maps {t} uploaded in {d} ms", .{ quality, start.untilNow(cache.io, .awake).toMilliseconds() });
    return .{ .textures = textures, .layouts = layouts, .quality = quality };
}

pub const Limits = struct { dimension_max: u32, layers_max: u32 };

/// One map as a texture array and its field of `uniforms.EarthMapUniforms`.
pub const TiledMap = struct { texture: gpu.Texture, layout: [4]u32 };

/// Tiles one map in the cache layout (`header`, `pixels`) to fit `limits`, staging
/// through `chunk`, which must hold at least one row of a tile.
pub fn uploadMap(
    device: gpu.Device,
    role: EarthMaps.Role,
    header: EarthMaps.Header,
    pixels: []const u8,
    limits: Limits,
    chunk: []u8,
) LoadError!TiledMap {
    assert(header.channels == EarthMaps.channelCount(role));
    const plan: TilePlan = try .plan(header.width, header.height, limits.dimension_max);
    if (plan.tile_count > limits.layers_max) return error.TooManyLayers;
    const texture = try initTexture(device, role, plan.layer_width, plan.map_height, plan.tile_count);
    errdefer texture.deinit();
    try uploadTiles(device, texture, plan, header.channels, pixels, chunk);
    return .{ .texture = texture, .layout = .{ plan.map_width, plan.map_height, plan.tile_width, plan.apron } };
}

/// Copies each tile through `chunk` in bands of whole rows: the mapped cache rows are
/// wider than a tile, and slang-rhi may read up to its row alignment past a source row.
fn uploadTiles(device: gpu.Device, texture: gpu.Texture, plan: TilePlan, channels: u32, pixels: []const u8, chunk: []u8) gpu.Error!void {
    const row_size = @as(usize, plan.layer_width) * channels;
    const band_rows: u32 = @intCast(@min(chunk.len / row_size, plan.map_height));
    assert(band_rows > 0);
    for (0..plan.tile_count) |tile_index| {
        const tile: u32 = @intCast(tile_index);
        var row_first: u32 = 0;
        while (row_first < plan.map_height) : (row_first += band_rows) {
            const row_count = @min(band_rows, plan.map_height - row_first);
            const band = chunk[0 .. row_size * row_count];
            plan.copyTile(tile, channels, pixels, row_first, band);
            try texture.upload(device, .{
                .x = 0,
                .y = row_first,
                .z = tile,
                .width = plan.layer_width,
                .height = row_count,
                .depth = 1,
            }, band, row_size);
        }
    }
}

fn initTexture(device: gpu.Device, role: EarthMaps.Role, width: u32, height: u32, layers: u32) gpu.Error!gpu.Texture {
    return .init(device, .{
        .kind = .@"2d_array",
        .format = switch (EarthMaps.channelCount(role)) {
            1 => .r8_unorm,
            4 => .rgba8_unorm,
            else => unreachable,
        },
        .width = width,
        .height = height,
        .depth_or_layers = layers,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .label = @tagName(role),
    });
}

fn layoutField(layouts: *uniforms.EarthMapUniforms, role: EarthMaps.Role) *[4]u32 {
    return switch (role) {
        inline else => |tag| &@field(layouts, @tagName(tag)),
    };
}
