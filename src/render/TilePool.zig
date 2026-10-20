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
/// Per height-layer indirection entry, the largest height of the tile, or of its
/// nearest present ancestor where it is absent; i32 metres. Static once loaded.
height_max: gpu.Buffer,
/// One bit per indirection entry, set by lookups that wanted the tile
/// (`TileStreamer` reads and clears it).
requests: gpu.Buffer,
residency: std.EnumArray(TileSet.Layer, TileResidency),
/// The `earth_tiles` uniforms.
layout: uniforms.TileUniforms,

/// Levels resident from load to exit.
pub const base_last_level = 3;
/// The array layer limit every Vulkan device supports.
pub const slot_count_max = 2048;

/// The tile pools for layers whose last levels are `last_levels`, `slot_count` free slots
/// each.
pub fn init(gpa: std.mem.Allocator, device: gpu.Device, last_levels: std.EnumArray(TileSet.Layer, u32), slot_count: u32) (gpu.Error || error{OutOfMemory})!TilePool {
    assert(slot_count <= slot_count_max);
    var residency: std.EnumArray(TileSet.Layer, TileResidency) = .initUndefined();
    residency.set(.height, try .init(gpa, last_levels.get(.height), slot_count));
    errdefer residency.getPtr(.height).deinit(gpa);
    residency.set(.surface, try .init(gpa, last_levels.get(.surface), slot_count));
    errdefer residency.getPtr(.surface).deinit(gpa);

    const heights = try initArray(device, .r16_sint, TileSet.height_side, slot_count, "tile heights");
    errdefer heights.deinit();
    const water = try initArray(device, .r8_unorm, TileSet.height_side, slot_count, "tile water");
    errdefer water.deinit();
    const surface = try initArray(device, .rgba8_unorm, TileSet.surface_side, slot_count, "tile surface");
    errdefer surface.deinit();
    const height_entries = residency.get(.height).indirection.len;
    const indirection: gpu.Buffer = try .init(device, .{
        .size = (height_entries + residency.get(.surface).indirection.len) * @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = "tile indirection",
    }, null);
    errdefer indirection.deinit();
    const height_max: gpu.Buffer = try .init(device, .{
        .size = height_entries * @sizeOf(i32),
        .element_size = @sizeOf(i32),
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = "tile height maxima",
    }, null);
    errdefer height_max.deinit();
    const requests: gpu.Buffer = try .init(device, .{
        .size = requestWords(height_entries + residency.get(.surface).indirection.len) * @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .unordered_access = true, .copy_source = true, .copy_destination = true },
        .memory = .device,
        .label = "tile requests",
    }, null);
    return .{
        .heights = heights,
        .water = water,
        .surface = surface,
        .indirection = indirection,
        .height_max = height_max,
        .requests = requests,
        .residency = residency,
        .layout = .{
            .height = .{ last_levels.get(.height), 0, 0, 0 },
            .surface = .{ last_levels.get(.surface), @intCast(height_entries), 0, 0 },
        },
    };
}

pub fn deinit(pool: *TilePool, gpa: std.mem.Allocator) void {
    pool.requests.deinit();
    pool.height_max.deinit();
    pool.indirection.deinit();
    pool.surface.deinit();
    pool.water.deinit();
    pool.heights.deinit();
    for (&pool.residency.values) |*residency| residency.deinit(gpa);
    pool.* = undefined;
}

pub fn requestWords(entry_count: usize) usize {
    return std.math.divCeil(usize, entry_count, 32) catch unreachable;
}

fn initArray(device: gpu.Device, format: gpu.Texture.Format, side: u32, slot_count: u32, label: [*:0]const u8) gpu.Error!gpu.Texture {
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

/// The tile files of both layers, one per face, as `python/tiles/bake.py` writes them
/// under `<data>/tiles/`.
pub const Files = struct {
    sets: std.EnumArray(TileSet.Layer, [TileResidency.face_count]TileSet),

    pub const OpenError = TileSet.OpenError || error{TileSetsDisagree};

    pub fn open(io: Io, tiles_dir: Io.Dir) OpenError!Files {
        var files: Files = .{ .sets = .initUndefined() };
        var opened: usize = 0;
        errdefer for (0..opened) |index| files.sets.values[index / TileResidency.face_count][index % TileResidency.face_count].close(io);
        for (std.enums.values(TileSet.Layer)) |layer| {
            for (files.sets.getPtr(layer), 0..) |*set, face| {
                var path_buffer: [32]u8 = undefined;
                const path = std.fmt.bufPrint(&path_buffer, "{t}/face{d}.tiles", .{ layer, face }) catch unreachable;
                set.* = try .open(io, tiles_dir, path, layer);
                opened += 1;
                if (set.header.face != face or set.header.last_level != files.sets.get(layer)[0].header.last_level) {
                    return error.TileSetsDisagree;
                }
            }
        }
        return files;
    }

    pub fn close(files: *Files, io: Io) void {
        for (&files.sets.values) |*faces| for (faces) |*set| set.close(io);
        files.* = undefined;
    }

    pub fn lastLevels(files: Files) std.EnumArray(TileSet.Layer, u32) {
        return .init(.{
            .height = files.sets.get(.height)[0].header.last_level,
            .surface = files.sets.get(.surface)[0].header.last_level,
        });
    }
};

pub const LoadError = Files.OpenError || TileSet.DecodeError || gpu.Error || error{ OutOfMemory, BaseTileMissing };

/// One flat tile per face and layer at sea level, black and dry, for kernels that
/// declare the tiles while no tile files are loaded.
pub fn placeholder(gpa: std.mem.Allocator, device: gpu.Device) (gpu.Error || error{OutOfMemory})!TilePool {
    var pool: TilePool = try .init(gpa, device, .initFill(0), TileResidency.face_count);
    errdefer pool.deinit(gpa);
    const scratch = try gpa.create(Scratch);
    defer gpa.destroy(scratch);
    @memset(&scratch.height.height, 0);
    @memset(&scratch.height.water, 0);
    @memset(&scratch.rgba, 0);
    const zeros: [TileResidency.face_count]i32 = @splat(0);
    try pool.height_max.upload(device, 0, std.mem.asBytes(&zeros));
    for (0..TileResidency.face_count) |face| {
        const tile: TileResidency.Tile = .{ .face = @intCast(face), .key = .{ .level = 0, .x = 0, .y = 0 } };
        try pool.insert(device, @intCast(face), tile, .{ .height = &scratch.height });
        try pool.insert(device, @intCast(face), tile, .{ .surface = &scratch.rgba });
    }
    try pool.uploadIndirection(device);
    return pool;
}

/// A pool with levels 0 to `base_last_level` of every face of `files` resident.
pub fn load(gpa: std.mem.Allocator, device: gpu.Device, files: *const Files) LoadError!TilePool {
    var pool: TilePool = try .init(gpa, device, files.lastLevels(), slot_count_max);
    errdefer pool.deinit(gpa);
    const scratch = try gpa.create(Scratch);
    defer gpa.destroy(scratch);
    var slot: u32 = 0;
    for (std.enums.values(TileSet.Layer)) |layer| {
        for (files.sets.get(layer), 0..) |set, face| {
            const resident_last: u32 = @min(base_last_level, set.header.last_level);
            for (0..resident_last + 1) |level| {
                const side = @as(u32, 1) << @intCast(level);
                for (0..side * side) |morton| {
                    const key = pool.residency.get(layer).tileAt(face * pool.residency.get(layer).entriesPerFace() + TileSet.entryCount(@intCast(level)) + morton).key;
                    const tile = set.entry(key) orelse return error.BaseTileMissing;
                    try pool.insert(device, slot, .{ .face = @intCast(face), .key = key }, try scratch.decode(layer, set.payload(tile)));
                    slot += 1;
                }
            }
        }
        slot = 0;
    }
    try pool.uploadIndirection(device);
    try pool.uploadHeightMaxima(gpa, device, files);
    return pool;
}

/// Each height tile's maximum from its file entry; absent tiles take their parent's,
/// which bounds the parent's surface that lookups there fall back to.
fn uploadHeightMaxima(pool: *const TilePool, gpa: std.mem.Allocator, device: gpu.Device, files: *const Files) (gpu.Error || error{OutOfMemory})!void {
    const residency = pool.residency.get(.height);
    const maxima = try gpa.alloc(i32, residency.indirection.len);
    defer gpa.free(maxima);
    for (files.sets.get(.height), 0..) |set, face| {
        const base = face * residency.entriesPerFace();
        for (0..residency.entriesPerFace()) |entry_index| {
            const tile = residency.tileAt(base + entry_index);
            maxima[base + entry_index] = if (set.entry(tile.key)) |entry|
                entry.height_max
            else
                maxima[residency.entryIndex(TileResidency.parentOf(tile))];
        }
    }
    try pool.height_max.upload(device, 0, std.mem.sliceAsBytes(maxima));
}

/// Decode buffers for one tile of either layer.
pub const Scratch = struct {
    planes: [3 * TileSet.height_side * TileSet.height_side]u8,
    height: TileSet.HeightTile,
    rgb: [TileSet.surface_side * TileSet.surface_side * 3]u8,
    rgba: [TileSet.surface_side * TileSet.surface_side * 4]u8,

    pub fn decode(scratch: *Scratch, layer: TileSet.Layer, payload: []const u8) TileSet.DecodeError!TileData {
        switch (layer) {
            .height => {
                try TileSet.decodeHeight(payload, &scratch.planes, &scratch.height);
                return .{ .height = &scratch.height };
            },
            .surface => {
                try TileSet.decodeSurface(payload, &scratch.rgb);
                for (0..scratch.rgb.len / 3) |at| {
                    scratch.rgba[4 * at ..][0..4].* = .{ scratch.rgb[3 * at], scratch.rgb[3 * at + 1], scratch.rgb[3 * at + 2], 255 };
                }
                return .{ .surface = &scratch.rgba };
            },
        }
    }
};

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
