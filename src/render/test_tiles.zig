//! Tile lookups (shaders/earth/tiles.slang) over a pool holding level 0 of every face
//! and one level-1 tile: the cube mapping, the fallback to the finest resident ancestor,
//! and the sample positions within a tile. Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const gpu = @import("../gpu.zig");
const TileSet = @import("../assets/TileSet.zig");
const GpuLuts = @import("GpuLuts.zig");
const TilePool = @import("TilePool.zig");
const uniforms = @import("uniforms.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;

const side = TileSet.height_side;

/// Heights 100 * face + column on level 0, 1000 + row on the level-1 tile; water and
/// albedo constant per tile.
fn heightTile(tile: *TileSet.HeightTile, base: i16, along_rows: bool, water: u8) void {
    for (0..side) |j| {
        for (0..side) |i| tile.height[j * side + i] = base + @as(i16, @intCast(if (along_rows) j else i));
    }
    @memset(&tile.water, water);
}

fn surfaceTile(rgba: []u8, face: u8) void {
    for (0..rgba.len / 4) |at| @memcpy(rgba[4 * at ..][0..4], &[4]u8{ face * 40, 255 - face * 40, 7, 255 });
}

test "lookups find the finest resident tile and sample it at the plan's positions" {
    const device = try createTestDevice();
    defer device.deinit();
    var pool: TilePool = try .init(testing.allocator, device, .init(.{ .height = 1, .surface = 1 }), 8);
    defer pool.deinit(testing.allocator);

    const height = try testing.allocator.create(TileSet.HeightTile);
    defer testing.allocator.destroy(height);
    const rgba = try testing.allocator.alloc(u8, TileSet.surface_side * TileSet.surface_side * 4);
    defer testing.allocator.free(rgba);
    for (0..6) |face| {
        const tile: TilePool.TileResidency.Tile = .{ .face = @intCast(face), .key = .{ .level = 0, .x = 0, .y = 0 } };
        heightTile(height, @intCast(100 * face), false, @intCast(face * 50));
        try pool.insert(device, @intCast(face), tile, .{ .height = height });
        surfaceTile(rgba, @intCast(face));
        try pool.insert(device, @intCast(face), tile, .{ .surface = rgba });
    }
    // Face 2 (+Y): s along +X, t along -Z; tile (0, 0) of level 1 covers u, v < 0.5.
    heightTile(height, 1000, true, 255);
    try pool.insert(device, 6, .{ .face = 2, .key = .{ .level = 1, .x = 0, .y = 0 } }, .{ .height = height });
    try pool.uploadIndirection(device);

    const a = @tan(std.math.pi / 8.0);
    const inputs = [_][4]f32{
        // Face 4 (+Z) centre and u = 0.75: columns 2 + 96 and 2 + 144.
        .{ 0, 0, 1, 1 },
        .{ a, 0, 1, 1 },
        // Face 2, u = v = 0.25 on the level-1 tile: row 2 + 96; on level 0, column 2 + 48.
        .{ -a, 1, a, 1 },
        // The same point asking for level 0 only.
        .{ -a, 1, a, 0 },
        // Face 2, u = v = 0.75: its level-1 tile is absent, so level 0 at column 2 + 144.
        .{ a, 1, -a, 1 },
        // Face 1 (-X) centre.
        .{ -1, 0, 0, 1 },
    };
    const expected = [_][6]f32{
        .{ 498, 200.0 / 255.0, 160, 95, 4, 0.5 },
        .{ 546, 200.0 / 255.0, 160, 95, 4, 0.75 },
        .{ 1098, 1, 80, 175, 2, 0.25 },
        .{ 250, 100.0 / 255.0, 80, 175, 2, 0.25 },
        .{ 346, 100.0 / 255.0, 80, 175, 2, 0.75 },
        .{ 198, 50.0 / 255.0, 40, 215, 1, 0.5 },
    };

    const kernel: gpu.Kernel = try .init(device, "tests/test_tiles", "test_tile_lookups", &.{});
    defer kernel.deinit();
    try kernel.checkLayout(uniforms.TileUniforms, "earth_tiles");
    const sampler: gpu.Sampler = try .init(device, .{ .filter = .linear, .address_u = .clamp, .address_v = .clamp, .address_w = .clamp });
    defer sampler.deinit();
    const input_buffer = try GpuLuts.initBuffer(device, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(&inputs)), "directions");
    defer input_buffer.deinit();
    var results: [inputs.len][8]f32 = undefined;
    const outputs: gpu.Buffer = try .init(device, .{
        .size = @sizeOf(@TypeOf(results)),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "lookups",
    }, null);
    defer outputs.deinit();
    const case_count: u32 = inputs.len;
    const command_list: gpu.CommandList = try .begin(device);
    const recorded = command_list.dispatch(kernel, &.{
        .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = input_buffer } },
        .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = outputs } },
        .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&case_count) } },
        .{ .param = try kernel.param("earth_tiles"), .resource = .{ .data = std.mem.asBytes(&pool.layout) } },
        .{ .param = try kernel.param("earth_tile_indirection"), .resource = .{ .buffer = pool.indirection } },
        .{ .param = try kernel.param("earth_tile_heights"), .resource = .{ .texture = pool.heights } },
        .{ .param = try kernel.param("earth_tile_water"), .resource = .{ .texture = pool.water } },
        .{ .param = try kernel.param("earth_tile_surface"), .resource = .{ .texture = pool.surface } },
        .{ .param = try kernel.param("earth_tile_sampler"), .resource = .{ .sampler = sampler } },
    }, kernel.groupCount(.{ case_count, 1, 1 }));
    try command_list.submit(null);
    try recorded;
    try outputs.read(device, 0, std.mem.asBytes(&results));

    for (expected, results) |want, got| {
        try testing.expectApproxEqAbs(want[0], got[0], 1e-2);
        try testing.expectApproxEqAbs(want[1], got[1], 1e-3);
        try testing.expectApproxEqAbs(want[2] / 255.0, got[2], 1e-3);
        try testing.expectApproxEqAbs(want[3] / 255.0, got[3], 1e-3);
        try testing.expectEqual(want[4], got[5]);
        try testing.expectApproxEqAbs(want[5], got[6], 1e-5);
    }
}
