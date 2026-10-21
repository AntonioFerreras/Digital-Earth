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

    const height = try testing.allocator.create(TilePool.HeightData);
    defer testing.allocator.destroy(height);
    const rgba = try testing.allocator.alloc(u8, TileSet.surface_side * TileSet.surface_side * 4);
    defer testing.allocator.free(rgba);
    for (0..6) |face| {
        const tile: TilePool.TileResidency.Tile = .{ .face = @intCast(face), .key = .{ .level = 0, .x = 0, .y = 0 } };
        heightTile(&height.tile, @intCast(100 * face), false, @intCast(face * 50));
        height.buildBlockMaxima();
        try pool.insert(device, @intCast(face), tile, .{ .height = height });
        surfaceTile(rgba, @intCast(face));
        try pool.insert(device, @intCast(face), tile, .{ .surface = rgba });
    }
    // Face 2 (+Y): s along +X, t along -Z; tile (0, 0) of level 1 covers u, v < 0.5.
    heightTile(&height.tile, 1000, true, 255);
    height.buildBlockMaxima();
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

const stb = @import("stb");
const TileStreamer = @import("TileStreamer.zig");

/// A zstd frame holding `bytes` in one raw block (RFC 8878), as the decoder needs no more.
fn rawZstdFrame(gpa: std.mem.Allocator, bytes: []const u8) ![]u8 {
    std.debug.assert(bytes.len < 128 << 10);
    const frame = try gpa.alloc(u8, 12 + bytes.len);
    std.mem.writeInt(u32, frame[0..4], 0xFD2FB528, .little);
    frame[4] = 0b1010_0000;
    std.mem.writeInt(u32, frame[5..9], @intCast(bytes.len), .little);
    std.mem.writeInt(u24, frame[9..12], @intCast(1 | bytes.len << 3), .little);
    @memcpy(frame[12..], bytes);
    return frame;
}

/// The payload of a height tile whose samples are all `height`: the first sample's
/// residual is the height, every other one predicts it exactly.
fn flatHeightPayload(gpa: std.mem.Allocator, height: i16) ![]u8 {
    const count = TileSet.height_side * TileSet.height_side;
    const planes = try gpa.alloc(u8, 3 * count);
    defer gpa.free(planes);
    @memset(planes, 0);
    const z: u16 = @bitCast((height << 1) ^ (height >> 15));
    planes[0] = @truncate(z);
    planes[count] = @truncate(z >> 8);
    return rawZstdFrame(gpa, planes);
}

fn surfacePayload(gpa: std.mem.Allocator, grey: u8) ![]u8 {
    const surface_side = TileSet.surface_side;
    const rgb = try gpa.alloc(u8, surface_side * surface_side * 3);
    defer gpa.free(rgb);
    @memset(rgb, grey);
    var jpeg: std.ArrayList(u8) = .empty;
    errdefer jpeg.deinit(gpa);
    const Sink = struct {
        fn write(context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void {
            const list: *std.ArrayList(u8) = @ptrCast(@alignCast(context));
            const bytes: [*]const u8 = @ptrCast(data);
            list.appendSlice(std.testing.allocator, bytes[0..@intCast(size)]) catch @panic("out of memory");
        }
    };
    std.debug.assert(stb.stbi_write_jpg_to_func(Sink.write, &jpeg, surface_side, surface_side, 3, rgb.ptr, 100) != 0);
    return jpeg.toOwnedSlice(gpa);
}

/// Writes one face of one layer whose tiles are all one payload per level.
fn writeTileFile(gpa: std.mem.Allocator, dir: std.Io.Dir, layer: TileSet.Layer, face: u32, last_level: u32, payloads: []const []const u8) !void {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    const header: TileSet.Header = .{
        .magic = TileSet.Header.magic_expected,
        .version = TileSet.Header.version_expected,
        .layer = layer,
        .face = face,
        .intervals = TileSet.intervals,
        .border = TileSet.border,
        .last_level = last_level,
        .codec = if (layer == .height) .height_planes_zstd else .jpeg,
    };
    try bytes.appendSlice(gpa, std.mem.asBytes(&header));
    const count = TileSet.entryCount(last_level + 1);
    const table_end = @sizeOf(TileSet.Header) + count * @sizeOf(TileSet.Entry);
    var offsets: [16]u64 = undefined;
    var offset: u64 = table_end;
    for (payloads, 0..) |payload, level| {
        offsets[level] = offset;
        offset += payload.len;
    }
    for (0..last_level + 1) |level| {
        const tiles_across = @as(usize, 1) << @intCast(level);
        const entry: TileSet.Entry = .{ .offset = offsets[level], .size = @intCast(payloads[level].len), .height_min = 0, .height_max = 0 };
        for (0..tiles_across * tiles_across) |_| try bytes.appendSlice(gpa, std.mem.asBytes(&entry));
    }
    for (payloads) |payload| try bytes.appendSlice(gpa, payload);
    var path_buffer: [32]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{t}/face{d}.tiles", .{ layer, face });
    try dir.writeFile(testing.io, .{ .sub_path = path, .data = bytes.items });
}

test "the streamer brings in the levels lookups want and those between in one round trip" {
    const device = try createTestDevice();
    defer device.deinit();
    const gpa = testing.allocator;
    const last_level = 5;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (std.enums.values(TileSet.Layer)) |layer| try tmp.dir.createDir(testing.io, @tagName(layer), .default_dir);
    for (0..6) |face| {
        var heights: [last_level + 1][]u8 = undefined;
        var surfaces: [last_level + 1][]u8 = undefined;
        for (0..last_level + 1) |level| {
            heights[level] = try flatHeightPayload(gpa, @intCast(1000 * level + 10 * face));
            surfaces[level] = try surfacePayload(gpa, @intCast(40 * level));
        }
        defer for (heights, surfaces) |height, surface| {
            gpa.free(height);
            gpa.free(surface);
        };
        try writeTileFile(gpa, tmp.dir, .height, @intCast(face), last_level, &heights);
        try writeTileFile(gpa, tmp.dir, .surface, @intCast(face), last_level, &surfaces);
    }
    var files: TilePool.Files = try .open(testing.io, tmp.dir);
    defer files.close();
    var pool: TilePool = try .load(gpa, device, &files);
    defer pool.deinit(gpa);
    var streamer: TileStreamer = try .init(gpa, device, &pool);
    defer streamer.deinit(gpa);

    const kernel: gpu.Kernel = try .init(device, "tests/test_tiles", "test_tile_lookups", &.{});
    defer kernel.deinit();
    const sampler: gpu.Sampler = try .init(device, .{ .filter = .linear, .address_u = .clamp, .address_v = .clamp, .address_w = .clamp });
    defer sampler.deinit();
    // Face 4 (+Z) wanting level 5, and face 0 (+X) wanting level 2, resident from the start.
    const inputs = [_][4]f32{ .{ 0.1, 0.2, 1, last_level }, .{ 1, 0.3, -0.2, 2 } };
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

    var heights_seen: [16]f32 = undefined;
    var frames: usize = 0;
    while (frames < heights_seen.len) : (frames += 1) {
        _ = try streamer.update(gpa, device, &pool, &files);
        const command_list: gpu.CommandList = try .begin(device);
        const recorded = record: {
            streamer.recordUploads(command_list, &pool) catch |err| break :record err;
            command_list.dispatch(kernel, &.{
                .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = input_buffer } },
                .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = outputs } },
                .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&case_count) } },
                .{ .param = try kernel.param("earth_tiles"), .resource = .{ .data = std.mem.asBytes(&pool.layout) } },
                .{ .param = try kernel.param("earth_tile_indirection"), .resource = .{ .buffer = pool.indirection } },
                .{ .param = try kernel.param("earth_tile_heights"), .resource = .{ .texture = pool.heights } },
                .{ .param = try kernel.param("earth_tile_water"), .resource = .{ .texture = pool.water } },
                .{ .param = try kernel.param("earth_tile_surface"), .resource = .{ .texture = pool.surface } },
                .{ .param = try kernel.param("earth_tile_sampler"), .resource = .{ .sampler = sampler } },
                .{ .param = try kernel.param("earth_tile_requests"), .resource = .{ .buffer = pool.requests } },
                .{ .param = try kernel.param("earth_tile_block_max"), .resource = .{ .buffer = pool.block_max } },
            }, kernel.groupCount(.{ case_count, 1, 1 })) catch |err| break :record err;
            break :record streamer.recordFeedback(command_list, &pool);
        };
        const signal = streamer.signal();
        try command_list.submit(signal);
        streamer.submitted();
        try recorded;
        try signal.fence.wait(device, signal.value);
        try outputs.read(device, 0, std.mem.asBytes(&results));
        heights_seen[frames] = results[0][0];
        try testing.expectEqual(2000, results[1][0]);
    }
    // Level 3 at first, then levels 4 and 5 together a readback round trip later.
    try testing.expectEqual(3040, heights_seen[0]);
    try testing.expectEqual(5040, heights_seen[heights_seen.len - 1]);
    var levels_seen: usize = 0;
    for (heights_seen[1..], heights_seen[0 .. heights_seen.len - 1]) |height, previous| {
        if (height != previous) levels_seen += 1;
    }
    try testing.expectEqual(1, levels_seen);
    // Grey 200 of level 5 reaches the surface layer too.
    try testing.expectApproxEqAbs(200.0 / 255.0, results[0][2], 2.0 / 255.0);
}
