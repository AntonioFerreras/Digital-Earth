//! Tiled Earth-map sampling (shaders/shared/earth_bindings.slang) against a host
//! bilinear reference, with the map split into longitude tiles and whole. The 2D limit
//! of the test device rarely forces tiling, so the tests force it. Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const gpu = @import("../gpu.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const GpuEarthMaps = @import("GpuEarthMaps.zig");
const GpuLuts = @import("GpuLuts.zig");
const uniforms = @import("uniforms.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;

const width = 10;
const height = 4;
const group_size = 64;

/// Bilinear at texel centres with u wrapping and v clamped, as the shaders sample.
fn reference(texels: *const [height][width]u8, uv: [2]f32) f64 {
    const x = (uv[0] - @floor(uv[0])) * width - 0.5;
    const y = std.math.clamp(uv[1], 0, 1) * height - 0.5;
    const x0 = @floor(x);
    const y0 = @floor(y);
    const column0: usize = @intCast(@mod(@as(i32, @intFromFloat(x0)), width));
    const column1 = (column0 + 1) % width;
    const row0: usize = @intCast(std.math.clamp(@as(i32, @intFromFloat(y0)), 0, height - 1));
    const row1: usize = @intCast(std.math.clamp(@as(i32, @intFromFloat(y0)) + 1, 0, height - 1));
    const fx: f64 = x - x0;
    const fy: f64 = y - y0;
    const texel = struct {
        fn at(t: *const [height][width]u8, row: usize, column: usize) f64 {
            return @as(f64, @floatFromInt(t[row][column])) / 255.0;
        }
    }.at;
    const south = texel(texels, row0, column0) * (1 - fx) + texel(texels, row0, column1) * fx;
    const north = texel(texels, row1, column0) * (1 - fx) + texel(texels, row1, column1) * fx;
    return south * (1 - fy) + north * fy;
}

/// Samples `texels` at `uvs` through a map tiled for `dimension_max`; returns
/// (hardware, f32) pairs.
fn sampleTiled(device: gpu.Device, texels: *const [height][width]u8, dimension_max: u32, uvs: []const [2]f32, results: [][2]f32) !void {
    var chunk: [width * height]u8 = undefined;
    var tiled = try GpuEarthMaps.uploadMap(device, .topography, .init(width, height, 1), std.mem.asBytes(texels), .{
        .dimension_max = dimension_max,
        .layers_max = 16,
    }, &chunk);
    defer tiled.texture.deinit();
    var layouts = std.mem.zeroes(uniforms.EarthMapUniforms);
    layouts.topography = tiled.layout;

    const kernel: gpu.Kernel = try .init(device, "tests/test_earth_maps", "test_tiled_sampling", &.{});
    defer kernel.deinit();
    try kernel.checkLayout(uniforms.EarthMapUniforms, "earth_maps");
    const sampler: gpu.Sampler = try .init(device, .{ .filter = .linear, .address_u = .repeat, .address_v = .clamp, .address_w = .clamp });
    defer sampler.deinit();
    const inputs = try GpuLuts.initBuffer(device, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(uvs)), "uvs");
    defer inputs.deinit();
    const outputs: gpu.Buffer = try .init(device, .{
        .size = results.len * @sizeOf([2]f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "samples",
    }, null);
    defer outputs.deinit();
    const case_count: u32 = @intCast(uvs.len);

    const command_list: gpu.CommandList = try .begin(device);
    const recorded = command_list.dispatch(kernel, &.{
        .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = inputs } },
        .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = outputs } },
        .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&case_count) } },
        .{ .param = try kernel.param("earth_maps"), .resource = .{ .data = std.mem.asBytes(&layouts) } },
        .{ .param = try kernel.param("earth_topography"), .resource = .{ .texture = tiled.texture } },
        .{ .param = try kernel.param("linear_repeat_u"), .resource = .{ .sampler = sampler } },
    }, kernel.groupCount(.{ case_count, 1, 1 }));
    try command_list.submit(null);
    try recorded;
    try outputs.read(device, 0, std.mem.sliceAsBytes(results));
}

test "tiled maps sample like the whole map, across the seam and the poles" {
    const device = try createTestDevice();
    defer device.deinit();

    var texels: [height][width]u8 = undefined;
    var prng: std.Random.DefaultPrng = .init(7);
    for (&texels) |*row| prng.random().bytes(row);

    var uvs: [21 * 9][2]f32 = undefined;
    for (&uvs, 0..) |*uv, index| uv.* = .{
        -1.0 + @as(f32, @floatFromInt(index % 21)) * 0.137,
        -0.1 + @as(f32, @floatFromInt(index / 21)) * 0.15,
    };
    var whole: [uvs.len][2]f32 = undefined;
    var tiled: [uvs.len][2]f32 = undefined;
    try sampleTiled(device, &texels, 64, &uvs, &whole);
    // Three layers of 4 + 2 apron columns: 4, 4, and 2 owned columns.
    try sampleTiled(device, &texels, 6, &uvs, &tiled);

    for (uvs, whole, tiled) |uv, whole_sample, tiled_sample| {
        const expected = reference(&texels, uv);
        // Hardware filtering weights carry 8 fractional bits.
        try testing.expectApproxEqAbs(expected, whole_sample[0], 4e-3);
        try testing.expectApproxEqAbs(expected, tiled_sample[0], 4e-3);
        try testing.expectApproxEqAbs(expected, whole_sample[1], 1e-5);
        try testing.expectApproxEqAbs(expected, tiled_sample[1], 1e-5);
    }
}
