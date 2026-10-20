//! Bounce sampling of the Earth BRDF (shaders/earth/brdf.slang): its pdf integrates
//! to one with the bounces reflected below the surface, and its estimate of the directional albedo matches uniform sphere sampling,
//! over ocean at grazing and steep views and over land. Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const gpu = @import("../gpu.zig");
const GpuLuts = @import("GpuLuts.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;

test "earth bounce sampling is normalized and matches uniform sampling" {
    const device = try createTestDevice();
    defer device.deinit();

    // (ocean, bathymetry, v, n): n.v is about 0.05, 0.18, and 0.86 over a tilted normal.
    const n = [3]f32{ 0.0, 0.8, 0.6 };
    const inputs = [_]f32{
        1.0, 0.0, 0.998, 0.064, -0.0005, n[0], n[1], n[2],
        1.0, 1.0, 0.954, 0.288, -0.084,  n[0], n[1], n[2],
        1.0, 0.5, 0.436, 0.84,  0.32,    n[0], n[1], n[2],
        0.8, 0.5, 0.954, 0.288, -0.084,  n[0], n[1], n[2],
        0.0, 1.0, 0.954, 0.288, -0.084,  n[0], n[1], n[2],
    };
    const case_count: u32 = inputs.len / 8;
    var results: [case_count * 4]f32 = undefined;

    const kernel: gpu.Kernel = try .init(device, "tests/test_earth", "test_earth_bounce", &.{});
    defer kernel.deinit();
    const input_buffer = try GpuLuts.initBuffer(device, &inputs, "inputs");
    defer input_buffer.deinit();
    const output_buffer: gpu.Buffer = try .init(device, .{
        .size = results.len * @sizeOf(f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "outputs",
    }, null);
    defer output_buffer.deinit();
    const command_list: gpu.CommandList = try .begin(device);
    const dispatched = command_list.dispatch(kernel, &.{
        .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = input_buffer } },
        .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = output_buffer } },
        .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&case_count) } },
    }, kernel.groupCount(.{ case_count, 1, 1 }));
    try command_list.submit(null);
    try dispatched;
    try output_buffer.read(device, 0, std.mem.sliceAsBytes(&results));

    for (0..case_count) |case| {
        const bounce, const uniform, const pdf_integral, const below_share = results[4 * case ..][0..4].*;
        // Uniform sampling of a lobe this narrow keeps about 0.5% noise at 2^20 samples.
        try testing.expectApproxEqRel(@as(f32, 1.0), pdf_integral + below_share, 0.02);
        try testing.expectApproxEqRel(uniform, bounce, 0.02);
    }
}
