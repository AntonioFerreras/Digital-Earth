//! The ray marcher's GPU-baked LUTs (shaders/tests/test_raymarch.slang): transmittance
//! from the column densities against a fine reference integral, the multiple-scattering
//! bake against its lookup, and physical sanity of the whole multiple-scattering table.
//! Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const gpu = @import("../gpu.zig");
const Luts = @import("../assets/Luts.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;
const GpuLuts = @import("GpuLuts.zig");
const RayMarcher = @import("RayMarcher.zig");
const Renderer = @import("Renderer.zig");
const scene = @import("scene.zig");
const uniforms = @import("uniforms.zig");

const Fixture = struct {
    device: gpu.Device,
    shared: Renderer.Shared,
    mode: RayMarcher,
    scene: uniforms.SceneUniforms,

    /// Bakes the LUTs of the `textured` preset, whose media match the original's bake.
    fn init() !Fixture {
        const device = try createTestDevice();
        errdefer device.deinit();
        var lut_dir = try std.Io.Dir.cwd().openDir(testing.io, "LUT", .{});
        defer lut_dir.close(testing.io);
        var luts: Luts = try .load(testing.allocator, testing.io, lut_dir);
        defer luts.deinit(testing.allocator);
        var shared: Renderer.Shared = .{
            .device = device,
            .luts = try .init(testing.allocator, device, &luts),
            .network = null,
            .lut4d = null,
            .earth_maps = undefined,
        };
        errdefer shared.luts.deinit(testing.allocator);
        shared.earth_maps = try .placeholder(device);
        errdefer shared.earth_maps.deinit();
        var mode: RayMarcher = try .init(testing.allocator, &shared);
        errdefer mode.deinit(testing.allocator);

        const scene_uniforms = scene.Preset.textured.sceneUniforms();
        const command_list: gpu.CommandList = try .begin(device);
        const recorded = mode.bake(command_list, &shared, &scene_uniforms);
        try command_list.submit(null);
        try recorded;
        try device.waitIdle();
        return .{ .device = device, .shared = shared, .mode = mode, .scene = scene_uniforms };
    }

    fn deinit(fixture: *Fixture) void {
        fixture.mode.deinit(testing.allocator);
        fixture.shared.earth_maps.deinit();
        fixture.shared.luts.deinit(testing.allocator);
        fixture.device.deinit();
    }

    /// Runs `entry` of the test module over `inputs`, `input_stride` floats per case.
    fn run(fixture: *Fixture, entry: [*:0]const u8, inputs: []const f32, input_stride: u32, results: []f32) !void {
        const device = fixture.device;
        const kernel: gpu.Kernel = try .init(device, "tests/test_raymarch", entry, &.{});
        defer kernel.deinit();
        const case_count: u32 = @intCast(inputs.len / input_stride);
        const input_buffer = try GpuLuts.initBuffer(device, inputs, "inputs");
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
            .{ .param = try kernel.param("scene"), .resource = .{ .data = std.mem.asBytes(&fixture.scene) } },
            .{ .param = try kernel.param("ozone_cross_section"), .resource = .{ .buffer = fixture.shared.luts.ozone_cross_section } },
            .{ .param = try kernel.param("ozone_profile"), .resource = .{ .buffer = fixture.shared.luts.ozone_profile } },
            .{ .param = try kernel.param("linear_clamp"), .resource = .{ .sampler = fixture.shared.luts.linear_clamp } },
            .{ .param = try kernel.param("column_density_lut"), .resource = .{ .texture = fixture.shared.luts.column_density } },
            .{ .param = try kernel.param("multiple_scattering_lut"), .resource = .{ .texture = fixture.mode.multiple_scattering } },
            .{ .param = try kernel.param("cie"), .resource = .{ .buffer = fixture.shared.luts.cie } },
        }, kernel.groupCount(.{ case_count, 1, 1 }));
        try command_list.submit(null);
        try dispatched;
        try output_buffer.read(device, 0, std.mem.sliceAsBytes(results));
    }
};

test "column-density LUT transmittance matches a fine integral" {
    var fixture: Fixture = try .init();
    defer fixture.deinit();

    // (mu, altitude m, wavelength nm): zenith to near-horizon rays across the spectrum,
    // ozone's Chappuis band at 600 nm, and three rays that hit the planet.
    const inputs = [_]f32{
        1.0,   0,    550.0,
        0.5,   0,    450.3,
        0.1,   0,    700.0,
        0.02,  0,    400.0,
        0.3,   2000, 600.5,
        -0.05, 10e3, 550.0,
        0.0,   30e3, 520.7,
        0.7,   90e3, 820.0,
        -0.2,  50e3, 480.0,
        -0.3,  0,    550.0,
        -0.06, 10e3, 550.0,
    };
    var results: [inputs.len / 3 * 2]f32 = undefined;
    try fixture.run("transmittance_reference", &inputs, 3, &results);
    for (0..inputs.len / 3) |case| {
        const lookup = results[2 * case];
        const reference = results[2 * case + 1];
        // The ray from 10 km that dips to a 2 km tangent crosses the aerosol profile's
        // kinks at 1.3 and 2.4 km, which the table's bilinear interpolation smooths:
        // its column is about 1% off. The other grazing rays end near 1e-5.
        const tolerance_rel: f32 = if (case == 5) 2e-2 else 1e-3;
        try testing.expectApproxEqAbs(reference, lookup, 1e-5 + tolerance_rel * reference);
    }
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, results[results.len - 6 ..]);
}

test "multiple-scattering LUT matches its bake and is physical" {
    var fixture: Fixture = try .init();
    defer fixture.deinit();

    const size = RayMarcher.multiple_scattering_size;
    // Texel centres and bundle members: the lookup there must return the bake's value
    // exactly. Slice i holds the bundle of u = i / (slices - 1).
    const texels = [_][4]u32{ .{ 31, 0, 45, 1 }, .{ 20, 5, 17, 0 }, .{ 16, 16, 87, 2 }, .{ 10, 2, 58, 3 }, .{ 0, 31, 0, 0 }, .{ 25, 28, 127, 3 } };
    var inputs: [texels.len * 4]f32 = undefined;
    for (texels, 0..) |texel, case| {
        for (0..2) |axis| {
            inputs[4 * case + axis] = (@as(f32, @floatFromInt(texel[axis])) + 0.5) / @as(f32, @floatFromInt(size[axis]));
        }
        inputs[4 * case + 2] = @as(f32, @floatFromInt(texel[2])) / @as(f32, @floatFromInt(size[2] - 1));
        inputs[4 * case + 3] = @floatFromInt(texel[3]);
    }
    var results: [texels.len * 4]f32 = undefined;
    try fixture.run("multiple_scattering_reference", &inputs, 4, &results);
    for (0..texels.len) |case| {
        const lookup, const direct, _, const transfer = results[4 * case ..][0..4].*;
        try testing.expectApproxEqRel(direct, lookup, 1e-6);
        try testing.expect(transfer > 0 and transfer < 1);
    }

    // The whole table: finite and non-negative; brighter with a higher sun, towards the
    // ground, and in blue than in red.
    const slice_size = size[0] * size[1];
    const table = try testing.allocator.alloc([4]f32, slice_size * size[2]);
    defer testing.allocator.free(table);
    try testing.expectEqual(gpu.Texture.Format.rgba32_float, fixture.mode.multiple_scattering.desc.format);
    for (0..size[2]) |slice| {
        const texels_out = table[slice * slice_size ..][0..slice_size];
        try fixture.mode.multiple_scattering.read(fixture.device, @intCast(slice), std.mem.sliceAsBytes(texels_out));
    }
    for (table) |texel| {
        for (texel) |value| try testing.expect(std.math.isFinite(value) and value >= 0);
    }
    const at = struct {
        fn at(values: []const [4]f32, u: usize, v: usize, slice: usize, member: usize) f32 {
            return values[(slice * size[1] + v) * size[0] + u][member];
        }
    }.at;
    // The middle slice's members sit at the CDF's 1/8, 3/8, 5/8, and 7/8: blue, green,
    // yellow-green, and red.
    const middle = size[2] / 2;
    for (size[0] / 2 + 1..size[0]) |u| try testing.expect(at(table, u, 0, middle, 1) > at(table, u - 1, 0, middle, 1));
    try testing.expect(at(table, size[0] - 1, size[1] - 1, middle, 1) < at(table, size[0] - 1, 0, middle, 1));
    try testing.expect(at(table, size[0] - 1, 0, middle, 0) > at(table, size[0] - 1, 0, middle, 3));
}
