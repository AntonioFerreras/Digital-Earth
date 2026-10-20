//! The scattering table lookup of shaders/atmosphere/lut4d.slang
//! (shaders/tests/test_lut4d.slang) on an rgba32f upload of a small table whose
//! texels all differ: grid vertices return the stored values, midpoints the mean of
//! the neighbours, and coordinates past the ends of a range clamp to it without
//! bleeding across nu blocks or mu halves. Skipped without a GPU.

const std = @import("std");
const testing = std.testing;
const gpu = @import("../gpu.zig");
const Lut4d = @import("../assets/Lut4d.zig");
const datagen_format = @import("../assets/datagen_format.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;
const BrunetonMode = @import("BrunetonMode.zig");
const GpuLuts = @import("GpuLuts.zig");

/// Distinct sizes, so a transposed axis cannot pass.
const shape = [4]u32{ 4, 3, 6, 5 };
const mu_half = shape[2] / 2;
/// The sampler's interpolation weights have 8 fractional bits.
const filter_weight_error = 1.0 / 256.0;

const Case = struct {
    uvwz: [4]f32,
    rgb: [3]f32,
    /// The absolute error allowed per channel.
    tolerance: f32,
};

fn vertex(texel: [4]u32) [4]f32 {
    const index = ((texel[3] * shape[2] + texel[2]) * shape[1] + texel[1]) * shape[0] + texel[0];
    return datagen_format.gridPoint(shape, index);
}

fn exact(lut: *const Lut4d, texel: [4]u32, uvwz: [4]f32) Case {
    const value = lut.at(texel);
    return .{ .uvwz = uvwz, .rgb = value, .tolerance = 1e-5 * (1 + @abs(value[0])) };
}

/// Every vertex, every vertex's midpoint to its upper neighbour along each axis, and
/// the centre of each 2^4 block of vertices, except across the two mu halves.
fn interiorCases(gpa: std.mem.Allocator, lut: *const Lut4d) ![]Case {
    var cases: std.ArrayList(Case) = .empty;
    errdefer cases.deinit(gpa);
    for (0..lut.records.len) |index| {
        var texel: [4]u32 = undefined;
        var rest = index;
        for (&texel, shape) |*coordinate, size| {
            coordinate.* = @intCast(rest % size);
            rest /= size;
        }
        const value = lut.at(texel);
        try cases.append(gpa, exact(lut, texel, vertex(texel)));

        var block_complete = true;
        var upper_corner = texel;
        for (0..4) |axis| {
            const half_end = axis == 2 and texel[axis] + 1 == mu_half;
            if (texel[axis] + 1 == shape[axis] or half_end) {
                block_complete = false;
                continue;
            }
            var upper = texel;
            upper[axis] += 1;
            upper_corner[axis] += 1;
            var uvwz = vertex(texel);
            uvwz[axis] = (uvwz[axis] + vertex(upper)[axis]) / 2;
            const upper_value = lut.at(upper);
            var mean: [3]f32 = undefined;
            for (&mean, value, upper_value) |*channel, lower_channel, upper_channel| channel.* = (lower_channel + upper_channel) / 2;
            // nu interpolates in shader arithmetic, the other axes in the sampler.
            const weight_error: f32 = if (axis == 0) 1e-5 else filter_weight_error;
            try cases.append(gpa, .{ .uvwz = uvwz, .rgb = mean, .tolerance = weight_error * @abs(upper_value[0] - value[0]) + 1e-5 });
        }
        if (!block_complete) continue;
        var block_rgb: [3]f32 = @splat(0);
        var spread: f32 = 0;
        for (0..16) |corner| {
            var corner_texel = texel;
            for (&corner_texel, 0..) |*coordinate, axis| coordinate.* += @intCast(corner >> @intCast(axis) & 1);
            const corner_value = lut.at(corner_texel);
            for (&block_rgb, corner_value) |*sum, channel| sum.* += channel / 16;
            spread = @max(spread, @abs(corner_value[0] - value[0]));
        }
        var uvwz: [4]f32 = undefined;
        for (&uvwz, vertex(texel), vertex(upper_corner)) |*coordinate, lower, upper| coordinate.* = (lower + upper) / 2;
        try cases.append(gpa, .{ .uvwz = uvwz, .rgb = block_rgb, .tolerance = 3 * filter_weight_error * spread + 1e-5 });
    }
    return cases.toOwnedSlice(gpa);
}

/// Coordinates beyond the ends of a range, each expected to return one vertex.
fn clampCases(lut: *const Lut4d) [11]Case {
    const last = [4]u32{ shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1 };
    var cases: [11]Case = undefined;
    // (texel, axis, coordinate on that axis): the other axes sit on the texel's vertex.
    const specs = [cases.len]struct { [4]u32, usize, f32 }{
        // nu clamps at both ends instead of wrapping.
        .{ .{ 0, 1, 2, 3 }, 0, -0.1 },
        .{ .{ last[0], 1, 2, 3 }, 0, 1.5 },
        // mu_s clamps inside its nu block: no bleed from the block below or above.
        .{ .{ 1, 0, 2, 3 }, 1, 0.0 },
        .{ .{ 1, last[1], 2, 3 }, 1, 1.0 },
        .{ .{ 2, last[1], 2, 3 }, 1, 1.0 - 1e-5 },
        // mu clamps inside the ground and the sky half, which meet at w = 0.5.
        .{ .{ 1, 1, 0, 3 }, 2, 0.0 },
        .{ .{ 1, 1, mu_half - 1, 3 }, 2, 0.5 - 1e-5 },
        .{ .{ 1, 1, mu_half, 3 }, 2, 0.5 },
        .{ .{ 1, 1, last[2], 3 }, 2, 1.0 },
        // r clamps in the sampler.
        .{ .{ 1, 1, 2, 0 }, 3, 0.0 },
        .{ .{ 1, 1, 2, last[3] }, 3, 1.0 },
    };
    for (&cases, specs) |*case, spec| {
        const texel, const axis, const coordinate = spec;
        var uvwz = vertex(texel);
        uvwz[axis] = coordinate;
        case.* = exact(lut, texel, uvwz);
    }
    return cases;
}

/// Runs `entry` of shaders/tests/test_lut4d.slang over `inputs`, binding the harness
/// globals and `bindings` (name, resource).
fn run(
    device: gpu.Device,
    entry: [*:0]const u8,
    inputs: []const f32,
    outputs: []f32,
    bindings: []const struct { [*:0]const u8, gpu.CommandList.Binding.Resource },
) !void {
    const kernel: gpu.Kernel = try .init(device, "tests/test_lut4d", entry, &.{});
    defer kernel.deinit();
    const input_stride = 4;
    const case_count: u32 = @intCast(inputs.len / input_stride);
    const input_buffer = try GpuLuts.initBuffer(device, inputs, "inputs");
    defer input_buffer.deinit();
    const output_buffer: gpu.Buffer = try .init(device, .{
        .size = outputs.len * @sizeOf(f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "outputs",
    }, null);
    defer output_buffer.deinit();

    var all_bindings: [8]gpu.CommandList.Binding = undefined;
    all_bindings[0..3].* = .{
        .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = input_buffer } },
        .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = output_buffer } },
        .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&case_count) } },
    };
    for (all_bindings[3..][0..bindings.len], bindings) |*binding, named| {
        binding.* = .{ .param = try kernel.param(named[0]), .resource = named[1] };
    }
    const command_list: gpu.CommandList = try .begin(device);
    const dispatched = command_list.dispatch(kernel, all_bindings[0 .. 3 + bindings.len], kernel.groupCount(.{ case_count, 1, 1 }));
    try command_list.submit(null);
    try dispatched;
    try output_buffer.read(device, 0, std.mem.sliceAsBytes(outputs));
}

test "lut4d_sample returns the vertices, interpolates between them, and clamps per range" {
    const gpa = testing.allocator;
    const device = try createTestDevice();
    defer device.deinit();
    var lut: Lut4d = try .parse(try Lut4d.testFile(gpa, shape));
    defer lut.deinit(gpa);

    const interior = try interiorCases(gpa, &lut);
    defer gpa.free(interior);
    const clamped = clampCases(&lut);
    const cases = try std.mem.concat(gpa, Case, &.{ interior, &clamped });
    defer gpa.free(cases);

    const inputs = try gpa.alloc(f32, cases.len * 4);
    defer gpa.free(inputs);
    for (cases, 0..) |case, index| inputs[index * 4 ..][0..4].* = case.uvwz;
    const results = try gpa.alloc(f32, cases.len * 3);
    defer gpa.free(results);

    const texture = try BrunetonMode.initTexture(gpa, device, &lut, .rgba32_float);
    defer texture.deinit();
    const sampler: gpu.Sampler = try .init(device, .{ .filter = .linear, .address_u = .clamp, .address_v = .clamp, .address_w = .clamp });
    defer sampler.deinit();
    try run(device, "lut4d_lookup", inputs, results, &.{
        .{ "linear_clamp", .{ .sampler = sampler } },
        .{ "scattering_lut", .{ .texture = texture } },
        .{ "scattering_lut_shape", .{ .data = std.mem.asBytes(&shape) } },
    });

    var failures: u32 = 0;
    for (cases, 0..) |case, index| {
        const result = results[index * 3 ..][0..3];
        for (result, case.rgb) |actual, expected| {
            if (@abs(actual - expected) <= case.tolerance) continue;
            failures += 1;
            if (failures <= 8) std.debug.print("uvwz {any}: {any}, expected {any} +- {d}\n", .{ case.uvwz, result.*, case.rgb, case.tolerance });
            break;
        }
    }
    try testing.expectEqual(0, failures);
}

test "the ends of each uvwz range decode to the ends of the domain" {
    const device = try createTestDevice();
    defer device.deinit();
    const ranges = .{ datagen_format.mu_s_range, datagen_format.mu_ground_range, datagen_format.mu_sky_range, datagen_format.r_range };
    const mu_s, const ground, const sky, const r = ranges;
    // (uvwz, altitude m (nan: any), mu (nan: the horizon), mu_s, intersects ground)
    const cases = [_]struct { [4]f32, f32, f32, f32, f32 }{
        .{ .{ 0.5, mu_s[0], ground[0], r[1] }, 110e3, std.math.nan(f32), -0.2, 1 },
        .{ .{ 0.5, mu_s[1], ground[1], r[1] }, 110e3, -1, 1, 1 },
        // At the top of the atmosphere the zenith ray has length 0 and no direction.
        .{ .{ 0.5, mu_s[1], sky[0], (r[0] + r[1]) / 2 }, std.math.nan(f32), 1, 1, 0 },
        .{ .{ 0.5, mu_s[0], sky[1], r[1] }, 110e3, std.math.nan(f32), -0.2, 0 },
        .{ .{ 0.5, mu_s[1], sky[0], r[0] }, 0, 1, 1, 0 },
        .{ .{ 0.5, mu_s[1], sky[1], r[0] }, 0, 0, 1, 0 },
    };
    var inputs: [cases.len * 4]f32 = undefined;
    for (cases, 0..) |case, index| inputs[index * 4 ..][0..4].* = case[0];
    var results: [cases.len * 6]f32 = undefined;
    try run(device, "decode_uvwz", &inputs, &results, &.{});
    for (cases, 0..) |case, index| {
        const altitude, const mu, const mu_s_decoded, _, const intersects_ground, const horizon = results[index * 6 ..][0..6].*;
        // r is near 6.4e6 m, where f32 resolves half a metre.
        if (!std.math.isNan(case[1])) try testing.expectApproxEqAbs(case[1], altitude, 4);
        try testing.expectApproxEqAbs(if (std.math.isNan(case[2])) horizon else case[2], mu, 1e-4);
        try testing.expectApproxEqAbs(case[3], mu_s_decoded, 1e-4);
        try testing.expectEqual(case[4], intersects_ground);
    }
}

test "a table beyond the half-float range is refused" {
    const gpa = testing.allocator;
    const device = try createTestDevice();
    defer device.deinit();
    const file = try Lut4d.testFile(gpa, shape);
    const records: []datagen_format.Record = @alignCast(std.mem.bytesAsSlice(datagen_format.Record, file[@sizeOf(datagen_format.Header)..]));
    records[5].rgb[2] = 1e6;
    var lut: Lut4d = try .parse(file);
    defer lut.deinit(gpa);
    try testing.expectError(error.Lut4dOutOfRange, BrunetonMode.initTexture(gpa, device, &lut, .rgba16_float));
    const texture = try BrunetonMode.initTexture(gpa, device, &lut, .rgba32_float);
    texture.deinit();
}
