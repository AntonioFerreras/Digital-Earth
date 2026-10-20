//! The shared Slang library against the Taichi reference values in tests/goldens,
//! driven as tests/goldens/README.md describes: one test per set, each compiling the
//! set's entry point, dispatching its inputs, and comparing every output with the set's
//! tolerance. A last test fails when a JSON set has no test here. Skipped without a GPU.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const Io = std.Io;
const gpu = @import("../gpu.zig");
const Luts = @import("../assets/Luts.zig");
const Network = @import("../assets/Network.zig");
const GpuLuts = @import("GpuLuts.zig");
const GpuNetwork = @import("GpuNetwork.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;

const golden_dir_path = "tests/goldens";
/// `harness.slang` GROUP_SIZE.
const group_size = 64;
const golden_file_size_max = 1 << 20;
/// Prefix of every set test's name, followed by "<file>.<function>".
const test_name_prefix = "golden ";

const GoldenFile = struct {
    shader: []const u8,
    sets: []const Set,
};

const Set = struct {
    function: []const u8,
    entry: []const u8,
    cases: u32,
    input_stride: u32,
    output_stride: u32,
    lut: ?struct { path: []const u8, dtype: []const u8 } = null,
    tolerance: struct { abs: f64, rel: f64 },
    inputs: []const f32,
    /// f32 values like the kernel's results, so a zero tolerance compares them exactly.
    outputs: []const f32,
};

fn loadGoldenFile(arena: std.mem.Allocator, io: Io, file: []const u8) !GoldenFile {
    const path = try std.fmt.allocPrint(arena, "{s}/{s}.json", .{ golden_dir_path, file });
    const text = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(golden_file_size_max));
    return std.json.parseFromSliceLeaky(GoldenFile, arena, text, .{ .ignore_unknown_fields = true });
}

/// Runs the set `function` of tests/goldens/`file`.json on the GPU and fails with the
/// worst mismatch: case, component, inputs, value, expected value, and tolerance.
fn expectGoldenSet(file: []const u8, function: []const u8) !void {
    const device = try createTestDevice();
    defer device.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const golden = try loadGoldenFile(arena, testing.io, file);
    const set = for (golden.sets) |set| {
        if (std.mem.eql(u8, set.function, function)) break set;
    } else return error.GoldenSetMissing;
    try testing.expectEqual(set.cases * set.input_stride, set.inputs.len);
    try testing.expectEqual(set.cases * set.output_stride, set.outputs.len);

    // "shaders/tests/test_x.slang" is the module "tests/test_x" on the search path "shaders".
    const module_path = std.mem.cutPrefix(u8, golden.shader, "shaders/").?;
    const module = try arena.dupeSentinel(u8, std.mem.cutSuffix(u8, module_path, ".slang").?, 0);
    const entry = try arena.dupeSentinel(u8, set.entry, 0);
    // A network set's kernel is specialized for the network, as the neural mode's is.
    var network: ?GpuNetwork = if (set.lut) |lut| if (std.mem.eql(u8, lut.dtype, "emlp")) try initNetwork(device, lut.path) else null else null;
    defer if (network) |*gpu_network| gpu_network.deinit(testing.allocator);
    const generic_args: []const [*:0]const u8 = if (network) |*gpu_network| &.{ gpu_network.generic_args[0], gpu_network.generic_args[1] } else &.{};
    const kernel: gpu.Kernel = try .init(device, module, entry, generic_args);
    defer kernel.deinit();

    const inputs = try GpuLuts.initBuffer(device, set.inputs, "golden inputs");
    defer inputs.deinit();
    const outputs: gpu.Buffer = try .init(device, .{
        .size = set.outputs.len * @sizeOf(f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "golden outputs",
    }, null);
    defer outputs.deinit();
    const lut: ?gpu.Buffer = if (set.lut != null and network == null) try initLut(device, set.lut.?.path) else null;
    defer if (lut) |buffer| buffer.deinit();
    const sampler: gpu.Sampler = try .init(device, .{ .filter = .linear, .address_u = .clamp, .address_v = .clamp, .address_w = .clamp });
    defer sampler.deinit();

    var buffer: [gpu.CommandList.bindings_max]gpu.CommandList.Binding = undefined;
    var bindings: std.ArrayList(gpu.CommandList.Binding) = .initBuffer(&buffer);
    bindings.appendSliceAssumeCapacity(&.{
        .{ .param = try kernel.param("inputs"), .resource = .{ .buffer = inputs } },
        .{ .param = try kernel.param("outputs"), .resource = .{ .buffer = outputs } },
        .{ .param = try kernel.param("case_count"), .resource = .{ .data = std.mem.asBytes(&set.cases) } },
    });
    if (lut) |table| bindings.appendAssumeCapacity(.{ .param = try kernel.param("lut"), .resource = .{ .buffer = table } });
    if (network) |*gpu_network| {
        var network_buffer: [GpuNetwork.binding_count_max]gpu.CommandList.Binding = undefined;
        bindings.appendSliceAssumeCapacity(try gpu_network.bindings(kernel, &network_buffer));
        bindings.appendAssumeCapacity(.{ .param = try kernel.param("linear_clamp"), .resource = .{ .sampler = sampler } });
    }
    const command_list: gpu.CommandList = try .begin(device);
    const dispatched = command_list.dispatch(kernel, bindings.items, kernel.groupCount(.{ set.cases, 1, 1 }));
    try command_list.submit(null);
    try dispatched;

    const results = try arena.alloc(f32, set.outputs.len);
    try outputs.read(device, 0, std.mem.sliceAsBytes(results));
    try expectWithinTolerance(set, results);
}

/// The network at `path` as the neural mode uploads it.
fn initNetwork(device: gpu.Device, path: []const u8) !GpuNetwork {
    var network: Network = try .load(testing.allocator, testing.io, .cwd(), path);
    defer network.deinit(testing.allocator);
    return .init(testing.allocator, device, &network);
}

/// The set's LUT as `assets.Luts` loads it, which is the layout README.md specifies.
fn initLut(device: gpu.Device, path: []const u8) !gpu.Buffer {
    var lut_dir = try Io.Dir.cwd().openDir(testing.io, "LUT", .{});
    defer lut_dir.close(testing.io);
    var luts: Luts = try .load(testing.allocator, testing.io, lut_dir);
    defer luts.deinit(testing.allocator);
    const tables = [_]struct { []const u8, []const f32 }{
        .{ "LUT/CIE.dat", luts.cie },
        .{ "LUT/srgb2spec.dat", luts.srgb_to_spectrum },
        .{ "LUT/ozone_cross_section.dat", luts.ozone_cross_section },
        .{ "LUT/ozone_profile.dat", luts.ozone_profile },
    };
    for (tables) |table| {
        if (std.mem.eql(u8, table[0], path)) return GpuLuts.initBuffer(device, table[1], "golden lut");
    }
    std.log.err("no loader for golden LUT {s}", .{path});
    return error.GoldenLutUnknown;
}

/// Passes when `|got - want| <= abs + rel * |want|` for every value.
fn expectWithinTolerance(set: Set, results: []const f32) !void {
    var failure_count: usize = 0;
    var worst_index: usize = 0;
    var worst_excess: f64 = 0;
    for (results, set.outputs, 0..) |got, want, index| {
        const allowed = set.tolerance.abs + set.tolerance.rel * @abs(want);
        const err = @abs(@as(f64, got) - @as(f64, want));
        // NaN fails by construction: no comparison with it holds.
        if (err <= allowed) continue;
        failure_count += 1;
        const excess = if (allowed > 0) err / allowed else std.math.inf(f64);
        if (failure_count == 1 or !(excess <= worst_excess)) {
            worst_excess = excess;
            worst_index = index;
        }
    }
    if (failure_count == 0) return;

    const case = worst_index / set.output_stride;
    std.log.err(
        "{s}: {d} of {d} values out of tolerance (abs {e}, rel {e}); worst: case {d} output {d} " ++
            "got {e} want {e} for inputs {any}",
        .{
            set.function,                                                failure_count,        results.len,
            set.tolerance.abs,                                           set.tolerance.rel,    case,
            worst_index % set.output_stride,                             results[worst_index], set.outputs[worst_index],
            set.inputs[case * set.input_stride ..][0..set.input_stride],
        },
    );
    return error.GoldenMismatch;
}

test "golden atmosphere.media_density" {
    try expectGoldenSet("atmosphere", "media_density");
}
test "golden atmosphere.media_extinction" {
    try expectGoldenSet("atmosphere", "media_extinction");
}
test "golden atmosphere.step_integral" {
    try expectGoldenSet("atmosphere", "step_integral");
}
test "golden atmosphere.phase" {
    try expectGoldenSet("atmosphere", "phase");
}
test "golden atmosphere.sample_phase" {
    try expectGoldenSet("atmosphere", "sample_phase");
}
test "golden atmosphere.sample_phase_edges" {
    try expectGoldenSet("atmosphere", "sample_phase_edges");
}
test "golden atmosphere.scattering_uvwz" {
    try expectGoldenSet("atmosphere", "scattering_uvwz");
}
test "golden atmosphere.scattering_uvwz_near_ground" {
    try expectGoldenSet("atmosphere", "scattering_uvwz_near_ground");
}
test "golden atmosphere.scattering_params_from_uvwz" {
    try expectGoldenSet("atmosphere", "scattering_params_from_uvwz");
}
test "golden atmosphere.scattering_params_from_ray" {
    try expectGoldenSet("atmosphere", "scattering_params_from_ray");
}
test "golden atmosphere.canonical_ray" {
    try expectGoldenSet("atmosphere", "canonical_ray");
}
test "golden atmosphere.transmittance_mu_r" {
    try expectGoldenSet("atmosphere", "transmittance_mu_r");
}
test "golden atmosphere.transmittance_uv" {
    try expectGoldenSet("atmosphere", "transmittance_uv");
}
test "golden common.ray_sphere" {
    try expectGoldenSet("common", "ray_sphere");
}
test "golden common.Rng.next4" {
    try expectGoldenSet("common", "Rng.next4");
}
test "golden common.sample_cone" {
    try expectGoldenSet("common", "sample_cone");
}
test "golden common.sample_hemisphere_cosine" {
    try expectGoldenSet("common", "sample_hemisphere_cosine");
}
test "golden common.sample_sphere_uniform" {
    try expectGoldenSet("common", "sample_sphere_uniform");
}
test "golden earth.sphere_uv" {
    try expectGoldenSet("earth", "sphere_uv");
}
test "golden earth.grade_albedo" {
    try expectGoldenSet("earth", "grade_albedo");
}
test "golden earth.earth_brdf" {
    try expectGoldenSet("earth", "earth_brdf");
}
test "golden earth.cloud_density" {
    try expectGoldenSet("earth", "cloud_density");
}
test "golden neural.legacy_mlp" {
    try expectGoldenSet("neural", "legacy_mlp");
}
test "golden neural.raw" {
    try expectGoldenSet("neural", "raw");
}
test "golden neural.fourier" {
    try expectGoldenSet("neural", "fourier");
}
test "golden neural.hash_grid" {
    try expectGoldenSet("neural", "hash_grid");
}
test "golden neural.k_planes" {
    try expectGoldenSet("neural", "k_planes");
}
test "golden neural.erf" {
    try expectGoldenSet("neural", "erf");
}
test "golden post.opendrt" {
    try expectGoldenSet("post", "opendrt");
}
test "golden post.crf_coords" {
    try expectGoldenSet("post", "crf_coords");
}
test "golden post.vignette" {
    try expectGoldenSet("post", "vignette");
}
test "golden post.srgb_oetf" {
    try expectGoldenSet("post", "srgb_oetf");
}
test "golden spectral.sample_wavelength" {
    try expectGoldenSet("spectral", "sample_wavelength");
}
test "golden spectral.sample_wavelength_tail" {
    try expectGoldenSet("spectral", "sample_wavelength_tail");
}
test "golden spectral.cie_cmf" {
    try expectGoldenSet("spectral", "cie_cmf");
}
test "golden spectral.xyz_to_rec709" {
    try expectGoldenSet("spectral", "xyz_to_rec709");
}
test "golden spectral.srgb_to_spectrum" {
    try expectGoldenSet("spectral", "srgb_to_spectrum");
}
test "golden spectral.planck" {
    try expectGoldenSet("spectral", "planck");
}

test "every golden set has a test" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var golden_dir = try Io.Dir.cwd().openDir(testing.io, golden_dir_path, .{ .iterate = true });
    defer golden_dir.close(testing.io);
    var file_count: usize = 0;
    var missing_count: usize = 0;
    var entries = golden_dir.iterate();
    while (try entries.next(testing.io)) |entry| {
        const file = std.mem.cutSuffix(u8, entry.name, ".json") orelse continue;
        file_count += 1;
        const golden = try loadGoldenFile(arena, testing.io, file);
        for (golden.sets) |set| {
            const name = try std.fmt.allocPrint(arena, test_name_prefix ++ "{s}.{s}", .{ file, set.function });
            for (builtin.test_functions) |test_function| {
                if (std.mem.endsWith(u8, test_function.name, name)) break;
            } else {
                std.log.err("{s}/{s}.json set {s} has no test \"{s}\"", .{ golden_dir_path, file, set.function, name });
                missing_count += 1;
            }
        }
    }
    try testing.expect(file_count > 0);
    try testing.expectEqual(0, missing_count);
}
