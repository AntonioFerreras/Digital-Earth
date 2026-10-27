//! The `cloud_bench` render mode (shaders/modes/cloud_bench.slang): the cloud volume
//! that `Renderer.Shared.cloud_volume` names, path-traced as the reference and marched
//! with the transfer probes as the approximation, each into an accumulation of its
//! own, then resolved into the frame as `Renderer.Frame.cloud_bench`'s view asks. Each
//! accumulation restarts only when its own inputs change, so the reference keeps
//! converging while the approximation's settings are tuned, and only what the view
//! shows is rendered. The light, the sky's radiance and the sun's irradiance at the
//! cloud (shaders/modes/cloud_bench_sky.slang), is rebaked whenever the sun or the
//! atmosphere changes, and the background, the sky through the camera at a quarter of
//! the frame, whenever the reference restarts. The transfer probes (shaders/modes/cloud_bench_probes.slang)
//! bake and fit a few rounds per frame while the approximation reads them, up to the bench's walk limit, restart only when the
//! medium or their layout (ProbeLayout.zig) changes, and are relit when the light
//! does. Their sums resume from ProbeCache's for the same settings, and are stored
//! there, read back from the GPU, when the bake stops, its settings change, or the
//! bench closes.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const CloudVolume = @import("../assets/CloudVolume.zig");
const LayoutDepths = @import("LayoutDepths.zig");
const uniforms = @import("uniforms.zig");
const scene = @import("scene.zig");
const ModeKernel = @import("ModeKernel.zig");
const MultipleScatteringLut = @import("MultipleScatteringLut.zig");
const ProbeCache = @import("ProbeCache.zig");
const ProbeLayout = @import("ProbeLayout.zig");
const BlueNoise = @import("BlueNoise.zig");
const Renderer = @import("Renderer.zig");

const CloudBench = @This();

/// `resolve_main`, which writes the frame.
kernel: ModeKernel,
resolve_params: Params,
reference: BenchKernel,
approximation: BenchKernel,
sky_bake: BenchKernel,
sky_view_bake: BenchKernel,
sun_bake: BenchKernel,
probe_bake: BenchKernel,
photon_clear: BenchKernel,
photon_trace: BenchKernel,
photon_counter_reset: BenchKernel,
sky_transfer_bake: BenchKernel,
/// The photons `trace_photons` has handed out in its dispatch.
photon_counter: gpu.Buffer,
sun_range_reset: BenchKernel,
probe_range: BenchKernel,
/// `range_probes`' least and greatest of each quantised coefficient.
sun_ranges: gpu.Buffer,
probe_fit: BenchKernel,
probe_lighting: BenchKernel,
probe_fill: BenchKernel,
sun_least_bake: BenchKernel,
sun_reach_bake: BenchKernel,
density: gpu.Texture,
majorant: gpu.Texture,
/// Of cells 2 and 4 times as large per axis.
majorant_2: gpu.Texture,
majorant_4: gpu.Texture,
sky: gpu.Texture,
sun: gpu.Texture,
/// Where the walks may skip the sun: the least optical depth towards any sun bin, per
/// cell and then over each cell's neighbours, as of `sun_reach_extinction`.
sun_least: gpu.Texture,
sun_reach: gpu.Texture,
blue_noise: gpu.Texture,
sun_reach_extinction: ?f32,
/// The volume's occupancy, from which ProbeLayout builds layouts.
layout_depths: LayoutDepths,
/// Null until the first frame lays them out.
probes: ?Probes,
/// A layout asked for, built once asked for `layout_settle_frames` frames running, so
/// that dragging a slider builds none of the layouts it passes.
layout_request: ?struct { key: LayoutKey, frames: u32 },
/// The sky's march reads the atmosphere's table, rebaked when the preset changes.
atmosphere: MultipleScatteringLut,
atmosphere_stale: bool,
/// The march also declares the ray marcher's cloud shadow map, which the bench's
/// cloudless sky never reads.
cloud_shadow_placeholder: gpu.Texture,
/// The volume's frame and grid, as `uniforms.CloudBenchUniforms` carries them.
origin: [4]f32,
grid_min: [4]f32,
grid_size: [4]u32,
/// Null until the first frame sizes them.
accumulations: ?Accumulations,
/// What the light was baked for.
light_key: ?LightKey,
/// What the probes' walks were baked for, how many each probe has, and how many of
/// them ProbeCache holds.
probe_key: ?ProbeCache.Key,
probe_walks: u32,
probe_cached_walks: u32,
/// Where the bake stops, or zero while the probes are off.
probe_walk_limit: u32,
/// ProbeCache's data root, and the device to read the sums back from when the bench
/// closes.
data: struct { io: std.Io, dir: std.Io.Dir },
/// Owned by the renderer.
probe_field: Renderer.ProbeField,
device: gpu.Device,
/// What the probes were last fitted and lit for.
probe_fit_key: ?ProbeFitKey,
probe_light_key: ?ProbeLightKey,
reference_key: ?Key,
/// The approximation also restarts with the scene's model constants, with how it reads
/// the probes, and each time they double their walks, so that their first noisy
/// estimates do not linger in its mean; null for the model.
approximation_key: ?struct { key: Key, scene: uniforms.SceneUniforms, probes: ?struct { flags: u32, layout: LayoutKey, doublings: u32 } },

/// The bench's globals that a kernel may declare, named as in the shaders.
const Global = enum {
    bench,
    bench_density,
    bench_majorant,
    bench_majorant_2,
    bench_majorant_4,
    bench_reference,
    bench_approximation,
    bench_sky,
    bench_sky_target,
    bench_sky_view,
    bench_sky_view_target,
    bench_sun,
    bench_sun_target,
    bench_probe_positions,
    bench_probe_cells,
    bench_probe_corners,
    bench_probe_transfer,
    bench_probe_sun_bins,
    bench_probe_sun_bins_raw,
    bench_photon_counter,
    bench_probe_volumes,
    bench_probe_sun,
    bench_probe_sun_ranges,
    bench_probe_sky,
    bench_probe_light_values,
    bench_probe_light,
    bench_probe_light_target,
    blue_noise,
    bench_sun_least,
    bench_sun_reach,
    multiple_scattering_lut,
    cloud_shadow_map,
};
const Params = std.EnumArray(Global, ?gpu.Kernel.Param);

const BenchKernel = struct {
    mode_kernel: ModeKernel,
    params: Params,

    fn init(device: gpu.Device, module: [*:0]const u8, entry: [*:0]const u8) gpu.Kernel.LayoutError!BenchKernel {
        const mode_kernel: ModeKernel = try .init(device, module, entry, &.{});
        errdefer mode_kernel.deinit();
        return .{ .mode_kernel = mode_kernel, .params = try findParams(mode_kernel.kernel) };
    }
};

const Accumulations = struct {
    size: [2]u32,
    reference: gpu.Texture,
    approximation: gpu.Texture,
    /// The background, a texel per SKY_VIEW_STRIDE pixels.
    sky_view: gpu.Texture,
    /// Samples in each.
    reference_samples: u32 = 0,
    approximation_samples: u32 = 0,

    fn deinit(accumulations: Accumulations) void {
        accumulations.sky_view.deinit();
        accumulations.reference.deinit();
        accumulations.approximation.deinit();
    }
};

const LightKey = struct { sun_direction: [4]f32, media: [4]f32 };
/// What a layout is built for: zero budget and ratio for the dense grid.
const LayoutKey = struct { adaptive: bool, budget: u32, depth_ratio: f32 };
const ProbeFitKey = struct { probe: ProbeCache.Key, walks: u32, sun_order3: bool, sun_linear: bool };
const ProbeLightKey = struct { probe: ProbeCache.Key, walks: u32, light: LightKey, flags: u32 };

/// What an accumulation's image depends on: the camera, the sun, the atmosphere, and
/// the bench but for its view, its probes, and its sample counts.
const Key = struct {
    frame: uniforms.FrameUniforms,
    media: [4]f32,
    bench: uniforms.CloudBenchUniforms,

    fn init(frame: *const Renderer.Frame) Key {
        var key: Key = .{ .frame = frame.uniforms, .media = frame.scene.media, .bench = frame.cloud_bench };
        key.frame.frame[0] = 0;
        key.bench.options[0] = 0;
        key.bench.options[3] = 0;
        var flags: uniforms.CloudBenchFlags = @bitCast(key.bench.options[2]);
        flags.probe_gather = false;
        flags.forward_peak = false;
        flags.p1_closure = false;
        flags.ratio_tracking = false;
        flags.adaptive_probes = false;
        flags.photons = false;
        flags.leaf_lengths = false;
        flags.sun_order3 = false;
        key.bench.options[2] = @bitCast(flags);
        key.bench.probe_origin = @splat(0);
        key.bench.probe_spacing = @splat(0);
        key.bench.probe_cells = @splat(0);
        key.bench.probe_counts = @splat(0);
        return key;
    }
};

/// BENCH_SKY_SIZE in shaders/clouds/bench_volume.slang.
const sky_size = [2]u32{ 256, 128 };
/// SKY_VIEW_STRIDE in shaders/modes/cloud_bench_sky.slang.
const sky_view_stride = 4;
/// SH3_COUNT in shaders/common/spherical_harmonics.slang.
const sh_count = 16;
/// Probes per axis of the dense layout.
const dense_resolution = 64;
/// The colour channels, each a third of `Probes.atlas` along z.
const channel_count = 3;
/// Leaves per row and per layer of `Probes.atlas`, of two texels per axis each.
const atlas_leaves_across = 64;
const layout_settle_frames = 10;
/// While the probes bake, their fit and light follow the walks only every this many
/// frames, as each costs about a frame's rounds.
const refit_frames = 4;
/// SUN_DEPTH_STRIDE in shaders/modes/cloud_bench_probes.slang.
const sun_depth_stride = 8;
/// SUN_BIN_COUNT in shaders/modes/cloud_bench_probes.slang; each bin sums four floats.
const sun_bin_count = 64;
/// SunTransfer in shaders/modes/cloud_bench_probes.slang: a float, then SUN_QUANTIZED
/// bytes of coefficients packed into uints.
const sun_quantized = 35 + 3 * 4;
const sun_transfer_bytes = @sizeOf(f32) + (sun_quantized + 3) / 4 * @sizeOf(u32);
/// SkyTransfer there: a float, then the other 4 SH3_COUNT - 1 values as bytes.
const sky_transfer_bytes = @sizeOf(f32) + (4 * sh_count - 1 + 3) / 4 * @sizeOf(u32);

/// A layout's probes on the GPU: where they lie, how a point finds its leaf, and the
/// walks' sums, fits, and light of each.
const Probes = struct {
    key: LayoutKey,
    origin: [3]f32,
    spacing: [3]f32,
    cells: [3]u32,
    count: u32,
    leaf_count: u32,
    positions: gpu.Buffer,
    cell_leaves: gpu.Buffer,
    corners: gpu.Buffer,
    /// Per probe, the integral of its trilinear weight over its leaves.
    volumes: gpu.Buffer,
    // Both copy both ways, for ProbeCache.
    transfer: gpu.Buffer,
    sun_bins: gpu.Buffer,
    sun: gpu.Buffer,
    sky: gpu.Buffer,
    light: gpu.Buffer,
    /// Each leaf's corners' light, for the approximation's filter.
    atlas: gpu.Texture,

    fn init(device: gpu.Device, key: LayoutKey, layout: *const ProbeLayout) gpu.Error!Probes {
        const count: u32 = @intCast(layout.positions.len);
        const leaf_count: u32 = @intCast(layout.corners.len);
        const positions = try initBuffer(device, std.mem.sliceAsBytes(layout.positions), @sizeOf([4]f32), "cloud bench probe positions");
        errdefer positions.deinit();
        const cell_leaves = try initBuffer(device, std.mem.sliceAsBytes(layout.cell_leaves), @sizeOf(u32), "cloud bench probe cells");
        errdefer cell_leaves.deinit();
        const corners = try initBuffer(device, std.mem.sliceAsBytes(layout.corners), @sizeOf(u32), "cloud bench probe corners");
        errdefer corners.deinit();
        const probe_volumes = try std.heap.page_allocator.alloc(f32, count);
        defer std.heap.page_allocator.free(probe_volumes);
        @memset(probe_volumes, 0);
        for (layout.corners) |corner| {
            var volume: f32 = 1;
            for (0..3) |axis| volume *= layout.positions[corner[@as(usize, 1) << @intCast(axis)]][axis] - layout.positions[corner[0]][axis];
            for (corner) |probe| probe_volumes[probe] += volume / 8;
        }
        const volumes = try initBuffer(device, std.mem.sliceAsBytes(probe_volumes), @sizeOf(f32), "cloud bench probe volumes");
        errdefer volumes.deinit();
        const sums_usage: gpu.Buffer.Usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true, .copy_destination = true };
        const transfer: gpu.Buffer = try .init(device, .{
            .size = @as(u64, count) * sh_count * 4 * @sizeOf(f32),
            .element_size = 4 * @sizeOf(f32),
            .usage = sums_usage,
            .memory = .device,
            .label = "cloud bench probe transfer",
        }, null);
        errdefer transfer.deinit();
        const sun_bins: gpu.Buffer = try .init(device, .{
            .size = @as(u64, count) * sun_bin_count * 4 * @sizeOf(f32),
            .element_size = 4 * @sizeOf(f32),
            .usage = sums_usage,
            .memory = .device,
            .label = "cloud bench probe sun bins",
        }, null);
        errdefer sun_bins.deinit();
        const sun: gpu.Buffer = try .init(device, .{
            .size = @as(u64, count) * sun_transfer_bytes,
            .element_size = sun_transfer_bytes,
            .usage = .{ .shader_resource = true, .unordered_access = true },
            .memory = .device,
            .label = "cloud bench probe sun transfer",
        }, null);
        errdefer sun.deinit();
        const sky: gpu.Buffer = try .init(device, .{
            .size = @as(u64, count) * sky_transfer_bytes,
            .element_size = sky_transfer_bytes,
            .usage = .{ .shader_resource = true, .unordered_access = true },
            .memory = .device,
            .label = "cloud bench probe sky transfer",
        }, null);
        errdefer sky.deinit();
        const light: gpu.Buffer = try .init(device, .{
            .size = @as(u64, count) * channel_count * 4 * @sizeOf(f32),
            .element_size = 4 * @sizeOf(f32),
            .usage = .{ .shader_resource = true, .unordered_access = true },
            .memory = .device,
            .label = "cloud bench probe light",
        }, null);
        errdefer light.deinit();
        const layers = std.math.divCeil(u32, leaf_count, atlas_leaves_across * atlas_leaves_across) catch unreachable;
        const atlas: gpu.Texture = try .init(device, .{
            .kind = .@"3d",
            .format = .rgba16_float,
            .width = 2 * atlas_leaves_across,
            .height = 2 * atlas_leaves_across,
            .depth_or_layers = 2 * layers * channel_count,
            .usage = .{ .shader_resource = true, .unordered_access = true },
            .label = "cloud bench probe atlas",
        });
        return .{
            .key = key,
            .origin = layout.origin,
            .spacing = layout.spacing,
            .cells = layout.cells,
            .count = count,
            .leaf_count = leaf_count,
            .positions = positions,
            .cell_leaves = cell_leaves,
            .corners = corners,
            .volumes = volumes,
            .transfer = transfer,
            .sun_bins = sun_bins,
            .sun = sun,
            .sky = sky,
            .light = light,
            .atlas = atlas,
        };
    }

    fn initBuffer(device: gpu.Device, data: []const u8, element_size: u32, label: [*:0]const u8) gpu.Error!gpu.Buffer {
        return .init(device, .{ .size = data.len, .element_size = element_size, .usage = .{ .shader_resource = true }, .memory = .device, .label = label }, data);
    }

    fn deinit(probes: Probes) void {
        probes.atlas.deinit();
        probes.light.deinit();
        probes.sky.deinit();
        probes.sun.deinit();
        probes.sun_bins.deinit();
        probes.transfer.deinit();
        probes.volumes.deinit();
        probes.corners.deinit();
        probes.cell_leaves.deinit();
        probes.positions.deinit();
    }
};

pub const InitError = gpu.Kernel.LayoutError || CloudVolume.LoadError || std.mem.Allocator.Error || error{CloudVolumeMissing};

const log = std.log.scoped(.cloud_bench);

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) InitError!CloudBench {
    const file = shared.cloud_volume orelse {
        log.err("the cloud bench needs a cloud volume: pass --cloud-volume <file.ecld>, or convert one into " ++
            CloudVolume.directory_path ++ " with python/clouds/convert_vdb.py and name it in the config's cloud_volume", .{});
        return error.CloudVolumeMissing;
    };
    var volume = CloudVolume.load(gpa, file.io, file.dir, file.path) catch |err| {
        log.err("{s}: {t}", .{ file.path, err });
        return err;
    };
    defer volume.deinit(gpa);
    const device = shared.device;

    const kernel: ModeKernel = try .init(device, "modes/cloud_bench", "resolve_main", &.{});
    errdefer kernel.deinit();
    const reference: BenchKernel = try .init(device, "modes/cloud_bench", "reference_main");
    errdefer reference.mode_kernel.deinit();
    const approximation: BenchKernel = try .init(device, "modes/cloud_bench", "approximation_main");
    errdefer approximation.mode_kernel.deinit();
    const sky_bake: BenchKernel = try .init(device, "modes/cloud_bench_sky", "bake_sky");
    errdefer sky_bake.mode_kernel.deinit();
    const sky_view_bake: BenchKernel = try .init(device, "modes/cloud_bench_sky", "bake_sky_view");
    errdefer sky_view_bake.mode_kernel.deinit();
    const sun_bake: BenchKernel = try .init(device, "modes/cloud_bench_sky", "bake_sun");
    errdefer sun_bake.mode_kernel.deinit();
    const probe_bake: BenchKernel = try .init(device, "modes/cloud_bench_probes", "bake_probes");
    errdefer probe_bake.mode_kernel.deinit();
    const photon_clear: BenchKernel = try .init(device, "modes/cloud_bench_probes", "clear_probe_sums");
    errdefer photon_clear.mode_kernel.deinit();
    const photon_trace: BenchKernel = try .init(device, "modes/cloud_bench_probes", "trace_photons");
    errdefer photon_trace.mode_kernel.deinit();
    const photon_counter_reset: BenchKernel = try .init(device, "modes/cloud_bench_probes", "reset_photon_counter");
    errdefer photon_counter_reset.mode_kernel.deinit();
    const sky_transfer_bake: BenchKernel = try .init(device, "modes/cloud_bench_probes", "bake_sky_transfer");
    errdefer sky_transfer_bake.mode_kernel.deinit();
    const photon_counter: gpu.Buffer = try .init(device, .{
        .size = @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .memory = .device,
        .label = "cloud bench photon counter",
    }, null);
    errdefer photon_counter.deinit();
    const sun_range_reset: BenchKernel = try .init(device, "modes/cloud_bench_probes", "reset_sun_ranges");
    errdefer sun_range_reset.mode_kernel.deinit();
    const probe_range: BenchKernel = try .init(device, "modes/cloud_bench_probes", "range_probes");
    errdefer probe_range.mode_kernel.deinit();
    const sun_ranges: gpu.Buffer = try .init(device, .{
        .size = 2 * sun_quantized * @sizeOf(u32),
        .element_size = @sizeOf(u32),
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .memory = .device,
        .label = "cloud bench sun ranges",
    }, null);
    errdefer sun_ranges.deinit();
    const probe_fit: BenchKernel = try .init(device, "modes/cloud_bench_probes", "fit_probes");
    errdefer probe_fit.mode_kernel.deinit();
    const probe_lighting: BenchKernel = try .init(device, "modes/cloud_bench_probes", "light_probes");
    errdefer probe_lighting.mode_kernel.deinit();
    const probe_fill: BenchKernel = try .init(device, "modes/cloud_bench_probes", "fill_probe_atlas");
    errdefer probe_fill.mode_kernel.deinit();
    const sun_least_bake: BenchKernel = try .init(device, "modes/cloud_bench_probes", "bake_sun_least");
    errdefer sun_least_bake.mode_kernel.deinit();
    const sun_reach_bake: BenchKernel = try .init(device, "modes/cloud_bench_probes", "bake_sun_reach");
    errdefer sun_reach_bake.mode_kernel.deinit();
    const atmosphere: MultipleScatteringLut = try .init(device);
    errdefer atmosphere.deinit();

    const density = try uploadVolume(device, volume.size, volume.density, .{ .shader_resource = true }, "cloud bench density");
    errdefer density.deinit();
    const majorant = try uploadVolume(device, volume.cells, volume.majorant, .{ .shader_resource = true }, "cloud bench majorants");
    errdefer majorant.deinit();
    const majorant_2 = try uploadCoarserMajorants(gpa, device, volume.cells, volume.majorant, 2, "cloud bench majorants x2");
    errdefer majorant_2.deinit();
    const majorant_4 = try uploadCoarserMajorants(gpa, device, volume.cells, volume.majorant, 4, "cloud bench majorants x4");
    errdefer majorant_4.deinit();
    const sky: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba16_float,
        .width = sky_size[0],
        .height = sky_size[1],
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud bench sky",
    });
    errdefer sky.deinit();
    const sun: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = 1 + sh_count,
        .height = 1,
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud bench sun",
    });
    errdefer sun.deinit();
    var depth_cells: [3]u32 = undefined;
    for (&depth_cells, volume.size) |*cells, size| cells.* = std.math.divCeil(u32, size, sun_depth_stride) catch unreachable;
    const sun_least: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = .r16_float,
        .width = depth_cells[0],
        .height = depth_cells[1],
        .depth_or_layers = depth_cells[2],
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud bench sun least",
    });
    errdefer sun_least.deinit();
    const sun_reach: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = .r16_float,
        .width = depth_cells[0],
        .height = depth_cells[1],
        .depth_or_layers = depth_cells[2],
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud bench sun reach",
    });
    errdefer sun_reach.deinit();
    const blue_noise: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba8_unorm,
        .width = BlueNoise.size,
        .height = BlueNoise.size,
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .label = "cloud bench blue noise",
    });
    errdefer blue_noise.deinit();
    try blue_noise.upload(device, .{ .x = 0, .y = 0, .z = 0, .width = BlueNoise.size, .height = BlueNoise.size, .depth = 1 }, &BlueNoise.generate(1), BlueNoise.size * BlueNoise.channels);
    var layout_depths: LayoutDepths = try .init(device, &volume, density);
    errdefer layout_depths.deinit();
    const cloud_shadow_placeholder: gpu.Texture = try .init(device, .{
        .kind = .@"2d_array",
        .format = .r16_float,
        .width = 1,
        .height = 1,
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true },
        .label = "cloud bench shadow placeholder",
    });
    return .{
        .kernel = kernel,
        .resolve_params = try findParams(kernel.kernel),
        .reference = reference,
        .approximation = approximation,
        .sky_bake = sky_bake,
        .sky_view_bake = sky_view_bake,
        .sun_bake = sun_bake,
        .probe_bake = probe_bake,
        .photon_clear = photon_clear,
        .photon_trace = photon_trace,
        .photon_counter_reset = photon_counter_reset,
        .sky_transfer_bake = sky_transfer_bake,
        .photon_counter = photon_counter,
        .sun_range_reset = sun_range_reset,
        .probe_range = probe_range,
        .sun_ranges = sun_ranges,
        .probe_fit = probe_fit,
        .probe_lighting = probe_lighting,
        .probe_fill = probe_fill,
        .sun_least_bake = sun_least_bake,
        .sun_reach_bake = sun_reach_bake,
        .density = density,
        .majorant = majorant,
        .majorant_2 = majorant_2,
        .majorant_4 = majorant_4,
        .sky = sky,
        .sun = sun,
        .sun_least = sun_least,
        .sun_reach = sun_reach,
        .blue_noise = blue_noise,
        .sun_reach_extinction = null,
        .layout_depths = layout_depths,
        .probes = null,
        .layout_request = null,
        .atmosphere = atmosphere,
        .atmosphere_stale = true,
        .cloud_shadow_placeholder = cloud_shadow_placeholder,
        .origin = scene.CloudBench.origin(volume.min_corner),
        .grid_min = .{ volume.min_corner[0], volume.min_corner[1], volume.min_corner[2], volume.voxel },
        .grid_size = .{ volume.size[0], volume.size[1], volume.size[2], volume.cell },
        .accumulations = null,
        .light_key = null,
        .probe_key = null,
        .probe_walks = 0,
        .probe_cached_walks = 0,
        .probe_walk_limit = 0,
        .data = .{ .io = file.io, .dir = file.dir },
        .probe_field = shared.probe_field,
        .device = device,
        .probe_fit_key = null,
        .probe_light_key = null,
        .reference_key = null,
        .approximation_key = null,
    };
}

pub fn deinit(mode: *CloudBench, gpa: std.mem.Allocator) void {
    mode.storeProbes(gpa) catch |err| log.warn("probes not stored: {t}", .{err});
    if (mode.accumulations) |accumulations| accumulations.deinit();
    if (mode.probes) |probes| probes.deinit();
    mode.layout_depths.deinit();
    mode.sun_ranges.deinit();
    mode.photon_counter.deinit();
    mode.blue_noise.deinit();
    mode.sun_reach.deinit();
    mode.sun_least.deinit();
    mode.cloud_shadow_placeholder.deinit();
    mode.atmosphere.deinit();
    mode.sun.deinit();
    mode.sky.deinit();
    mode.majorant_4.deinit();
    mode.majorant_2.deinit();
    mode.majorant.deinit();
    mode.density.deinit();
    mode.sun_reach_bake.mode_kernel.deinit();
    mode.sun_least_bake.mode_kernel.deinit();
    mode.probe_fill.mode_kernel.deinit();
    mode.probe_lighting.mode_kernel.deinit();
    mode.probe_fit.mode_kernel.deinit();
    mode.probe_range.mode_kernel.deinit();
    mode.sun_range_reset.mode_kernel.deinit();
    mode.probe_bake.mode_kernel.deinit();
    mode.sky_transfer_bake.mode_kernel.deinit();
    mode.photon_counter_reset.mode_kernel.deinit();
    mode.photon_trace.mode_kernel.deinit();
    mode.photon_clear.mode_kernel.deinit();
    mode.sun_bake.mode_kernel.deinit();
    mode.sky_view_bake.mode_kernel.deinit();
    mode.sky_bake.mode_kernel.deinit();
    mode.approximation.mode_kernel.deinit();
    mode.reference.mode_kernel.deinit();
    mode.kernel.deinit();
    mode.* = undefined;
}

pub fn record(
    mode: *CloudBench,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    var bench_frame = frame.*;
    bench_frame.cloud_bench.origin = mode.origin;
    bench_frame.cloud_bench.grid_min = mode.grid_min;
    bench_frame.cloud_bench.grid_size = mode.grid_size;
    const size = frame.uniforms.frame[1..3].*;
    if (mode.accumulations == null or !std.mem.eql(u32, &mode.accumulations.?.size, &size)) {
        if (mode.accumulations) |previous| {
            // Frames in flight may still write them.
            try shared.device.waitIdle();
            previous.deinit();
            mode.accumulations = null;
        }
        mode.accumulations = try initAccumulations(shared.device, size);
    }
    const accumulations = &mode.accumulations.?;

    if (mode.atmosphere_stale) {
        try mode.atmosphere.bake(command_list, shared, &frame.scene);
        mode.atmosphere_stale = false;
        mode.light_key = null;
    }
    const light_key: LightKey = .{ .sun_direction = frame.uniforms.sun_direction, .media = frame.scene.media };
    if (mode.light_key == null or !std.meta.eql(mode.light_key.?, light_key)) {
        const sky_groups = mode.sky_bake.mode_kernel.kernel.groupCount(.{ sky_size[0], sky_size[1], 1 });
        try mode.dispatch(mode.sky_bake.mode_kernel, mode.sky_bake.params, command_list, shared, &bench_frame, sky_groups);
        try mode.dispatch(mode.sun_bake.mode_kernel, mode.sun_bake.params, command_list, shared, &bench_frame, .{ 1, 1, 1 });
        mode.light_key = light_key;
        mode.probe_light_key = null;
    }
    const extinction = frame.cloud_bench.medium[0];
    const view: scene.CloudBench.View = @fromBackingInt(@intCast(frame.cloud_bench.options[0]));
    const flags: uniforms.CloudBenchFlags = @bitCast(frame.cloud_bench.options[2]);
    const bounce_limit = frame.cloud_bench.options[1];
    const probes_on = bounce_limit > 1 and view != .reference;
    // The approximation binds the probes even where it does not read them.
    if (mode.probes == null or probes_on) {
        const layout_key: LayoutKey = if (flags.adaptive_probes)
            .{ .adaptive = true, .budget = frame.cloud_bench.probe_cells[3], .depth_ratio = frame.cloud_bench.probe_origin[3] }
        else
            .{ .adaptive = false, .budget = 0, .depth_ratio = 0 };
        try mode.requestLayout(shared.device, layout_key);
    }
    const probes = &mode.probes.?;
    bench_frame.cloud_bench.probe_origin = .{ probes.origin[0], probes.origin[1], probes.origin[2], 0 };
    bench_frame.cloud_bench.probe_spacing = .{ probes.spacing[0], probes.spacing[1], probes.spacing[2], 0 };
    bench_frame.cloud_bench.probe_cells = .{ probes.cells[0], probes.cells[1], probes.cells[2], 0 };
    const rounds_per_frame = frame.cloud_bench.probe_counts[2];
    bench_frame.cloud_bench.probe_counts = .{ probes.count, probes.leaf_count, rounds_per_frame, 0 };
    mode.probe_walk_limit = if (probes_on) frame.cloud_bench.options[3] else 0;
    if (probes_on) {
        const probe_key: ProbeCache.Key = .{
            .dense_resolution = if (probes.key.adaptive) 0 else dense_resolution,
            .probe_budget = probes.key.budget,
            .depth_ratio = probes.key.depth_ratio,
            .grid_size = mode.grid_size,
            .grid_min = mode.grid_min,
            .extinction = frame.cloud_bench.medium[0],
            .bounce_limit = bounce_limit,
            .options = .{ .ratio_tracking = flags.ratio_tracking, .photons = flags.photons, .leaf_lengths = flags.photons and flags.leaf_lengths },
        };
        const walk_limit = mode.probe_walk_limit;
        const changed = mode.probe_key == null or !std.meta.eql(mode.probe_key.?, probe_key);
        // The walks of the frame that reached the limit have run by the next one.
        if (changed or mode.probe_walks >= walk_limit)
            mode.storeProbes(std.heap.page_allocator) catch |err| log.warn("probes not stored: {t}", .{err});
        if (changed) {
            mode.probe_key = probe_key;
            mode.probe_walks = mode.loadProbes(probe_key, probes);
            mode.probe_cached_walks = mode.probe_walks;
        }
        const probe_groups = mode.probe_bake.mode_kernel.kernel.groupCount(.{ probes.count, 1, 1 });
        if (mode.probe_walks < walk_limit and mode.sun_reach_extinction != extinction) {
            const cells: [3]u32 = .{ mode.sun_reach.desc.width, mode.sun_reach.desc.height, mode.sun_reach.desc.depth_or_layers };
            try mode.dispatch(mode.sun_least_bake.mode_kernel, mode.sun_least_bake.params, command_list, shared, &bench_frame, mode.sun_least_bake.mode_kernel.kernel.groupCount(cells));
            try mode.dispatch(mode.sun_reach_bake.mode_kernel, mode.sun_reach_bake.params, command_list, shared, &bench_frame, mode.sun_reach_bake.mode_kernel.kernel.groupCount(cells));
            mode.sun_reach_extinction = extinction;
        }
        const frame_walks = @min(rounds_per_frame, walk_limit -| mode.probe_walks);
        if (!flags.photons) {
            for (0..frame_walks) |_| {
                bench_frame.cloud_bench.samples = .{ 0, 0, mode.probe_walks, 0 };
                try mode.dispatch(mode.probe_bake.mode_kernel, mode.probe_bake.params, command_list, shared, &bench_frame, probe_groups);
                mode.probe_walks += 1;
            }
        } else if (frame_walks > 0) {
            // One dispatch traces the frame's rounds together.
            bench_frame.cloud_bench.samples = .{ 0, 0, mode.probe_walks, frame_walks };
            if (mode.probe_walks == 0) try mode.dispatch(mode.photon_clear.mode_kernel, mode.photon_clear.params, command_list, shared, &bench_frame, probe_groups);
            try mode.dispatch(mode.photon_counter_reset.mode_kernel, mode.photon_counter_reset.params, command_list, shared, &bench_frame, .{ 1, 1, 1 });
            try mode.dispatch(mode.photon_trace.mode_kernel, mode.photon_trace.params, command_list, shared, &bench_frame, probe_groups);
            try mode.dispatch(mode.sky_transfer_bake.mode_kernel, mode.sky_transfer_bake.params, command_list, shared, &bench_frame, probe_groups);
            mode.probe_walks += frame_walks;
        }
        bench_frame.cloud_bench.samples = .{ 0, 0, mode.probe_walks, 0 };
        var probe_fit_key: ProbeFitKey = .{ .probe = probe_key, .walks = mode.probe_walks, .sun_order3 = flags.sun_order3, .sun_linear = flags.sun_linear };
        if (mode.probe_fit_key) |fitted| {
            const lagging = mode.probe_walks < walk_limit and mode.probe_walks > fitted.walks and
                mode.probe_walks < fitted.walks + refit_frames * rounds_per_frame;
            if (lagging and std.meta.eql(fitted.probe, probe_key) and fitted.sun_order3 == flags.sun_order3 and fitted.sun_linear == flags.sun_linear)
                probe_fit_key.walks = fitted.walks;
        }
        if (mode.probe_fit_key == null or !std.meta.eql(mode.probe_fit_key.?, probe_fit_key)) {
            try mode.dispatch(mode.sun_range_reset.mode_kernel, mode.sun_range_reset.params, command_list, shared, &bench_frame, .{ 1, 1, 1 });
            try mode.dispatch(mode.probe_range.mode_kernel, mode.probe_range.params, command_list, shared, &bench_frame, probe_groups);
            try mode.dispatch(mode.probe_fit.mode_kernel, mode.probe_fit.params, command_list, shared, &bench_frame, probe_groups);
            mode.probe_fit_key = probe_fit_key;
        }
        const probe_light_key: ProbeLightKey = .{ .probe = probe_key, .walks = probe_fit_key.walks, .light = light_key, .flags = frame.cloud_bench.options[2] };
        if (mode.probe_light_key == null or !std.meta.eql(mode.probe_light_key.?, probe_light_key)) {
            try mode.dispatch(mode.probe_lighting.mode_kernel, mode.probe_lighting.params, command_list, shared, &bench_frame, probe_groups);
            const leaf_groups = mode.probe_fill.mode_kernel.kernel.groupCount(.{ probes.leaf_count, 1, 1 });
            try mode.dispatch(mode.probe_fill.mode_kernel, mode.probe_fill.params, command_list, shared, &bench_frame, leaf_groups);
            mode.probe_light_key = probe_light_key;
        }
    }

    const key: Key = .init(&bench_frame);
    if (mode.reference_key == null or !std.meta.eql(mode.reference_key.?, key)) {
        const view_groups = mode.sky_view_bake.mode_kernel.kernel.groupCount(.{ accumulations.sky_view.desc.width, accumulations.sky_view.desc.height, 1 });
        try mode.dispatch(mode.sky_view_bake.mode_kernel, mode.sky_view_bake.params, command_list, shared, &bench_frame, view_groups);
        accumulations.reference_samples = 0;
        mode.reference_key = key;
    }
    const approximation_key: @typeInfo(@TypeOf(mode.approximation_key)).optional.child = .{
        .key = key,
        .scene = frame.scene,
        .probes = if (probes_on) .{ .flags = frame.cloud_bench.options[2], .layout = probes.key, .doublings = std.math.log2_int(u32, @max(mode.probe_walks, 1)) } else null,
    };
    if (mode.approximation_key == null or !std.meta.eql(mode.approximation_key.?, approximation_key)) {
        accumulations.approximation_samples = 0;
        mode.approximation_key = approximation_key;
    }

    const pixel_groups = mode.kernel.kernel.groupCount(.{ size[0], size[1], 1 });
    if (view != .approximation and view != .coverage) {
        bench_frame.cloud_bench.samples = .{ accumulations.reference_samples, accumulations.approximation_samples, 0, 0 };
        try mode.dispatch(mode.reference.mode_kernel, mode.reference.params, command_list, shared, &bench_frame, pixel_groups);
        accumulations.reference_samples += 1;
    }
    if (view != .reference) {
        bench_frame.cloud_bench.samples = .{ accumulations.reference_samples, accumulations.approximation_samples, 0, 0 };
        try mode.dispatch(mode.approximation.mode_kernel, mode.approximation.params, command_list, shared, &bench_frame, pixel_groups);
        accumulations.approximation_samples += 1;
    }
    bench_frame.cloud_bench.samples = .{ accumulations.reference_samples, accumulations.approximation_samples, 0, 0 };
    try mode.dispatch(mode.kernel, mode.resolve_params, command_list, shared, &bench_frame, pixel_groups);
}

/// The accumulations restart on their own keys.
pub fn invalidate(mode: *CloudBench, scene_uniforms: *const uniforms.SceneUniforms) void {
    _ = scene_uniforms;
    mode.atmosphere_stale = true;
}

/// The walks of the probes' field for `key`, uploaded from the cache or the field's
/// file, or zero.
fn loadProbes(mode: *CloudBench, key: ProbeCache.Key, probes: *const Probes) u32 {
    var buffer: ProbeCache.PathBuffer = undefined;
    const io, const dir, const path = switch (mode.probe_field) {
        .cache => .{ mode.data.io, mode.data.dir, ProbeCache.cachePath(&buffer, key) },
        .file => |file| .{ (file orelse return 0).io, file.?.dir, file.?.path },
    };
    const loaded = ProbeCache.load(std.heap.page_allocator, io, dir, path, mode.device, key, probes.transfer, probes.sun_bins) catch |err| {
        log.err("probes {s} not loaded: {t}", .{ path, err });
        return 0;
    };
    if (mode.probe_field == .file) log.info("probes: {d} walks from {s}", .{ loaded orelse 0, path });
    return loaded orelse 0;
}

/// Stores the probes' walks in the cache, once the GPU is done with them, if they are
/// at least ProbeCache.walks_min and it holds fewer.
fn storeProbes(mode: *CloudBench, gpa: std.mem.Allocator) !void {
    if (mode.probe_field != .cache) return;
    const key = mode.probe_key orelse return;
    const probes = mode.probes orelse return;
    if (mode.probe_walks <= mode.probe_cached_walks or mode.probe_walks < ProbeCache.walks_min) return;
    try mode.device.waitIdle();
    var buffer: ProbeCache.PathBuffer = undefined;
    try ProbeCache.save(gpa, mode.data.io, mode.data.dir, ProbeCache.cachePath(&buffer, key), mode.device, key, mode.probe_walks, probes.transfer, probes.sun_bins);
    mode.probe_cached_walks = mode.probe_walks;
}

/// Writes the probes' field to `path`, once the GPU is done with it, for validation.
pub fn saveProbes(mode: *CloudBench, gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !void {
    const key = mode.probe_key orelse return error.NoProbes;
    const probes = mode.probes orelse return error.NoProbes;
    try mode.device.waitIdle();
    try ProbeCache.save(gpa, io, dir, path, mode.device, key, mode.probe_walks, probes.transfer, probes.sun_bins);
}

/// Lays the probes out for `key`: at once if none are, else once it has been asked
/// for `layout_settle_frames` frames running. The old layout's walks are stored first.
fn requestLayout(mode: *CloudBench, device: gpu.Device, key: LayoutKey) gpu.Error!void {
    if (mode.probes) |probes| {
        if (std.meta.eql(probes.key, key)) {
            mode.layout_request = null;
            return;
        }
        if (mode.layout_request == null or !std.meta.eql(mode.layout_request.?.key, key)) {
            mode.layout_request = .{ .key = key, .frames = 0 };
            return;
        }
        mode.layout_request.?.frames += 1;
        if (mode.layout_request.?.frames < layout_settle_frames) return;
    }
    mode.storeProbes(std.heap.page_allocator) catch |err| log.warn("probes not stored: {t}", .{err});
    const gpa = std.heap.page_allocator;
    const start = std.Io.Timestamp.now(mode.data.io, .awake);
    var layout = if (key.adaptive)
        ProbeLayout.adaptive(gpa, mode.data.io, mode.layout_depths.grid, mode.layout_depths.source(), key.budget, key.depth_ratio) catch |err| switch (err) {
            // The renderer's thread is never canceled.
            error.Canceled => unreachable,
            else => |other| return other,
        }
    else
        try ProbeLayout.dense(gpa, mode.layout_depths.grid, dense_resolution);
    defer layout.deinit(gpa);
    log.info("probe layout: {d} probes in {d} leaves, finest spacing {d:.2}, built in {d} ms", .{
        layout.positions.len, layout.corners.len, layout.spacing[0], start.untilNow(mode.data.io, .awake).toMilliseconds(),
    });
    // Frames in flight may still read the old probes.
    try device.waitIdle();
    if (mode.probes) |probes| probes.deinit();
    mode.probes = null;
    mode.probes = try .init(device, key, &layout);
    mode.layout_request = null;
    mode.probe_key = null;
    mode.probe_fit_key = null;
    mode.probe_light_key = null;
}

/// Deletes what ProbeCache holds for the present settings and bakes the probes anew.
pub fn resetProbes(mode: *CloudBench) void {
    if (mode.probe_field != .cache) return;
    const key = mode.probe_key orelse return;
    var buffer: ProbeCache.PathBuffer = undefined;
    ProbeCache.delete(mode.data.io, mode.data.dir, ProbeCache.cachePath(&buffer, key)) catch |err| log.warn("probes not deleted: {t}", .{err});
    mode.probe_walks = 0;
    mode.probe_cached_walks = 0;
}

/// How far the probes' bake has come, from 0 to 1, or null while it is not running.
pub fn bakeProgress(mode: *const CloudBench) ?f32 {
    if (mode.probe_walks >= mode.probe_walk_limit) return null;
    return @as(f32, @floatFromInt(mode.probe_walks)) / @as(f32, @floatFromInt(mode.probe_walk_limit));
}

/// Samples in the reference and the approximation, and walks per probe, for display.
pub fn sampleCounts(mode: *const CloudBench) [3]u32 {
    const accumulations = mode.accumulations orelse return .{ 0, 0, mode.probe_walks };
    return .{ accumulations.reference_samples, accumulations.approximation_samples, mode.probe_walks };
}

fn dispatch(
    mode: *const CloudBench,
    mode_kernel: ModeKernel,
    params: Params,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
    groups: [3]u32,
) gpu.Error!void {
    const accumulations = &mode.accumulations.?;
    var buffer: [std.enums.values(Global).len]gpu.CommandList.Binding = undefined;
    var bindings: std.ArrayList(gpu.CommandList.Binding) = .initBuffer(&buffer);
    inline for (comptime std.enums.values(Global)) |global| {
        if (params.get(global)) |param| {
            const resource: gpu.CommandList.Binding.Resource = switch (global) {
                .bench => .{ .data = std.mem.asBytes(&frame.cloud_bench) },
                .bench_density => .{ .texture = mode.density },
                .bench_majorant => .{ .texture = mode.majorant },
                .bench_majorant_2 => .{ .texture = mode.majorant_2 },
                .bench_majorant_4 => .{ .texture = mode.majorant_4 },
                .bench_reference => .{ .texture = accumulations.reference },
                .bench_approximation => .{ .texture = accumulations.approximation },
                .bench_sky, .bench_sky_target => .{ .texture = mode.sky },
                .bench_sky_view, .bench_sky_view_target => .{ .texture = accumulations.sky_view },
                .bench_sun, .bench_sun_target => .{ .texture = mode.sun },
                .bench_probe_positions => .{ .buffer = mode.probes.?.positions },
                .bench_probe_cells => .{ .buffer = mode.probes.?.cell_leaves },
                .bench_probe_corners => .{ .buffer = mode.probes.?.corners },
                .bench_probe_transfer => .{ .buffer = mode.probes.?.transfer },
                .bench_probe_sun_bins, .bench_probe_sun_bins_raw => .{ .buffer = mode.probes.?.sun_bins },
                .bench_photon_counter => .{ .buffer = mode.photon_counter },
                .bench_probe_volumes => .{ .buffer = mode.probes.?.volumes },
                .bench_probe_sun => .{ .buffer = mode.probes.?.sun },
                .bench_probe_sun_ranges => .{ .buffer = mode.sun_ranges },
                .bench_probe_sky => .{ .buffer = mode.probes.?.sky },
                .bench_probe_light_values => .{ .buffer = mode.probes.?.light },
                .bench_probe_light, .bench_probe_light_target => .{ .texture = mode.probes.?.atlas },
                .bench_sun_least => .{ .texture = mode.sun_least },
                .bench_sun_reach => .{ .texture = mode.sun_reach },
                .blue_noise => .{ .texture = mode.blue_noise },
                .multiple_scattering_lut => .{ .texture = mode.atmosphere.texture },
                .cloud_shadow_map => .{ .texture = mode.cloud_shadow_placeholder },
            };
            bindings.appendAssumeCapacity(.{ .param = param, .resource = resource });
        }
    }
    try mode_kernel.dispatchGroups(command_list, shared, frame, bindings.items, groups);
}

fn findParams(kernel: gpu.Kernel) gpu.Kernel.LayoutError!Params {
    try kernel.checkLayout(uniforms.CloudBenchUniforms, "bench");
    var params: Params = undefined;
    inline for (comptime std.enums.values(Global)) |global| params.set(global, try kernel.paramOptional(@tagName(global)));
    return params;
}

/// A single-channel half-float 3D texture of `values`, x fastest, for `usage`.
pub fn uploadVolume(device: gpu.Device, size: [3]u32, values: []const f16, usage: gpu.Texture.Usage, label: [*:0]const u8) gpu.Error!gpu.Texture {
    assert(values.len == @as(u64, size[0]) * size[1] * size[2]);
    var texture_usage = usage;
    texture_usage.copy_destination = true;
    const texture: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = .r16_float,
        .width = size[0],
        .height = size[1],
        .depth_or_layers = size[2],
        .usage = texture_usage,
        .label = label,
    });
    errdefer texture.deinit();
    const row_pitch = @as(u64, size[0]) * @sizeOf(f16);
    const slice_size = row_pitch * size[1];
    // Slabs bound the staging memory of one upload.
    const slab_slices = @max(1, upload_slab_bytes / slice_size);
    const bytes = std.mem.sliceAsBytes(values);
    var z: u32 = 0;
    while (z < size[2]) {
        const depth: u32 = @intCast(@min(slab_slices, size[2] - z));
        const start = slice_size * z;
        try texture.upload(device, .{ .x = 0, .y = 0, .z = z, .width = size[0], .height = size[1], .depth = depth }, bytes[start..][0 .. slice_size * depth], row_pitch);
        z += depth;
    }
    return texture;
}

const upload_slab_bytes = 64 << 20;

fn uploadCoarserMajorants(
    gpa: std.mem.Allocator,
    device: gpu.Device,
    cells: [3]u32,
    majorant: []const f16,
    factor: u32,
    label: [*:0]const u8,
) (gpu.Error || std.mem.Allocator.Error)!gpu.Texture {
    var coarse_cells: [3]u32 = undefined;
    for (&coarse_cells, cells) |*coarse, fine| coarse.* = std.math.divCeil(u32, fine, factor) catch unreachable;
    const coarse = try gpa.alloc(f16, @as(usize, coarse_cells[0]) * coarse_cells[1] * coarse_cells[2]);
    defer gpa.free(coarse);
    coarserMajorants(cells, majorant, factor, coarse_cells, coarse);
    return uploadVolume(device, coarse_cells, coarse, .{ .shader_resource = true }, label);
}

/// Each of `coarse`, a cell `factor` times as large per axis, bounds the cells of
/// `majorant` it covers.
fn coarserMajorants(cells: [3]u32, majorant: []const f16, factor: u32, coarse_cells: [3]u32, coarse: []f16) void {
    @memset(coarse, 0);
    for (0..cells[2]) |z| for (0..cells[1]) |y| for (0..cells[0]) |x| {
        const bound = &coarse[x / factor + coarse_cells[0] * (y / factor + coarse_cells[1] * (z / factor))];
        bound.* = @max(bound.*, majorant[x + cells[0] * (y + cells[1] * z)]);
    };
}

test "coarserMajorants bounds the cells each covers, past the grid's end too" {
    var coarse: [2]f16 = undefined;
    coarserMajorants(.{ 3, 1, 1 }, &.{ 0.25, 0.5, 0.75 }, 2, .{ 2, 1, 1 }, &coarse);
    try std.testing.expectEqualSlices(f16, &.{ 0.5, 0.75 }, &coarse);
}

fn initAccumulations(device: gpu.Device, size: [2]u32) gpu.Error!Accumulations {
    const reference = try initAccumulation(device, size, "cloud bench reference");
    errdefer reference.deinit();
    const approximation = try initAccumulation(device, size, "cloud bench approximation");
    errdefer approximation.deinit();
    const sky_view: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba16_float,
        .width = std.math.divCeil(u32, size[0], sky_view_stride) catch unreachable,
        .height = std.math.divCeil(u32, size[1], sky_view_stride) catch unreachable,
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = "cloud bench sky view",
    });
    return .{ .size = size, .reference = reference, .approximation = approximation, .sky_view = sky_view };
}

fn initAccumulation(device: gpu.Device, size: [2]u32, label: [*:0]const u8) gpu.Error!gpu.Texture {
    return .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = size[0],
        .height = size[1],
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true },
        .label = label,
    });
}
