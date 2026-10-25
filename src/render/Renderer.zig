//! Progressive rendering: each `render` adds one sample per pixel from the active mode
//! to an rgba32f accumulation texture, then runs the display pass into an rgba8 image.
//! Camera, sun, FOV, aspect, resolution, mode, preset, and texture quality changes
//! restart the accumulation; the display settings (`Post.Tone`) never do.
//!
//! # Adding a render mode
//!
//! A mode is a struct in `src/render/` registered in `ModeKind` and `Mode` below. The
//! comptime check in `Mode` enforces this interface:
//!
//!     kernel: ModeKernel,               // the per-pixel kernel
//!     pub fn init(gpa: Allocator, shared: *const Shared) !Self;
//!     pub fn deinit(mode: *Self, gpa: Allocator) void;
//!     pub fn record(mode: *Self, command_list: gpu.CommandList, shared: *const Shared,
//!         frame: *const Frame) gpu.Error!void;
//!     pub fn invalidate(mode: *Self, scene: *const SceneUniforms) void;
//!
//! - `init` compiles the mode's kernels and creates its resources; `shared.device`
//!   is the device. The mode owns everything it creates and frees it in `deinit`.
//! - `record` adds exactly one sample to every pixel of `frame.accum`. It only records
//!   dispatches into `command_list` (the renderer submits once per frame), and it never
//!   reads GPU data back, but for the cloud bench's ProbeCache, which stores a finished
//!   bake once. Lookup-table bakes are recorded here too, before the sample.
//! - `invalidate` reports a new scene preset. A mode that bakes from `SceneUniforms`
//!   marks its bake stale and rebakes in the next `record`; others ignore it. `init`
//!   is followed by `invalidate` with the current preset.
//!
//! The per-pixel kernel imports `shared.bindings` (shaders/shared/bindings.slang),
//! which declares the globals every mode kernel shares: `frame`
//! (`ConstantBuffer<FrameUniforms>`), `scene` (`ConstantBuffer<SceneUniforms>`),
//! `accum` (`RWTexture2D<float4>`, rgba32f), the LUT buffers `cie`,
//! `srgb_to_spectrum_basis`, `ozone_cross_section`, and `ozone_profile`
//! (`StructuredBuffer<float>`),
//! and the samplers `linear_clamp` and `linear_repeat_u`; a kernel that imports
//! `shared.earth_bindings` (directly or through earth/surface.slang) also gets the
//! Earth maps of `Shared.earth_maps`, which the renderer uploads only once such a
//! kernel runs a preset that samples them. The kernel runs
//! `[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]` (8 x 8), one thread per pixel,
//! and goes through `PixelSample`:
//!
//!     PixelSample s = PixelSample(thread.xy);
//!     if (!s.valid) return;
//!     float3 dir = s.camera_ray_direction();  // draws RNG dimension 0
//!     ...                                     // further draws from s.rng
//!     s.accumulate(linear_rec709);
//!
//! On the host, `ModeKernel` compiles such a kernel, verifies the group size and the
//! uniform layouts, and binds the shared globals by name on every dispatch; the mode
//! adds only its own parameters:
//!
//!     kernel: ModeKernel,
//!     ... = try ModeKernel.init(shared.device, "modes/pathtrace", "pathtrace_main", &.{});
//!     try mode.kernel.dispatch(command_list, shared, frame, &.{
//!         .{ .param = mode.weights_param, .resource = .{ .buffer = mode.weights } },
//!     });
//!
//! Auxiliary kernels (LUT bakes) are plain `gpu.Kernel`s dispatched on the same
//! command list; consecutive dispatches are separated by a UAV barrier.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const Luts = @import("../assets/Luts.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const uniforms = @import("uniforms.zig");
const scene = @import("scene.zig");
const DebugMode = @import("DebugMode.zig");
const PathTracer = @import("PathTracer.zig");
const RayMarcher = @import("RayMarcher.zig");
const NeuralMode = @import("NeuralMode.zig");
const BrunetonMode = @import("BrunetonMode.zig");
const CloudBench = @import("CloudBench.zig");
const GpuEarthMaps = @import("GpuEarthMaps.zig");
const TilePool = @import("TilePool.zig");
const TileStreamer = @import("TileStreamer.zig");
const GpuLuts = @import("GpuLuts.zig");
const ModeKernel = @import("ModeKernel.zig");
const Post = @import("Post.zig");

const Renderer = @This();

gpa: std.mem.Allocator,
shared: Shared,
post: Post,
mode: Mode,
preset: scene.Preset,
/// Renders `scene.singleScattering` of the preset.
single_scattering: bool,
/// Off renders `scene.withoutClouds` of the preset.
clouds: bool,
water: scene.Water,
cloud_scattering: scene.CloudScattering,
/// Set directly: the `cloud_bench` mode restarts its own accumulations as its
/// settings require.
cloud_bench: scene.CloudBench,
/// On renders `scene.cloudsOnly` of the preset.
clouds_only: bool,
/// Off renders `scene.marchedCloudShadows` of the preset.
cloud_shadow_map: bool,
/// Renders `scene.withBounceLimit` of the preset.
bounce_limit: u32,
/// On renders `scene.withTiles` of the preset; only with `tile_files`.
tiles: bool,
/// Must outlive the renderer, which streams from them.
tile_files: ?*const TilePool.Files,
/// Null without tile files, or once streaming failed.
tile_streamer: ?TileStreamer,
/// Holds the resident tiles as they are, so that accumulation never restarts.
tiles_frozen: bool,
/// The Earth map set asked for, uploaded to `shared.earth_maps` once the active mode
/// samples the maps under the active preset.
texture_quality: EarthMaps.Quality,
earth_map_cache: ?GpuEarthMaps.Cache,
accum: gpu.Texture,
/// The display output, rgba8 with row 0 at the top; what `en_present` blits.
image: gpu.Texture,
/// The uniforms of the last frame, except that `frame[0]` already counts it: it is the
/// number of samples in `accum` and the index of the next frame.
frame_uniforms: uniforms.FrameUniforms,

/// Resources every mode reads and none owns.
pub const Shared = struct {
    device: gpu.Device,
    luts: GpuLuts,
    /// For the `neural` mode, which reads it when it starts and on `setNetwork`.
    /// The renderer owns the path.
    network: ?InputFile,
    /// For the `bruneton` mode, which reads it when it starts. The renderer owns the path.
    lut4d: ?InputFile,
    /// For the `cloud_bench` mode, which reads it when it starts. The renderer owns the path.
    cloud_volume: ?InputFile,
    /// The `cloud_bench` mode's probes. The renderer owns the path.
    probe_field: ProbeField,
    /// Placeholders (`quality == null`) until a mode samples the maps under a preset
    /// that needs them; an upload then stays bound across mode and preset changes.
    earth_maps: GpuEarthMaps,
    /// The base levels of the tile files, or a placeholder without them.
    tiles: TilePool,
};

/// What one `record` call renders.
pub const Frame = struct {
    /// `frame[0]` is the index since the last reset, `frame[1..3]` the image size,
    /// `frame[3]` the seed.
    uniforms: uniforms.FrameUniforms,
    scene: uniforms.SceneUniforms,
    /// Read by the `cloud_bench` mode only.
    cloud_bench: uniforms.CloudBenchUniforms,
    /// Sums of the samples so far; rows count from the bottom of the image.
    accum: gpu.Texture,
};

/// A mode's input file, read only when the mode starts, so that a missing or broken
/// file fails that mode alone.
pub const InputFile = struct {
    io: std.Io,
    dir: std.Io.Dir,
    path: []const u8,
};

/// Where the `cloud_bench` mode's probes come from: the probe cache, which resumes and
/// keeps their bakes, or for validation (TESTING.md) a field read once from a file, or
/// none, which the cache is never read for nor written by.
pub const ProbeField = union(enum) {
    cache,
    file: ?InputFile,
};

pub const ModeKind = enum { debug, path_tracer, raymarch, neural, bruneton, cloud_bench };

pub const Mode = union(ModeKind) {
    debug: DebugMode,
    path_tracer: PathTracer,
    raymarch: RayMarcher,
    neural: NeuralMode,
    bruneton: BrunetonMode,
    cloud_bench: CloudBench,

    comptime {
        for (@typeInfo(Mode).@"union".field_types) |T| {
            assert(@FieldType(T, "kernel") == ModeKernel);
            assert(@TypeOf(T.deinit) == fn (*T, std.mem.Allocator) void);
            assert(@TypeOf(T.record) == fn (*T, gpu.CommandList, *const Shared, *const Frame) gpu.Error!void);
            assert(@TypeOf(T.invalidate) == fn (*T, *const uniforms.SceneUniforms) void);
        }
    }

    fn init(kind: ModeKind, gpa: std.mem.Allocator, shared: *const Shared, preset: scene.Preset) !Mode {
        switch (kind) {
            inline else => |tag| {
                const T = @FieldType(Mode, @tagName(tag));
                var mode = try T.init(gpa, shared);
                mode.invalidate(&preset.sceneUniforms());
                return @unionInit(Mode, @tagName(tag), mode);
            },
        }
    }

    fn deinit(mode: *Mode, gpa: std.mem.Allocator) void {
        switch (mode.*) {
            inline else => |*active| active.deinit(gpa),
        }
    }

    /// Whether the per-pixel kernel declares the Earth maps; the shader is the only
    /// place that knows.
    fn samplesEarthMaps(mode: *const Mode) bool {
        return switch (mode.*) {
            inline else => |*active| active.kernel.earth_params != null,
        };
    }
};

pub const View = @import("View.zig");

pub const Options = struct {
    size: [2]u32,
    mode: ModeKind,
    preset: scene.Preset,
    single_scattering: bool = false,
    clouds: bool = true,
    water: scene.Water = .{},
    cloud_scattering: scene.CloudScattering = .{},
    cloud_bench: scene.CloudBench = .{},
    clouds_only: bool = false,
    cloud_shadow_map: bool = true,
    bounce_limit: u32 = scene.atmosphere.bounce_count_max,
    /// Keys the random numbers of frame i by i + seed, so renders seeded `n` samples
    /// apart are independent and together equal one longer render.
    seed: u32 = 0,
    texture_quality: EarthMaps.Quality,
    /// Without it, presets that need the Earth maps are refused. Must outlive the
    /// renderer, which reloads from it when the texture quality changes.
    earth_map_cache: ?GpuEarthMaps.Cache = null,
    /// For the `neural` mode, which fails to start without it; the path is copied.
    network: ?InputFile = null,
    /// For the `bruneton` mode, which fails to start without it; the path is copied.
    lut4d: ?InputFile = null,
    /// For the `cloud_bench` mode, which fails to start without it; the path is copied.
    cloud_volume: ?InputFile = null,
    /// The path is copied.
    probe_field: ProbeField = .cache,
    /// With them, scenes with the Earth maps take their surface from the tiles while
    /// `tiles` is on.
    tile_files: ?*const TilePool.Files = null,
    tiles: bool = true,
};

pub const InitError = GpuEarthMaps.LoadError || TilePool.LoadError || gpu.Kernel.LayoutError ||
    NeuralMode.InitError || BrunetonMode.InitError || CloudBench.InitError || error{EarthMapsMissing};

/// Compiles the display and mode kernels, uploads `luts`, which the caller may free
/// afterwards, and uploads the Earth maps of `options.texture_quality` if the mode
/// samples them under the preset.
pub fn init(gpa: std.mem.Allocator, device: gpu.Device, luts: *const Luts, options: Options) InitError!Renderer {
    try checkPreset(options.earth_map_cache, options.preset);
    var gpu_luts: GpuLuts = try .init(gpa, device, luts);
    errdefer gpu_luts.deinit(gpa);
    const network = try copyInputFile(gpa, options.network);
    errdefer freeInputFile(gpa, network);
    const lut4d = try copyInputFile(gpa, options.lut4d);
    errdefer freeInputFile(gpa, lut4d);
    const cloud_volume = try copyInputFile(gpa, options.cloud_volume);
    errdefer freeInputFile(gpa, cloud_volume);
    const probe_field: ProbeField = switch (options.probe_field) {
        .cache => .cache,
        .file => |file| .{ .file = try copyInputFile(gpa, file) },
    };
    errdefer if (probe_field == .file) freeInputFile(gpa, probe_field.file);
    var shared: Shared = .{
        .device = device,
        .luts = gpu_luts,
        .network = network,
        .lut4d = lut4d,
        .cloud_volume = cloud_volume,
        .probe_field = probe_field,
        .earth_maps = undefined,
        .tiles = undefined,
    };
    shared.earth_maps = try .placeholder(device);
    errdefer shared.earth_maps.deinit();
    shared.tiles = if (options.tile_files) |files| try .load(gpa, device, files) else try .placeholder(gpa, device);
    errdefer shared.tiles.deinit(gpa);
    var tile_streamer: ?TileStreamer = if (options.tile_files != null) try .init(gpa, device, &shared.tiles) else null;
    errdefer if (tile_streamer) |*streamer| streamer.deinit(gpa);
    const post: Post = try .init(device);
    errdefer post.deinit();
    var mode: Mode = try .init(options.mode, gpa, &shared, options.preset);
    errdefer mode.deinit(gpa);
    _ = try prepareEarthMaps(gpa, &shared, options.earth_map_cache, &mode, options.preset, options.texture_quality);
    const accum = try initAccum(device, options.size);
    errdefer accum.deinit();
    const image = try initImage(device, options.size);
    return .{
        .gpa = gpa,
        .shared = shared,
        .post = post,
        .mode = mode,
        .preset = options.preset,
        .single_scattering = options.single_scattering,
        .clouds = options.clouds,
        .water = options.water,
        .cloud_scattering = options.cloud_scattering,
        .cloud_bench = options.cloud_bench,
        .clouds_only = options.clouds_only,
        .cloud_shadow_map = options.cloud_shadow_map,
        .bounce_limit = options.bounce_limit,
        .tiles = options.tiles and options.tile_files != null,
        .tile_files = options.tile_files,
        .tile_streamer = tile_streamer,
        .tiles_frozen = false,
        .texture_quality = options.texture_quality,
        .earth_map_cache = options.earth_map_cache,
        .accum = accum,
        .image = image,
        .frame_uniforms = resetUniforms(options.size, options.seed),
    };
}

/// Waits for the GPU, then releases everything the renderer created.
pub fn deinit(renderer: *Renderer) void {
    renderer.shared.device.waitIdle() catch {};
    renderer.image.deinit();
    renderer.accum.deinit();
    renderer.mode.deinit(renderer.gpa);
    renderer.post.deinit();
    if (renderer.tile_streamer) |*streamer| streamer.deinit(renderer.gpa);
    renderer.shared.tiles.deinit(renderer.gpa);
    renderer.shared.earth_maps.deinit();
    freeInputFile(renderer.gpa, renderer.shared.cloud_volume);
    if (renderer.shared.probe_field == .file) freeInputFile(renderer.gpa, renderer.shared.probe_field.file);
    freeInputFile(renderer.gpa, renderer.shared.lut4d);
    freeInputFile(renderer.gpa, renderer.shared.network);
    renderer.shared.luts.deinit(renderer.gpa);
    renderer.* = undefined;
}

/// Samples per pixel accumulated since the last reset.
pub fn sampleCount(renderer: *const Renderer) u32 {
    return renderer.frame_uniforms.frame[0];
}

pub fn size(renderer: *const Renderer) [2]u32 {
    return .{ renderer.frame_uniforms.frame[1], renderer.frame_uniforms.frame[2] };
}

/// Records one sample per pixel from the active mode and the display pass into one
/// command list and submits it; nothing waits for the GPU.
pub fn render(renderer: *Renderer, view: View, tone: Post.Tone) Post.RecordError!void {
    var frame_uniforms = renderer.frame_uniforms;
    view.writeUniforms(&frame_uniforms, renderer.size());
    if (!std.meta.eql(frame_uniforms, renderer.frame_uniforms)) frame_uniforms.frame[0] = 0;
    if (renderer.tile_streamer) |*streamer| if (!renderer.tiles_frozen) {
        // Samples of coarser tiles would blur what the new ones resolve.
        const changed = streamer.update(renderer.gpa, renderer.shared.device, &renderer.shared.tiles, renderer.tile_files.?) catch |err| failed: {
            std.log.scoped(.render).err("tile streaming stopped: {t}", .{err});
            streamer.deinit(renderer.gpa);
            renderer.tile_streamer = null;
            break :failed false;
        };
        if (changed and renderer.tiles) frame_uniforms.frame[0] = 0;
    };

    const frame: Frame = .{
        .uniforms = frame_uniforms,
        .scene = renderer.sceneUniforms(),
        .cloud_bench = renderer.cloud_bench.benchUniforms(renderer.single_scattering),
        .accum = renderer.accum,
    };
    const command_list: gpu.CommandList = try .begin(renderer.shared.device);
    const streamer = if (renderer.tile_streamer) |*streamer| streamer else null;
    const recorded = switch (renderer.mode) {
        inline else => |*mode| record: {
            if (streamer) |s| s.recordUploads(command_list, &renderer.shared.tiles) catch |err| break :record err;
            mode.record(command_list, &renderer.shared, &frame) catch |err| break :record err;
            if (streamer) |s| s.recordFeedback(command_list, &renderer.shared.tiles) catch |err| break :record err;
            break :record renderer.post.record(
                command_list,
                &renderer.shared.luts,
                renderer.accum,
                renderer.image,
                tone,
                frame_uniforms.frame[0] + 1,
            );
        },
    };
    try command_list.submit(if (streamer) |s| s.signal() else null);
    if (streamer) |s| s.submitted();
    try recorded;

    frame_uniforms.frame[0] += 1;
    renderer.frame_uniforms = frame_uniforms;
}

/// Replaces the active mode, uploading the Earth maps if it is the first to sample
/// them; on failure the previous mode stays active.
pub fn setMode(renderer: *Renderer, kind: ModeKind) !void {
    if (renderer.mode == kind) return;
    var mode: Mode = try .init(kind, renderer.gpa, &renderer.shared, renderer.preset);
    errdefer mode.deinit(renderer.gpa);
    _ = try prepareEarthMaps(renderer.gpa, &renderer.shared, renderer.earth_map_cache, &mode, renderer.preset, renderer.texture_quality);
    try renderer.shared.device.waitIdle();
    renderer.mode.deinit(renderer.gpa);
    renderer.mode = mode;
    renderer.restart();
}

/// Makes `file` the `neural` mode's network, reading it now if that mode is active; on
/// failure the previous network stays.
pub fn setNetwork(renderer: *Renderer, file: InputFile) (NeuralMode.InitError || error{OutOfMemory})!void {
    const copy = (try copyInputFile(renderer.gpa, file)).?;
    const previous = renderer.shared.network;
    renderer.shared.network = copy;
    switch (renderer.mode) {
        .neural => |*mode| {
            mode.select(renderer.gpa, &renderer.shared) catch |err| {
                renderer.shared.network = previous;
                freeInputFile(renderer.gpa, copy);
                return err;
            };
            renderer.restart();
        },
        else => {},
    }
    freeInputFile(renderer.gpa, previous);
}

/// Switches the preset, uploading the Earth maps if the active mode samples them under
/// it; on failure the previous preset stays active.
pub fn setPreset(renderer: *Renderer, preset: scene.Preset) (error{EarthMapsMissing} || GpuEarthMaps.LoadError)!void {
    if (renderer.preset == preset) return;
    try checkPreset(renderer.earth_map_cache, preset);
    _ = try prepareEarthMaps(renderer.gpa, &renderer.shared, renderer.earth_map_cache, &renderer.mode, preset, renderer.texture_quality);
    renderer.preset = preset;
    switch (renderer.mode) {
        inline else => |*mode| mode.invalidate(&renderer.sceneUniforms()),
    }
    renderer.restart();
}

/// Takes the surface from the tiles or from the Earth maps; without tile files the maps
/// stay.
pub fn setTiles(renderer: *Renderer, tiles: bool) void {
    const effective = tiles and renderer.tile_files != null;
    if (renderer.tiles == effective) return;
    renderer.tiles = effective;
    renderer.restart();
}

/// Turns the cloud shell on or off; scenes without clouds are unaffected.
pub fn setClouds(renderer: *Renderer, clouds: bool) void {
    if (renderer.clouds == clouds) return;
    renderer.clouds = clouds;
    renderer.restart();
}

/// Caps the bounces of every path; single scattering caps them at one regardless.
pub fn setBounceLimit(renderer: *Renderer, limit: u32) void {
    if (renderer.bounce_limit == limit) return;
    renderer.bounce_limit = limit;
    renderer.restart();
}

pub fn setCloudsOnly(renderer: *Renderer, clouds_only: bool) void {
    if (renderer.clouds_only == clouds_only) return;
    renderer.clouds_only = clouds_only;
    renderer.restart();
}

pub fn setCloudShadowMap(renderer: *Renderer, cloud_shadow_map: bool) void {
    if (renderer.cloud_shadow_map == cloud_shadow_map) return;
    renderer.cloud_shadow_map = cloud_shadow_map;
    renderer.restart();
}

pub fn setCloudScattering(renderer: *Renderer, cloud_scattering: scene.CloudScattering) void {
    if (std.meta.eql(renderer.cloud_scattering, cloud_scattering)) return;
    renderer.cloud_scattering = cloud_scattering;
    renderer.restart();
}

pub fn setWater(renderer: *Renderer, water: scene.Water) void {
    if (std.meta.eql(renderer.water, water)) return;
    renderer.water = water;
    renderer.restart();
}

/// The active preset with this renderer's overrides applied.
fn sceneUniforms(renderer: *const Renderer) uniforms.SceneUniforms {
    var result = renderer.preset.sceneUniforms();
    if (!renderer.clouds) result = scene.withoutClouds(result);
    result = scene.withWater(result, renderer.water);
    result = scene.withCloudScattering(result, renderer.cloud_scattering);
    if (renderer.clouds_only) result = scene.cloudsOnly(result);
    if (!renderer.cloud_shadow_map) result = scene.marchedCloudShadows(result);
    result = scene.withBounceLimit(result, renderer.bounce_limit);
    if (renderer.tiles) result = scene.withTiles(result);
    if (renderer.single_scattering) result = scene.singleScattering(result);
    return result;
}

/// Replaces the Earth maps with those of `quality` if the active mode samples them under
/// the active preset; otherwise the choice is recorded for the next mode that does.
pub fn setTextureQuality(renderer: *Renderer, quality: EarthMaps.Quality) GpuEarthMaps.LoadError!void {
    if (renderer.texture_quality == quality) return;
    if (try prepareEarthMaps(renderer.gpa, &renderer.shared, renderer.earth_map_cache, &renderer.mode, renderer.preset, quality)) {
        renderer.restart();
    }
    renderer.texture_quality = quality;
}

/// What the tile streamer holds and wants, for display; null without tile files.
pub const TileStatus = struct { height: usize, surface: usize, wanted: u32, settled: bool };

pub fn tileStatus(renderer: *const Renderer) ?TileStatus {
    const streamer = if (renderer.tile_streamer) |*streamer| streamer else return null;
    const pool = &renderer.shared.tiles;
    return .{
        .height = streamer.residentCount(pool, .height),
        .surface = streamer.residentCount(pool, .surface),
        .wanted = streamer.wanted_last,
        .settled = streamer.settled(),
    };
}

/// Whether the tiles the view wants have arrived, or there is nothing to stream.
pub fn tilesSettled(renderer: *const Renderer) bool {
    const streamer = renderer.tile_streamer orelse return true;
    return !renderer.tiles or streamer.settled();
}

/// Starts accumulating afresh with the next `render`.
pub fn restart(renderer: *Renderer) void {
    renderer.frame_uniforms.frame[0] = 0;
}

/// Recreates the accumulation and display textures at `new_size`; on failure the
/// renderer keeps its previous size.
pub fn resize(renderer: *Renderer, new_size: [2]u32) gpu.Error!void {
    if (std.mem.eql(u32, &new_size, &renderer.size())) return;
    const accum = try initAccum(renderer.shared.device, new_size);
    errdefer accum.deinit();
    const image = try initImage(renderer.shared.device, new_size);
    errdefer image.deinit();
    try renderer.shared.device.waitIdle();
    renderer.accum.deinit();
    renderer.image.deinit();
    renderer.accum = accum;
    renderer.image = image;
    renderer.frame_uniforms = resetUniforms(new_size, renderer.frame_uniforms.frame[3]);
}

fn copyInputFile(gpa: std.mem.Allocator, file: ?InputFile) error{OutOfMemory}!?InputFile {
    var copy = file orelse return null;
    copy.path = try gpa.dupe(u8, copy.path);
    return copy;
}

fn freeInputFile(gpa: std.mem.Allocator, file: ?InputFile) void {
    if (file) |owned| gpa.free(owned.path);
}

/// A preset that samples the Earth maps needs a cache to load them from, whether or
/// not the active mode samples them yet.
fn checkPreset(earth_map_cache: ?GpuEarthMaps.Cache, preset: scene.Preset) error{EarthMapsMissing}!void {
    if (!presetSamplesEarthMaps(preset) or earth_map_cache != null) return;
    std.log.scoped(.render).err("the {t} preset needs the Earth maps, which were not baked", .{preset});
    return error.EarthMapsMissing;
}

fn presetSamplesEarthMaps(preset: scene.Preset) bool {
    const flags: uniforms.SceneFlags = @bitCast(preset.sceneUniforms().config[0]);
    return flags.earth_maps;
}

/// Uploads the maps of `quality` into `shared` when `mode` samples them under `preset`
/// and the bound set was loaded for another quality; returns whether the bound set
/// changed. The new set is uploaded before the old one is freed, so on failure the
/// previous maps stay bound.
fn prepareEarthMaps(
    gpa: std.mem.Allocator,
    shared: *Shared,
    earth_map_cache: ?GpuEarthMaps.Cache,
    mode: *const Mode,
    preset: scene.Preset,
    quality: EarthMaps.Quality,
) GpuEarthMaps.LoadError!bool {
    if (!mode.samplesEarthMaps()) return false;
    if (!presetSamplesEarthMaps(preset)) return false;
    if (shared.earth_maps.quality == quality) return false;
    var earth_maps: GpuEarthMaps = try .load(gpa, shared.device, earth_map_cache.?, quality);
    errdefer earth_maps.deinit();
    try shared.device.waitIdle();
    shared.earth_maps.deinit();
    shared.earth_maps = earth_maps;
    return true;
}

/// Zeroed so that the first `render` sees a changed view and starts at frame 0.
fn resetUniforms(image_size: [2]u32, seed: u32) uniforms.FrameUniforms {
    var frame_uniforms = std.mem.zeroes(uniforms.FrameUniforms);
    frame_uniforms.frame = .{ 0, image_size[0], image_size[1], seed };
    return frame_uniforms;
}

fn initAccum(device: gpu.Device, image_size: [2]u32) gpu.Error!gpu.Texture {
    return .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = image_size[0],
        .height = image_size[1],
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true },
        .label = "accumulation",
    });
}

fn initImage(device: gpu.Device, image_size: [2]u32) gpu.Error!gpu.Texture {
    return .init(device, .{
        .kind = .@"2d",
        .format = .rgba8_unorm,
        .width = image_size[0],
        .height = image_size[1],
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true },
        .label = "display image",
    });
}
