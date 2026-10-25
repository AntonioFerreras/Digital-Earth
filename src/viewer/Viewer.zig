//! The interactive viewer: owns the window, the device, the surface, the UI, and the
//! renderer, and runs the frame loop until the window closes. The loop only records and
//! submits GPU work; it never waits on or reads back from the GPU.

const std = @import("std");
const Io = std.Io;
const c = @import("c");
const gpu = @import("../gpu.zig");
const DataRoot = @import("../assets/DataRoot.zig");
const Image = @import("../assets/Image.zig");
const Luts = @import("../assets/Luts.zig");
const GpuEarthMaps = @import("../render/GpuEarthMaps.zig");
const TilePool = @import("../render/TilePool.zig");
const Renderer = @import("../render/Renderer.zig");
const Post = @import("../render/Post.zig");
const View = @import("../render/View.zig");
const Window = @import("../platform/Window.zig");
const Config = @import("Config.zig");
const Camera = @import("Camera.zig");
const Controls = @import("Controls.zig");
const Panel = @import("Panel.zig");
const NetworkList = @import("NetworkList.zig");
const PresetList = @import("PresetList.zig");

const Viewer = @This();

io: Io,
gpa: std.mem.Allocator,
window: Window,
device: gpu.Device,
surface: gpu.Surface,
renderer: Renderer,
panel: Panel,
config: Config,
networks: NetworkList,
presets: PresetList,
controls: Controls,
/// The sun as rendered, easing towards the config's: its direction, and its elevation
/// and azimuth at the camera with their velocities, radians and radians per second. The
/// azimuths are unwrapped: the goal follows the config's the shorter way each frame, and
/// stays within `azimuth_lead_max` ahead, so that the eased one never turns back however
/// fast it is dragged, nor runs on for turns after.
sun_shown: View.Vec3,
sun_elevation: f64,
sun_azimuth: f64,
sun_azimuth_goal: f64,
sun_velocities: [2]f64,
/// The camera's eased thrust (forward, left, up) and the look not yet turned.
thrust_shown: [3]f64,
look_pending: [2]f64,
/// The axis a sun drag has locked to, 0 azimuth and 1 elevation, and the travel before
/// it has, until the button comes up.
sun_drag_axis: ?u1,
sun_drag_pending: [2]f64,
/// Borrowed; outlives the viewer.
data_root: DataRoot,

const log = std.log.scoped(.viewer);

pub const Options = struct {
    backend: gpu.Device.Backend,
    validation: bool,
    shader_search_paths: []const [*:0]const u8,
    data_root: DataRoot,
    /// Copied; the caller keeps ownership.
    config: Config,
    /// Without it only the presets that need no Earth maps run. Must outlive the viewer.
    earth_map_cache: ?GpuEarthMaps.Cache,
    /// The neural mode's first network; the path is copied.
    network: ?Renderer.InputFile,
    /// The bruneton mode's table; the path is copied.
    lut4d: ?Renderer.InputFile,
    /// The cloud bench's volume; the path is copied.
    cloud_volume: ?Renderer.InputFile,
    /// Must outlive the viewer.
    tile_files: ?*const TilePool.Files,
};

pub const Error = Window.Error || gpu.Error || Luts.LoadError || Renderer.InitError ||
    error{UnknownCameraResponse};

const window_size_default = [2]u32{ 1920, 1080 };
/// Progressive accumulation wants every frame the GPU can produce.
const vsync = false;
/// The cloud bench's cloud spans a few kilometres, seen from about as high.
const cloud_bench_speed_scale = 0.16;
/// The eased sun moves its elevation and azimuth at the camera apart: each speeds up at
/// `sun_acceleration` to `sun_speed` and brakes at the same rate to stop on the config's,
/// so that rising and setting cross each height alike. The elevation moves
/// `low_sun_slowdown` times as slowly at or below `low_sun_cos_zenith`, where the light
/// changes fastest, blending by a smoothstep up to `high_sun_cos_zenith`.
const sun_speed = std.math.degreesToRadians(90.0);
const sun_acceleration = std.math.degreesToRadians(360.0);
/// Azimuth moves take a third of the time: three times the speed, and nine times the
/// acceleration, so that speeding up and braking shrink alike.
const azimuth_speed = 3 * sun_speed;
const azimuth_acceleration = 9 * sun_acceleration;
const low_sun_slowdown = 2.0;
const low_sun_cos_zenith = 0.1;
const sun_drag_range_px = 600.0;
/// Travel before a sun drag locks to the axis it went further along.
const sun_drag_lock_px = 6.0;
const azimuth_lead_max = 0.9 * std.math.pi;
/// Time constants of the camera's eased thrust and look, and the thrust below which it
/// stops, so that accumulation resumes.
const move_smoothing_s = 0.15;
const look_smoothing_s = 0.05;
const motion_settled = 1e-3;
const high_sun_cos_zenith = 0.2;
/// Bounds the camera step after a stall (window drag, breakpoint), above the frame
/// times of a probe bake, so that the camera keeps its speed through one.
const frame_seconds_max = 0.5;

pub fn init(gpa: std.mem.Allocator, io: Io, options: Options) Error!Viewer {
    // The device comes first so a machine without a usable GPU fails before a window opens.
    const device: gpu.Device = try .init(.{
        .backend = options.backend,
        .validation = options.validation,
        .shader_search_paths = options.shader_search_paths,
        .shader_cache_path = null,
    });
    errdefer device.deinit();
    const info = device.info();
    log.info("adapter: {s}", .{std.mem.sliceTo(&info.adapter_name, 0)});
    const window: Window = try .init("Digital Earth", window_size_default);
    errdefer window.deinit();

    const surface: gpu.Surface = try .init(device, try window.nativeHandle());
    errdefer surface.deinit();
    try surface.configure(window.pixelSize(), vsync);
    try gpu.check(c.en_ui_init(device.handle, surface.handle, window.sdl_window), "en_ui_init");
    errdefer c.en_ui_shutdown();

    var config = try options.config.clone(gpa);
    errdefer config.deinit(gpa);
    if (options.earth_map_cache == null) config.preset = .symmetric;
    var lut_dir = try options.data_root.openSubdir(io, .lut, .{});
    defer lut_dir.close(io);
    var luts: Luts = try .load(gpa, io, lut_dir);
    defer luts.deinit(gpa);
    var panel: Panel = .{ .camera_response_names = try Panel.joinLabels(gpa, luts.crf_names) };
    errdefer panel.deinit(gpa);
    var networks: NetworkList = try .init(gpa, io, options.data_root, options.network);
    errdefer networks.deinit(gpa);
    var presets: PresetList = try .init(gpa, io, options.data_root);
    errdefer presets.deinit(gpa);
    var renderer_options: Renderer.Options = .{
        .size = window.pixelSize(),
        .mode = config.mode,
        .preset = config.preset,
        .clouds = config.clouds,
        .water = config.water,
        .cloud_scattering = config.cloud_scattering,
        .cloud_bench = config.cloud_bench,
        .clouds_only = config.clouds_only,
        .cloud_shadow_map = config.cloud_shadow_map,
        .bounce_limit = config.bounce_limit,
        .tile_files = options.tile_files,
        .tiles = config.tiles,
        .texture_quality = config.texture_quality,
        .earth_map_cache = options.earth_map_cache,
        .network = options.network,
        .lut4d = options.lut4d,
        .cloud_volume = options.cloud_volume,
    };
    // A mode whose input is missing or broken must not keep the viewer from starting.
    var renderer: Renderer = Renderer.init(gpa, device, &luts, renderer_options) catch |err| fallback: {
        log.err("starting in the {t} mode failed: {t}; starting in debug", .{ config.mode, err });
        config.mode = .debug;
        renderer_options.mode = .debug;
        break :fallback try .init(gpa, device, &luts, renderer_options);
    };
    errdefer renderer.deinit();
    // The panel relies on the config naming a loaded camera response.
    _ = try renderer.shared.luts.cameraResponseIndex(config.camera_response);
    const sun_at_camera = config.camera.horizon().elevationAzimuth(config.sunDirection());
    return .{
        .io = io,
        .gpa = gpa,
        .window = window,
        .device = device,
        .surface = surface,
        .renderer = renderer,
        .panel = panel,
        .config = config,
        .networks = networks,
        .presets = presets,
        .controls = .{},
        .sun_shown = config.sunDirection(),
        .sun_elevation = sun_at_camera[0],
        .sun_azimuth = sun_at_camera[1],
        .sun_azimuth_goal = sun_at_camera[1],
        .sun_velocities = @splat(0),
        .thrust_shown = @splat(0),
        .look_pending = @splat(0),
        .sun_drag_axis = null,
        .sun_drag_pending = @splat(0),
        .data_root = options.data_root,
    };
}

/// Releases in reverse creation order once the GPU is idle.
pub fn deinit(viewer: *Viewer) void {
    viewer.device.waitIdle() catch {};
    viewer.presets.deinit(viewer.gpa);
    viewer.networks.deinit(viewer.gpa);
    viewer.panel.deinit(viewer.gpa);
    viewer.config.deinit(viewer.gpa);
    viewer.renderer.deinit();
    c.en_ui_shutdown();
    viewer.surface.deinit();
    viewer.window.deinit();
    viewer.device.deinit();
    viewer.* = undefined;
}

pub fn run(viewer: *Viewer) Post.RecordError!void {
    var frame_start = Io.Timestamp.now(viewer.io, .awake);
    while (true) {
        const input = viewer.window.pump();
        if (input.quit) return;
        const size = viewer.window.pixelSize();
        if (size[0] == 0 or size[1] == 0) {
            // Minimized: nothing to present until the window comes back.
            c.SDL_Delay(16);
            continue;
        }
        if (input.resized) try viewer.surface.configure(size, vsync);
        try viewer.renderer.resize(size);

        const now = Io.Timestamp.now(viewer.io, .awake);
        const frame_ns: f64 = @floatFromInt(frame_start.durationTo(now).toNanoseconds());
        frame_start = now;
        const dt_s = @min(frame_ns / std.time.ns_per_s, frame_seconds_max);

        const intent = viewer.controls.map(input);
        var motion = intent.motion;
        viewer.easeMotion(&motion, dt_s);
        if (viewer.config.mode == .cloud_bench) motion.speed_scale = cloud_bench_speed_scale;
        _ = viewer.config.camera.update(motion, dt_s);
        if (intent.toggle_gui) viewer.config.gui_visible = !viewer.config.gui_visible;
        if (viewer.config.mode == .cloud_bench) viewer.dragSun(intent.sun_drag, intent.sun_drag_held);

        // A hidden panel still needs ImGui frames, or its input capture flags go stale.
        c.en_ui_new_frame();
        const actions: Panel.Actions = if (viewer.config.gui_visible)
            viewer.panel.draw(viewer.gpa, &viewer.config, &viewer.networks, &viewer.presets, &viewer.renderer)
        else
            .{};
        Panel.drawBakeProgress(&viewer.renderer);
        if (intent.save_config) viewer.saveConfig(DataRoot.config_path);
        if (intent.load_config) viewer.loadConfig(DataRoot.config_path);
        if (actions.save_preset) viewer.savePreset();
        if (actions.load_preset) viewer.loadPreset();
        if (actions.rescan_networks) viewer.rescanNetworks();
        viewer.applySelections();
        if (actions.restart) viewer.renderer.restart();
        if (actions.reset_probes) switch (viewer.renderer.mode) {
            .cloud_bench => |*mode| mode.resetProbes(),
            else => {},
        };
        viewer.easeSun(dt_s);
        try viewer.renderer.render(viewer.config.camera.view(viewer.sun_shown), viewer.config.tone());
        if (viewer.config.auto_exposure) viewer.config.exposure = Post.autoExposure(viewer.config.camera.cosZenith(viewer.sun_shown));
        if (intent.screenshot or actions.screenshot) viewer.saveScreenshot();
        viewer.surface.present(viewer.device, viewer.renderer.image) catch |err| switch (err) {
            error.SurfaceOutOfDate => try viewer.surface.configure(size, vsync),
            else => return err,
        };
    }
}

/// Moves the rendered sun towards the config's along the sphere.
/// Eases the keys' thrust, and releases the look's travel over a few frames.
fn easeMotion(viewer: *Viewer, motion: *Camera.Motion, dt_s: f64) void {
    const thrust = [3]f64{ motion.forward, motion.left, motion.up };
    const thrust_blend = 1 - @exp(-dt_s / move_smoothing_s);
    for (&viewer.thrust_shown, thrust) |*shown, wanted| {
        shown.* += (wanted - shown.*) * thrust_blend;
        if (wanted == 0 and @abs(shown.*) < motion_settled) shown.* = 0;
    }
    motion.forward, motion.left, motion.up = viewer.thrust_shown;
    const look_blend = 1 - @exp(-dt_s / look_smoothing_s);
    for (&viewer.look_pending, &motion.drag) |*pending, *drag| {
        pending.* += drag.*;
        drag.* = if (@abs(pending.*) < motion_settled * 1e-3) pending.* else pending.* * look_blend;
        pending.* -= drag.*;
    }
}

/// Sideways turns the azimuth, up raises the sun, each through its slider's range per
/// `sun_drag_range_px`, as a slider about that wide moves it; one or the other, by the
/// axis the drag set out along.
fn dragSun(viewer: *Viewer, drag_px: [2]f64, held: bool) void {
    if (!held) {
        viewer.sun_drag_axis = null;
        viewer.sun_drag_pending = @splat(0);
        return;
    }
    var travel = drag_px;
    if (viewer.sun_drag_axis == null) {
        const pending = &viewer.sun_drag_pending;
        pending.* = .{ pending[0] + drag_px[0], pending[1] + drag_px[1] };
        if (@max(@abs(pending[0]), @abs(pending[1])) < sun_drag_lock_px) return;
        viewer.sun_drag_axis = @intFromBool(@abs(pending[1]) > @abs(pending[0]));
        travel = pending.*;
    }
    const sun = &viewer.config.bench_sun;
    switch (viewer.sun_drag_axis.?) {
        0 => {
            const azimuth = Panel.ranges.bench_sun_azimuth_degrees;
            const turned = sun.azimuth_degrees - travel[0] / sun_drag_range_px * (azimuth.max - azimuth.min);
            sun.azimuth_degrees = turned - 360 * @round(turned / 360);
        },
        1 => {
            const height = Panel.ranges.bench_sun_cos_zenith;
            sun.cos_zenith = std.math.clamp(sun.cos_zenith - travel[1] / sun_drag_range_px * (height.max - height.min), height.min, height.max);
        },
    }
}

fn easeSun(viewer: *Viewer, dt_s: f64) void {
    const target = viewer.config.sunDirection();
    const horizon = viewer.config.camera.horizon();
    const wanted = horizon.elevationAzimuth(target);
    const turn = wanted[1] - viewer.sun_azimuth_goal;
    viewer.sun_azimuth_goal += std.math.atan2(@sin(turn), @cos(turn));
    // A goal more than half a turn ahead would be reached the long way round.
    const lead = std.math.clamp(viewer.sun_azimuth_goal - viewer.sun_azimuth, -azimuth_lead_max, azimuth_lead_max);
    viewer.sun_azimuth_goal = viewer.sun_azimuth + lead;
    const t = std.math.clamp((@sin(viewer.sun_elevation) - low_sun_cos_zenith) / (high_sun_cos_zenith - low_sun_cos_zenith), 0, 1);
    const pace = std.math.lerp(1.0 / low_sun_slowdown, 1.0, t * t * (3 - 2 * t));
    const velocities = &viewer.sun_velocities;
    viewer.sun_elevation = approach(viewer.sun_elevation, wanted[0], &velocities[0], pace * sun_speed, pace * sun_acceleration, dt_s);
    viewer.sun_azimuth = approach(viewer.sun_azimuth, viewer.sun_azimuth_goal, &velocities[1], azimuth_speed, azimuth_acceleration, dt_s);
    // Arrived, the config's own direction, so that accumulation resumes.
    const arrived = viewer.sun_elevation == wanted[0] and viewer.sun_azimuth == viewer.sun_azimuth_goal;
    viewer.sun_shown = if (arrived) target else horizon.direction(viewer.sun_elevation, viewer.sun_azimuth);
}

/// `value` moved towards `target` at `velocity`, which changes by at most `acceleration`
/// a second, up to `max_speed`, and never so fast that it cannot brake to a stop on the
/// target; it lands there exactly.
fn approach(value: f64, target: f64, velocity: *f64, max_speed: f64, acceleration: f64, dt_s: f64) f64 {
    const gap = target - value;
    const wanted = std.math.copysign(@min(max_speed, @sqrt(2 * acceleration * @abs(gap))), gap);
    velocity.* += std.math.clamp(wanted - velocity.*, -acceleration * dt_s, acceleration * dt_s);
    const step = velocity.* * dt_s;
    if (step * gap >= 0 and @abs(step) >= @abs(gap)) {
        velocity.* = 0;
        return target;
    }
    return value + step;
}

/// A mode or network that fails to load or build (e.g. a broken file or a shader
/// compile error, already logged) leaves the previous one running and selected.
fn applySelections(viewer: *Viewer) void {
    const networks = &viewer.networks;
    if (networks.selected != networks.active) {
        if (viewer.renderer.setNetwork(networks.files[networks.selected])) {
            networks.active = networks.selected;
        } else |err| {
            log.err("switching to the network {s} failed: {t}", .{ networks.label(networks.selected), err });
            networks.selected = networks.active;
        }
    }
    viewer.renderer.setMode(viewer.config.mode) catch |err| {
        log.err("switching to {t} failed: {t}", .{ viewer.config.mode, err });
        viewer.config.mode = viewer.renderer.mode;
    };
    viewer.renderer.setPreset(viewer.config.preset) catch |err| {
        log.err("switching to the {t} preset failed: {t}", .{ viewer.config.preset, err });
        viewer.config.preset = viewer.renderer.preset;
    };
    viewer.renderer.setClouds(viewer.config.clouds);
    viewer.renderer.setWater(viewer.config.water);
    viewer.renderer.setCloudScattering(viewer.config.cloud_scattering);
    viewer.renderer.cloud_bench = viewer.config.cloud_bench;
    viewer.renderer.setCloudsOnly(viewer.config.clouds_only);
    viewer.renderer.setCloudShadowMap(viewer.config.cloud_shadow_map);
    viewer.renderer.setBounceLimit(viewer.config.bounce_limit);
    viewer.renderer.setTiles(viewer.config.tiles);
    viewer.renderer.setTextureQuality(viewer.config.texture_quality) catch |err| {
        log.err("switching to Earth maps {t} failed: {t}", .{ viewer.config.texture_quality, err });
        viewer.config.texture_quality = viewer.renderer.texture_quality;
    };
}

/// Lists <data>/cache/neural afresh, after the renderer's network.
fn rescanNetworks(viewer: *Viewer) void {
    const networks = NetworkList.init(viewer.gpa, viewer.io, viewer.data_root, viewer.renderer.shared.network) catch {
        log.err("out of memory listing the networks", .{});
        return;
    };
    viewer.networks.deinit(viewer.gpa);
    viewer.networks = networks;
}

/// A failed save, load, or screenshot leaves the session running; the cause is logged.
/// `path` is relative to the data root.
fn saveConfig(viewer: *Viewer, path: []const u8) void {
    viewer.config.save(viewer.gpa, viewer.io, viewer.data_root.dir, path) catch |err| {
        viewer.data_root.logFailure(path, err);
        return;
    };
    log.info("saved {s}", .{path});
}

/// Saves to the presets directory under the save field's name, then lists it.
fn savePreset(viewer: *Viewer) void {
    const name = viewer.presets.savedName() orelse {
        log.err("a preset needs a name, without path separators or a leading dot", .{});
        return;
    };
    var dir = viewer.data_root.createSubdir(viewer.io, .presets) catch return;
    dir.close(viewer.io);
    var buffer: PresetList.PathBuffer = undefined;
    viewer.saveConfig(PresetList.path(&buffer, name));
    viewer.presets.rescan(viewer.gpa, viewer.io, viewer.data_root, name) catch log.err("out of memory listing the presets", .{});
}

/// Loads the picker's preset and puts its name in the save field.
fn loadPreset(viewer: *Viewer) void {
    const presets = &viewer.presets;
    const name = presets.stems[presets.selected];
    var buffer: PresetList.PathBuffer = undefined;
    viewer.loadConfig(PresetList.path(&buffer, name));
    presets.setName(name);
}

/// `path` is relative to the data root.
fn loadConfig(viewer: *Viewer, path: []const u8) void {
    var config = Config.load(viewer.gpa, viewer.io, viewer.data_root.dir, path) catch |err| {
        viewer.data_root.logFailure(path, err);
        return;
    };
    _ = viewer.renderer.shared.luts.cameraResponseIndex(config.camera_response) catch {
        config.deinit(viewer.gpa);
        return;
    };
    viewer.config.deinit(viewer.gpa);
    viewer.config = config;
    log.info("loaded {s}", .{path});
}

/// Writes the display image, as presented, to `screenshots/<UTC time>.png`. Reading it
/// back waits for the GPU once.
fn saveScreenshot(viewer: *Viewer) void {
    var name_buffer: [screenshot_name_max]u8 = undefined;
    const now: u64 = @intCast(Io.Timestamp.now(viewer.io, .real).toSeconds());
    const name = screenshotName(&name_buffer, now);
    viewer.writeScreenshot(name) catch |err| {
        var path_buffer: [screenshot_name_max + 32]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buffer, "{s}/{s}", .{ DataRoot.Subdir.screenshots.path(), name }) catch unreachable;
        viewer.data_root.logFailure(path, err);
        return;
    };
    log.info("saved {s}/{s}", .{ DataRoot.Subdir.screenshots.path(), name });
}

fn writeScreenshot(viewer: *Viewer, name: []const u8) !void {
    var dir = try viewer.data_root.createSubdir(viewer.io, .screenshots);
    defer dir.close(viewer.io);
    const image = viewer.renderer.image;
    const pixels = try viewer.gpa.alloc(u8, @as(usize, image.desc.width) * image.desc.height * 4);
    defer viewer.gpa.free(pixels);
    try image.read(viewer.device, 0, pixels);
    try Image.writePng(viewer.io, dir, name, image.desc.width, image.desc.height, 4, pixels);
}

const screenshot_name_max = "YYYY-MM-DD-HHMMSS.png".len;

/// The original viewer's timestamp format, in UTC.
fn screenshotName(buffer: *[screenshot_name_max]u8, epoch_seconds: u64) []const u8 {
    const seconds: std.time.epoch.EpochSeconds = .{ .secs = epoch_seconds };
    const year_day = seconds.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_seconds = seconds.getDaySeconds();
    return std.fmt.bufPrint(buffer, "{d:0>4}-{d:0>2}-{d:0>2}-{d:0>2}{d:0>2}{d:0>2}.png", .{
        year_day.year,
        @backingInt(month_day.month),
        month_day.day_index + 1,
        day_seconds.getHoursIntoDay(),
        day_seconds.getMinutesIntoHour(),
        day_seconds.getSecondsIntoMinute(),
    }) catch unreachable;
}

test "screenshot names are UTC timestamps" {
    var buffer: [screenshot_name_max]u8 = undefined;
    try std.testing.expectEqualStrings("1970-01-01-000000.png", screenshotName(&buffer, 0));
    try std.testing.expectEqualStrings("2026-10-18-153007.png", screenshotName(&buffer, 1792337407));
}
