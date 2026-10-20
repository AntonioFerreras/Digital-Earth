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
const Renderer = @import("../render/Renderer.zig");
const Post = @import("../render/Post.zig");
const Window = @import("../platform/Window.zig");
const Config = @import("Config.zig");
const Controls = @import("Controls.zig");
const Panel = @import("Panel.zig");
const NetworkList = @import("NetworkList.zig");

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
controls: Controls,
/// Borrowed; outlives the viewer.
data_root: DataRoot,
/// Where the config dialog opens: the presets directory, absolute and with a trailing
/// separator, which marks it a folder. The dialog may read it until the user answers.
dialog_location: [Io.Dir.max_path_bytes + 1:0]u8,

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
};

pub const Error = Window.Error || gpu.Error || Luts.LoadError || Renderer.InitError ||
    error{UnknownCameraResponse};

const window_size_default = [2]u32{ 1920, 1080 };
/// Progressive accumulation wants every frame the GPU can produce.
const vsync = false;
/// Bounds the camera step after a stall (window drag, breakpoint).
const frame_seconds_max = 0.1;
/// Static because the dialog reads it until the user answers.
const config_file_filters = [_]c.SDL_DialogFileFilter{.{ .name = "Digital Earth config", .pattern = "zon" }};

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
    var panel: Panel = try .init(gpa, luts.crf_names);
    errdefer panel.deinit(gpa);
    var networks: NetworkList = try .init(gpa, io, options.data_root, options.network);
    errdefer networks.deinit(gpa);
    var renderer_options: Renderer.Options = .{
        .size = window.pixelSize(),
        .mode = config.mode,
        .preset = config.preset,
        .clouds = config.clouds,
        .texture_quality = config.texture_quality,
        .earth_map_cache = options.earth_map_cache,
        .network = options.network,
        .lut4d = options.lut4d,
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
        .controls = .{},
        .data_root = options.data_root,
        .dialog_location = undefined,
    };
}

/// Releases in reverse creation order once the GPU is idle.
pub fn deinit(viewer: *Viewer) void {
    viewer.device.waitIdle() catch {};
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
        _ = viewer.config.camera.update(intent.motion, dt_s);
        if (intent.toggle_gui) viewer.config.gui_visible = !viewer.config.gui_visible;

        // A hidden panel still needs ImGui frames, or its input capture flags go stale.
        c.en_ui_new_frame();
        const actions: Panel.Actions = if (viewer.config.gui_visible)
            viewer.panel.draw(viewer.gpa, &viewer.config, &viewer.networks, &viewer.renderer)
        else
            .{};
        if (intent.save_config or actions.save_config) viewer.saveConfig();
        if (actions.rescan_networks) viewer.rescanNetworks();
        if (actions.load_config) viewer.window.showOpenDialog(&config_file_filters, viewer.presetsLocation());
        if (intent.load_config) viewer.loadConfig(DataRoot.config_path);
        if (input.opened_path) |path| viewer.loadConfig(path);
        viewer.applySelections();
        if (actions.restart) viewer.renderer.restart();
        try viewer.renderer.render(viewer.config.camera.view(viewer.config.sun), viewer.config.tone());
        if (intent.screenshot or actions.screenshot) viewer.saveScreenshot();
        viewer.surface.present(viewer.device, viewer.renderer.image) catch |err| switch (err) {
            error.SurfaceOutOfDate => try viewer.surface.configure(size, vsync),
            else => return err,
        };
    }
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
fn saveConfig(viewer: *Viewer) void {
    viewer.config.save(viewer.gpa, viewer.io, viewer.data_root.dir, DataRoot.config_path) catch |err| {
        viewer.data_root.logFailure(DataRoot.config_path, err);
        return;
    };
    log.info("saved {s}", .{DataRoot.config_path});
}

/// The presets directory in `dialog_location`, or null when it cannot be resolved.
fn presetsLocation(viewer: *Viewer) ?[*:0]const u8 {
    const subdir = DataRoot.Subdir.presets.path();
    const buffer = &viewer.dialog_location;
    const len = viewer.data_root.dir.realPathFile(viewer.io, subdir, buffer[0 .. buffer.len - 1]) catch |err| {
        viewer.data_root.logFailure(subdir, err);
        return null;
    };
    buffer[len] = std.fs.path.sep;
    buffer[len + 1] = 0;
    return buffer[0 .. len + 1 :0];
}

/// `path` is absolute or relative to the data root.
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
