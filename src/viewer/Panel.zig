//! The ImGui parameter panel. It edits the viewer's `Config` in place and reports the
//! buttons pressed; whether an edit restarts accumulation is the renderer's decision.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const DataRoot = @import("../assets/DataRoot.zig");
const EarthMaps = @import("../assets/EarthMaps.zig");
const Renderer = @import("../render/Renderer.zig");
const View = @import("../render/View.zig");
const scene = @import("../render/scene.zig");
const Post = @import("../render/Post.zig");
const Config = @import("Config.zig");
const InputList = @import("InputList.zig");
const PresetList = @import("PresetList.zig");

const Panel = @This();

/// Camera response names separated and terminated by NUL, as `ImGui_Combo` takes them.
camera_response_names: [:0]u8,

pub const Range = struct { min: f64, max: f64 };

/// Slider limits, also enforced on Ctrl+click input. Loaded configs may lie outside
/// them; a slider shows such a value until it is dragged.
pub const ranges = struct {
    pub const sun_angle_degrees: Range = .{ .min = 0, .max = 360 };
    pub const sun_path_rotation_degrees: Range = .{ .min = -105, .max = 105 };
    /// From the lowest sun auto exposure knows (Post.auto_exposure_table) to overhead.
    pub const bench_sun_cos_zenith: Range = .{ .min = Post.auto_exposure_table[0][0], .max = 1 };
    pub const bench_sun_azimuth_degrees: Range = .{ .min = -180, .max = 180 };
    /// The original slider stopped at tan(half angle) = pi/2, about 115 degrees.
    pub const vertical_fov_degrees: Range = .{ .min = View.vertical_fov_degrees_min, .max = 115 };
    pub const aspect_scale: Range = .{ .min = 0.75, .max = 1.25 };
    pub const exposure: Range = .{ .min = -1, .max = 11 };
    pub const bounce_limit: Range = .{ .min = 1, .max = scene.atmosphere.bounce_count_max };
    pub const wind_speed: Range = .{ .min = 0, .max = 25 };
    pub const texture_strength: Range = .{ .min = 0, .max = 2 };
    pub const texture_size: Range = .{ .min = 0.05, .max = 2 };
    pub const wind_map_influence: Range = .{ .min = 0, .max = 1 };
    pub const glint_tail: Range = .{ .min = 0, .max = 1 };
    pub const streak_length: Range = .{ .min = 0, .max = 6 };
    pub const texture_warp: Range = .{ .min = 0, .max = 2 };
    pub const wind_roughness: Range = .{ .min = 0, .max = 1 };
    pub const cloud_strength: Range = .{ .min = 0, .max = 50 };
    pub const cloud_rate: Range = .{ .min = 0, .max = 1 };
    pub const cloud_relaxed_depth: Range = .{ .min = 0.1, .max = 100 };
    pub const cloud_phase_g: Range = .{ .min = -0.9, .max = 0.95 };
    pub const cloud_profile_strength: Range = .{ .min = 0, .max = 1 };
    pub const cloud_profile_depth: Range = .{ .min = 0.1, .max = 100 };
    pub const bench_bounce_limit: Range = .{ .min = 1, .max = 16384 };
    pub const bench_probe_walks: Range = .{ .min = 1, .max = 65536 };
    /// The walks' sums take over a kilobyte per probe.
    pub const bench_probe_budget: Range = .{ .min = 1 << 12, .max = 1 << 19 };
    pub const bench_probe_depth_ratio: Range = .{ .min = 0.5, .max = 8 };
    /// From the march's shortest step to past a majorant cell, where the cap is off.
    pub const bench_view_step_max_voxels: Range = .{ .min = 0.5, .max = 64 };
    /// Greyscale up to a strong chroma boost; 1 keeps OpenDRT's look.
    pub const saturation: Range = .{ .min = 0, .max = 3.6 };
    /// Keeps every preset's contrast power at or above the shader's floor of 0.25.
    pub const contrast: Range = .{ .min = -0.75, .max = 1 };
    pub const gamma: Range = .{ .min = 0.45, .max = 2.2 };
};

/// Buttons pressed this frame.
pub const Actions = struct {
    restart: bool = false,
    /// To the save field's name in the presets directory, and from the picker's.
    save_preset: bool = false,
    load_preset: bool = false,
    screenshot: bool = false,
    rescan_networks: bool = false,
    rescan_clouds: bool = false,
    /// The cloud bench's probes, for their present settings.
    reset_probes: bool = false,
};

/// Throughput of progressive rendering, which adds one sample per pixel per frame.
pub const Readout = struct {
    sample_count: u32,
    frame_ms: f64,
    samples_per_second: f64,

    /// A zero frame rate, before ImGui has measured one, reads as an infinite frame time.
    pub fn init(frames_per_second: f64, image_size: [2]u32, sample_count: u32) Readout {
        assert(frames_per_second >= 0);
        const pixel_count: f64 = @floatFromInt(@as(u64, image_size[0]) * image_size[1]);
        return .{
            .sample_count = sample_count,
            .frame_ms = std.time.ms_per_s / frames_per_second,
            .samples_per_second = frames_per_second * pixel_count,
        };
    }
};

/// `names`, each NUL-terminated, as `ImGui_Combo` takes them.
pub fn joinLabels(gpa: std.mem.Allocator, names: []const []const u8) error{OutOfMemory}![:0]u8 {
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(gpa);
    for (names) |name| {
        assert(std.mem.findScalar(u8, name, 0) == null);
        try joined.appendSlice(gpa, name);
        try joined.append(gpa, 0);
    }
    return joined.toOwnedSliceSentinel(gpa, 0);
}

pub fn deinit(panel: *Panel, gpa: std.mem.Allocator) void {
    gpa.free(panel.camera_response_names);
    panel.* = undefined;
}

/// `config.camera_response` must name a loaded camera response; `gpa` owns it. The
/// network and cloud pickers edit their lists' `selected`, the preset picker and save
/// field `presets`.
pub fn draw(panel: Panel, gpa: std.mem.Allocator, config: *Config, networks: *InputList, clouds: *InputList, presets: *PresetList, renderer: *const Renderer) Actions {
    var actions: Actions = .{};
    c.ImGui_SetNextWindowPos(.{ .x = 24, .y = 24 }, c.ImGuiCond_FirstUseEver);
    defer c.ImGui_End();
    if (!c.ImGui_Begin("Digital Earth", null, c.ImGuiWindowFlags_AlwaysAutoResize)) return actions;

    const readout: Readout = .init(c.ImGui_GetIO().*.Framerate, renderer.size(), renderer.sampleCount());
    var text: [96]u8 = undefined;
    c.ImGui_TextUnformatted(std.mem.printSentinel(&text, "{d} spp   {d:.2} ms/frame   {d:.1} Msamples/s", .{
        readout.sample_count, readout.frame_ms, readout.samples_per_second / 1e6,
    }, 0) catch unreachable);

    if (c.ImGui_CollapsingHeader("render", c.ImGuiTreeNodeFlags_DefaultOpen)) {
        enumCombo(Renderer.ModeKind, "mode", &config.mode);
    }
    // The cloud bench has a world of its own, which uses none of the Earth's settings.
    if (config.mode == .cloud_bench) {
        drawCloudBench(config, clouds, renderer, &text, &actions);
    } else {
        drawEarth(config, networks, renderer, &text, &actions);
    }
    // Only the ray marcher models the clouds' multiple scattering.
    if (config.mode == .raymarch and c.ImGui_CollapsingHeader("cloud scattering", 0)) {
        const s = &config.cloud_scattering;
        logSlider("strength", &s.strength, ranges.cloud_strength);
        slider("rate away from sun", &s.rate_away, ranges.cloud_rate);
        slider("rate to sun, surface", &s.rate_surface, ranges.cloud_rate);
        slider("rate to sun, deep", &s.rate_deep, ranges.cloud_rate);
        logSlider("relaxed depth", &s.relaxed_depth, ranges.cloud_relaxed_depth);
        slider("phase g", &s.phase_g, ranges.cloud_phase_g);
        // Only the cloud bench's volume has a shape to take the profile from.
        if (config.mode == .cloud_bench) {
            slider("profile strength", &s.profile_strength, ranges.cloud_profile_strength);
            logSlider("profile depth", &s.profile_depth, ranges.cloud_profile_depth);
        }
        if (c.ImGui_Button("reset cloud scattering")) s.* = .{};
    }
    if (c.ImGui_CollapsingHeader("camera", c.ImGuiTreeNodeFlags_DefaultOpen)) {
        const altitude_m = config.camera.altitude();
        c.ImGui_TextUnformatted(if (altitude_m < 10e3)
            std.mem.printSentinel(&text, "elevation {d:.0} m", .{altitude_m}, 0) catch unreachable
        else
            std.mem.printSentinel(&text, "elevation {d:.2} km", .{altitude_m / 1e3}, 0) catch unreachable);
        slider("vertical fov", &config.camera.vertical_fov_degrees, ranges.vertical_fov_degrees);
        slider("aspect scale", &config.camera.aspect_scale, ranges.aspect_scale);
        _ = c.ImGui_Checkbox("vsync", &config.vsync);
        _ = c.ImGui_Checkbox("auto exposure", &config.auto_exposure);
        slider("exposure", &config.exposure, ranges.exposure);
        slider("saturation", &config.saturation, ranges.saturation);
        slider("contrast", &config.contrast, ranges.contrast);
        _ = c.ImGui_Checkbox("tone mapping", &config.tone_map);
        enumCombo(Post.TonescalePreset, "tonescale", &config.tonescale);
        const luts = &renderer.shared.luts;
        var camera_response: c_int = @intCast(luts.cameraResponseIndex(config.camera_response) catch unreachable);
        if (c.ImGui_Combo("camera response", &camera_response, panel.camera_response_names)) {
            const name = luts.camera_response_names[@intCast(camera_response)];
            config.setCameraResponse(gpa, name) catch std.log.scoped(.viewer).err("out of memory selecting {s}", .{name});
        }
        slider("gamma", &config.gamma, ranges.gamma);
    }

    c.ImGui_Separator();
    if (presets.stems.len > 0) {
        var selected: c_int = @intCast(presets.selected);
        if (c.ImGui_Combo("preset", &selected, presets.labels)) presets.selected = @intCast(selected);
        c.ImGui_SameLine();
        actions.load_preset = c.ImGui_Button("load");
    }
    _ = c.ImGui_InputText("preset name", &presets.name, presets.name.len + 1, 0);
    c.ImGui_SameLine();
    actions.save_preset = c.ImGui_Button("save");
    actions.restart = c.ImGui_Button("reset accumulation");
    c.ImGui_SameLine();
    actions.screenshot = c.ImGui_Button("screenshot");
    c.ImGui_TextDisabled("G panel   I/O save/load " ++ DataRoot.config_path ++ "   P screenshot");
    return actions;
}

/// The Earth's settings.
fn drawEarth(config: *Config, networks: *InputList, renderer: *const Renderer, text: *[96]u8, actions: *Actions) void {
    if (c.ImGui_CollapsingHeader("scene", c.ImGuiTreeNodeFlags_DefaultOpen)) {
        enumCombo(scene.Preset, "scene preset", &config.preset);
        enumCombo(EarthMaps.Quality, "texture quality", &config.texture_quality);
        actions.rescan_networks = inputCombo("network", networks);
        // The ray marcher takes at most its own few bounces under the limit.
        if (config.mode == .path_tracer or config.mode == .raymarch) {
            const bounce_limit_min: u32 = @intFromFloat(ranges.bounce_limit.min);
            const bounce_limit_max: u32 = @intFromFloat(ranges.bounce_limit.max);
            _ = c.ImGui_SliderScalarEx("bounces", c.ImGuiDataType_U32, &config.bounce_limit, &bounce_limit_min, &bounce_limit_max, "%u", c.ImGuiSliderFlags_AlwaysClamp);
        }
        switch (renderer.mode) {
            .neural => |*mode| c.ImGui_TextDisabled("%s", mode.network.description.ptr),
            else => {},
        }
    }
    if (c.ImGui_CollapsingHeader("world", c.ImGuiTreeNodeFlags_DefaultOpen)) {
        _ = c.ImGui_Checkbox("clouds", &config.clouds);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox("clouds only", &config.clouds_only);
        if (config.mode == .raymarch) {
            c.ImGui_SameLine();
            _ = c.ImGui_Checkbox("cloud shadow map", &config.cloud_shadow_map);
        }
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox("tiles", &config.tiles);
        if (renderer.tileStatus()) |status| {
            c.ImGui_TextDisabled("%s", (std.mem.printSentinel(text, "tiles: {d} height, {d} surface, {d} wanted{s}", .{
                status.height, status.surface, status.wanted, if (status.settled) "" else ", streaming",
            }, 0) catch unreachable).ptr);
        }
        sunSliders(&config.sun);
    }
    if (c.ImGui_CollapsingHeader("water", 0)) {
        _ = c.ImGui_Checkbox("wind glint", &config.water.glint);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox("water body", &config.water.body);
        _ = c.ImGui_Checkbox("clear deep water", &config.water.clear_deep_water);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox("calm lakes", &config.water.calm_lakes);
        _ = c.ImGui_Checkbox("ERA5 wind", &config.water.wind_map);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox("roughness texture", &config.water.roughness_texture);
        slider("wind speed m/s", &config.water.wind_speed, ranges.wind_speed);
        slider("ERA5 influence", &config.water.wind_map_influence, ranges.wind_map_influence);
        slider("wind roughness", &config.water.wind_roughness, ranges.wind_roughness);
        slider("glint tail", &config.water.glint_tail, ranges.glint_tail);
        slider("streak length", &config.water.streak_length, ranges.streak_length);
        slider("texture warp", &config.water.texture_warp, ranges.texture_warp);
        slider("texture strength", &config.water.texture_strength, ranges.texture_strength);
        slider("texture size", &config.water.texture_size, ranges.texture_size);
    }
}

/// A picker over `list` with a rescan button; returns whether that was pressed.
fn inputCombo(label: [*:0]const u8, list: *InputList) bool {
    var selected: c_int = @intCast(list.selected);
    if (list.files.len > 0 and c.ImGui_Combo(label, &selected, list.labels)) list.selected = @intCast(selected);
    c.ImGui_SameLine();
    return c.ImGui_Button("rescan");
}

/// Over everything, panel or not, while the cloud bench's probes bake.
pub fn drawBakeProgress(renderer: *const Renderer) void {
    const progress = switch (renderer.mode) {
        .cloud_bench => |*mode| mode.bakeProgress() orelse return,
        else => return,
    };
    var text: [32]u8 = undefined;
    const label = std.mem.printSentinel(&text, "Baking probes {d}%", .{@as(u32, @intFromFloat(progress * 100))}, 0) catch unreachable;
    const draw_list = c.ImGui_GetForegroundDrawList();
    const font_size = 3 * c.ImGui_GetFontSize();
    const origin: c.ImVec2 = .{ .x = 24, .y = 24 };
    const outline = 2;
    for ([_][2]f32{ .{ -1, -1 }, .{ 0, -1 }, .{ 1, -1 }, .{ -1, 0 }, .{ 1, 0 }, .{ -1, 1 }, .{ 0, 1 }, .{ 1, 1 } }) |offset| {
        const pos: c.ImVec2 = .{ .x = origin.x + outline * offset[0], .y = origin.y + outline * offset[1] };
        c.ImDrawList_AddTextImFontPtr(draw_list, c.ImGui_GetFont(), font_size, pos, 0xff000000, label.ptr);
    }
    c.ImDrawList_AddTextImFontPtr(draw_list, c.ImGui_GetFont(), font_size, origin, 0xffffffff, label.ptr);
}

fn drawCloudBench(config: *Config, clouds: *InputList, renderer: *const Renderer, text: *[96]u8, actions: *Actions) void {
    if (!c.ImGui_CollapsingHeader("cloud bench", c.ImGuiTreeNodeFlags_DefaultOpen)) return;
    const bench = &config.cloud_bench;
    actions.rescan_clouds = inputCombo("cloud", clouds);
    switch (renderer.mode) {
        .cloud_bench => |*mode| {
            const counts = mode.sampleCounts();
            c.ImGui_TextDisabled("%s", (std.mem.printSentinel(text, "reference {d} spp, approximation {d} spp, probes {d} walks", .{ counts[0], counts[1], counts[2] }, 0) catch unreachable).ptr);
            if (mode.probes) |probes|
                c.ImGui_TextDisabled("%s", (std.mem.printSentinel(text, "{d} probes, finest spacing {d:.2}", .{ probes.count, probes.spacing[0] }, 0) catch unreachable).ptr);
        },
        else => {},
    }
    enumCombo(scene.CloudBench.View, "view", &bench.view);
    _ = c.ImGui_Checkbox("sun", &bench.sun);
    _ = c.ImGui_Checkbox("sky", &bench.sky);
    slider("sun cos zenith", &config.bench_sun.cos_zenith, ranges.bench_sun_cos_zenith);
    slider("sun azimuth", &config.bench_sun.azimuth_degrees, ranges.bench_sun_azimuth_degrees);
    var extinction_index: c_int = @intCast(closestIndex(&bench_extinctions, bench.extinction));
    var extinction_text: [16]u8 = undefined;
    const extinction_label = std.mem.printSentinel(&extinction_text, "{d}", .{bench_extinctions[@intCast(extinction_index)]}, 0) catch unreachable;
    if (c.ImGui_SliderIntEx("extinction /m", &extinction_index, 0, bench_extinctions.len - 1, extinction_label.ptr, c.ImGuiSliderFlags_AlwaysClamp))
        bench.extinction = bench_extinctions[@intCast(extinction_index)];
    const bounce_limit_min: u32 = @intFromFloat(ranges.bench_bounce_limit.min);
    const bounce_limit_max: u32 = @intFromFloat(ranges.bench_bounce_limit.max);
    _ = c.ImGui_SliderScalarEx("bounces", c.ImGuiDataType_U32, &bench.bounce_limit, &bounce_limit_min, &bounce_limit_max, "%u", c.ImGuiSliderFlags_AlwaysClamp | c.ImGuiSliderFlags_Logarithmic);
    const rounds_min: u32 = 1;
    const rounds_max: u32 = 64;
    _ = c.ImGui_SliderScalarEx("bake rounds per frame", c.ImGuiDataType_U32, &bench.probe_rounds_per_frame, &rounds_min, &rounds_max, "%u", c.ImGuiSliderFlags_AlwaysClamp | c.ImGuiSliderFlags_Logarithmic);
    const probe_walks_min: u32 = @intFromFloat(ranges.bench_probe_walks.min);
    const probe_walks_max: u32 = @intFromFloat(ranges.bench_probe_walks.max);
    _ = c.ImGui_SliderScalarEx("probe walks", c.ImGuiDataType_U32, &bench.probe_walks, &probe_walks_min, &probe_walks_max, "%u", c.ImGuiSliderFlags_AlwaysClamp | c.ImGuiSliderFlags_Logarithmic);
    logSlider("view step max voxels", &bench.view_step_max_voxels, ranges.bench_view_step_max_voxels);
    enumCombo(scene.CloudBench.ProbeLayout, "probe layout", &bench.probe_layout);
    if (bench.probe_layout == .adaptive) {
        const probe_budget_min: u32 = @intFromFloat(ranges.bench_probe_budget.min);
        const probe_budget_max: u32 = @intFromFloat(ranges.bench_probe_budget.max);
        _ = c.ImGui_SliderScalarEx("probe budget", c.ImGuiDataType_U32, &bench.probe_budget, &probe_budget_min, &probe_budget_max, "%u", c.ImGuiSliderFlags_AlwaysClamp | c.ImGuiSliderFlags_Logarithmic);
        slider("probe depth ratio", &bench.probe_depth_ratio, ranges.bench_probe_depth_ratio);
    }
    enumCombo(scene.CloudBench.ProbeBake, "probe bake", &bench.probe_bake);
    switch (bench.probe_bake) {
        .walks => enumCombo(scene.CloudBench.SunEstimate, "probe sun estimate", &bench.probe_sun_estimate),
        .photons => enumCombo(scene.CloudBench.PhotonDeposit, "photon deposit", &bench.photon_deposit),
    }
    enumCombo(scene.CloudBench.SunOrder, "probe sun order", &bench.probe_sun_order);
    enumCombo(scene.CloudBench.SunFit, "probe sun fit", &bench.probe_sun_fit);
    enumCombo(scene.CloudBench.SkyOrder, "probe sky order", &bench.probe_sky_order);
    enumCombo(scene.CloudBench.FirstScatter, "first scatter", &bench.first_scatter);
    enumCombo(scene.CloudBench.Closure, "probe closure", &bench.probe_closure);
    actions.reset_probes = c.ImGui_Button("delete probe cache and rebake");
    if (c.ImGui_Button("reset cloud bench")) bench.* = .{};
}

/// Per metre. Each extinction bakes probes of its own, so the slider keeps to a few that
/// their caches can be reused for.
const bench_extinctions = [_]f64{ 0.05, 0.1, 0.2, 0.4 };

fn closestIndex(values: []const f64, value: f64) usize {
    var closest: usize = 0;
    for (values, 0..) |candidate, i| {
        if (@abs(candidate - value) < @abs(values[closest] - value)) closest = i;
    }
    return closest;
}

fn sunSliders(sun: *View.Sun) void {
    slider("sun angle", &sun.angle_degrees, ranges.sun_angle_degrees);
    slider("sun path rotation", &sun.path_rotation_degrees, ranges.sun_path_rotation_degrees);
}

fn slider(label: [*:0]const u8, value: *f64, range: Range) void {
    _ = c.ImGui_SliderScalarEx(label, c.ImGuiDataType_Double, value, &range.min, &range.max, "%.3f", c.ImGuiSliderFlags_AlwaysClamp);
}

/// For values that span orders of magnitude.
fn logSlider(label: [*:0]const u8, value: *f64, range: Range) void {
    _ = c.ImGui_SliderScalarEx(label, c.ImGuiDataType_Double, value, &range.min, &range.max, "%.4g", c.ImGuiSliderFlags_AlwaysClamp | c.ImGuiSliderFlags_Logarithmic);
}

/// A combo over every tag of `T`, labelled by tag name.
fn enumCombo(comptime T: type, label: [*:0]const u8, value: *T) void {
    const names = comptime names: {
        var names: [std.enums.values(T).len][*:0]const u8 = undefined;
        for (&names, std.enums.values(T)) |*name, tag| name.* = @tagName(tag);
        break :names names;
    };
    var index: c_int = @intCast(@backingInt(value.*));
    if (c.ImGui_ComboChar(label, &index, &names, names.len)) value.* = @fromBackingInt(@as(@typeInfo(T).@"enum".tag_type, @intCast(index)));
}

const testing = std.testing;

test "defaults lie inside the slider ranges, and the ranges inside what View accepts" {
    const defaults: Config = .{};
    const values = .{
        .sun_angle_degrees = defaults.sun.angle_degrees,
        .sun_path_rotation_degrees = defaults.sun.path_rotation_degrees,
        .bench_sun_cos_zenith = defaults.bench_sun.cos_zenith,
        .bench_sun_azimuth_degrees = defaults.bench_sun.azimuth_degrees,
        .vertical_fov_degrees = defaults.camera.vertical_fov_degrees,
        .aspect_scale = defaults.camera.aspect_scale,
        .exposure = defaults.exposure,
        .bounce_limit = @as(f64, @floatFromInt(defaults.bounce_limit)),
        .wind_speed = defaults.water.wind_speed,
        .texture_strength = defaults.water.texture_strength,
        .texture_size = defaults.water.texture_size,
        .wind_map_influence = defaults.water.wind_map_influence,
        .glint_tail = defaults.water.glint_tail,
        .streak_length = defaults.water.streak_length,
        .texture_warp = defaults.water.texture_warp,
        .wind_roughness = defaults.water.wind_roughness,
        .cloud_rate = defaults.cloud_scattering.rate_away,
        .cloud_strength = defaults.cloud_scattering.strength,
        .cloud_relaxed_depth = defaults.cloud_scattering.relaxed_depth,
        .cloud_phase_g = defaults.cloud_scattering.phase_g,
        .cloud_profile_strength = defaults.cloud_scattering.profile_strength,
        .cloud_profile_depth = defaults.cloud_scattering.profile_depth,
        .bench_bounce_limit = @as(f64, @floatFromInt(defaults.cloud_bench.bounce_limit)),
        .bench_probe_walks = @as(f64, @floatFromInt(defaults.cloud_bench.probe_walks)),
        .bench_probe_budget = @as(f64, @floatFromInt(defaults.cloud_bench.probe_budget)),
        .bench_probe_depth_ratio = defaults.cloud_bench.probe_depth_ratio,
        .bench_view_step_max_voxels = defaults.cloud_bench.view_step_max_voxels,
        .saturation = defaults.saturation,
        .contrast = defaults.contrast,
        .gamma = defaults.gamma,
    };
    inline for (comptime std.meta.declarations(ranges)) |name| {
        const range: Range = @field(ranges, name);
        const value: f64 = @field(values, name);
        try testing.expect(range.min < range.max);
        try testing.expect(value >= range.min and value <= range.max);
    }
    try testing.expect(ranges.vertical_fov_degrees.max <= View.vertical_fov_degrees_max);
    try testing.expect(ranges.aspect_scale.min > 0);
    try testing.expect(std.mem.indexOfScalar(f64, &bench_extinctions, defaults.cloud_bench.extinction) != null);
}

test "readout converts the frame rate to frame time and pixel samples" {
    const readout: Readout = .init(50, .{ 1920, 1080 }, 7);
    try testing.expectEqual(7, readout.sample_count);
    try testing.expectApproxEqAbs(20, readout.frame_ms, 1e-12);
    try testing.expectApproxEqAbs(50 * 1920 * 1080, readout.samples_per_second, 1e-6);
}

test "labels join into one NUL-separated list" {
    const labels = try joinLabels(testing.allocator, &.{ "Neutral", "FP2900ZG" });
    defer testing.allocator.free(labels);
    try testing.expectEqualStrings("Neutral\x00FP2900ZG\x00", labels);
}
