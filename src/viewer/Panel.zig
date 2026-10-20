//! The ImGui parameter panel. It edits the viewer's `Config` in place and reports the
//! buttons pressed; whether an edit restarts accumulation is the renderer's decision.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const EarthMaps = @import("../assets/EarthMaps.zig");
const Renderer = @import("../render/Renderer.zig");
const View = @import("../render/View.zig");
const scene = @import("../render/scene.zig");
const Post = @import("../render/Post.zig");
const Config = @import("Config.zig");
const NetworkList = @import("NetworkList.zig");

const Panel = @This();

/// Camera response names separated and terminated by NUL, as `ImGui_Combo` takes them.
camera_response_names: [:0]u8,

pub const Range = struct { min: f64, max: f64 };

/// Slider limits, also enforced on Ctrl+click input. Loaded configs may lie outside
/// them; a slider shows such a value until it is dragged.
pub const ranges = struct {
    pub const sun_angle_degrees: Range = .{ .min = 0, .max = 360 };
    pub const sun_path_rotation_degrees: Range = .{ .min = -105, .max = 105 };
    /// The original slider stopped at tan(half angle) = pi/2, about 115 degrees.
    pub const vertical_fov_degrees: Range = .{ .min = View.vertical_fov_degrees_min, .max = 115 };
    pub const aspect_scale: Range = .{ .min = 0.75, .max = 1.25 };
    pub const exposure: Range = .{ .min = -1, .max = 10 };
    /// Greyscale up to a strong chroma boost; 1 keeps OpenDRT's look.
    pub const saturation: Range = .{ .min = 0, .max = 3.6 };
    /// Keeps every preset's contrast power at or above the shader's floor of 0.25.
    pub const contrast: Range = .{ .min = -0.75, .max = 1 };
    pub const gamma: Range = .{ .min = 0.45, .max = 2.2 };
};

/// Buttons pressed this frame.
pub const Actions = struct {
    restart: bool = false,
    save_config: bool = false,
    load_config: bool = false,
    screenshot: bool = false,
    rescan_networks: bool = false,
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

pub fn init(gpa: std.mem.Allocator, camera_response_names: []const []const u8) error{OutOfMemory}!Panel {
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(gpa);
    for (camera_response_names) |name| {
        assert(std.mem.findScalar(u8, name, 0) == null);
        try joined.appendSlice(gpa, name);
        try joined.append(gpa, 0);
    }
    return .{ .camera_response_names = try joined.toOwnedSliceSentinel(gpa, 0) };
}

pub fn deinit(panel: *Panel, gpa: std.mem.Allocator) void {
    gpa.free(panel.camera_response_names);
    panel.* = undefined;
}

/// `config.camera_response` must name a loaded camera response; `gpa` owns it. The
/// network picker edits `networks.selected`.
pub fn draw(panel: Panel, gpa: std.mem.Allocator, config: *Config, networks: *NetworkList, renderer: *const Renderer) Actions {
    var actions: Actions = .{};
    c.ImGui_SetNextWindowPos(.{ .x = 24, .y = 24 }, c.ImGuiCond_FirstUseEver);
    defer c.ImGui_End();
    if (!c.ImGui_Begin("Digital Earth", null, c.ImGuiWindowFlags_AlwaysAutoResize)) return actions;

    const readout: Readout = .init(c.ImGui_GetIO().*.Framerate, renderer.size(), renderer.sampleCount());
    var text: [96]u8 = undefined;
    c.ImGui_TextUnformatted(std.mem.printSentinel(&text, "{d} spp   {d:.2} ms/frame   {d:.1} Msamples/s", .{
        readout.sample_count, readout.frame_ms, readout.samples_per_second / 1e6,
    }, 0) catch unreachable);

    c.ImGui_SeparatorText("render");
    enumCombo(Renderer.ModeKind, "mode", &config.mode);
    enumCombo(scene.Preset, "scene preset", &config.preset);
    enumCombo(EarthMaps.Quality, "texture quality", &config.texture_quality);
    var selected: c_int = @intCast(networks.selected);
    if (networks.files.len > 0 and c.ImGui_Combo("network", &selected, networks.labels)) networks.selected = @intCast(selected);
    c.ImGui_SameLine();
    actions.rescan_networks = c.ImGui_Button("rescan");
    switch (renderer.mode) {
        .neural => |*mode| c.ImGui_TextDisabled("%s", mode.network.description.ptr),
        else => {},
    }

    c.ImGui_SeparatorText("world");
    _ = c.ImGui_Checkbox("clouds", &config.clouds);
    slider("sun angle", &config.sun.angle_degrees, ranges.sun_angle_degrees);
    slider("sun path rotation", &config.sun.path_rotation_degrees, ranges.sun_path_rotation_degrees);

    c.ImGui_SeparatorText("camera");
    const altitude_m = config.camera.altitude();
    c.ImGui_TextUnformatted(if (altitude_m < 10e3)
        std.mem.printSentinel(&text, "elevation {d:.0} m", .{altitude_m}, 0) catch unreachable
    else
        std.mem.printSentinel(&text, "elevation {d:.2} km", .{altitude_m / 1e3}, 0) catch unreachable);
    slider("vertical fov", &config.camera.vertical_fov_degrees, ranges.vertical_fov_degrees);
    slider("aspect scale", &config.camera.aspect_scale, ranges.aspect_scale);
    slider("exposure", &config.exposure, ranges.exposure);
    slider("saturation", &config.saturation, ranges.saturation);
    slider("contrast", &config.contrast, ranges.contrast);
    enumCombo(Post.TonescalePreset, "tonescale", &config.tonescale);
    const luts = &renderer.shared.luts;
    var camera_response: c_int = @intCast(luts.cameraResponseIndex(config.camera_response) catch unreachable);
    if (c.ImGui_Combo("camera response", &camera_response, panel.camera_response_names)) {
        const name = luts.camera_response_names[@intCast(camera_response)];
        config.setCameraResponse(gpa, name) catch std.log.scoped(.viewer).err("out of memory selecting {s}", .{name});
    }
    slider("gamma", &config.gamma, ranges.gamma);

    c.ImGui_Separator();
    actions.restart = c.ImGui_Button("reset accumulation");
    c.ImGui_SameLine();
    actions.save_config = c.ImGui_Button("save config");
    c.ImGui_SameLine();
    actions.load_config = c.ImGui_Button("load config");
    c.ImGui_SameLine();
    actions.screenshot = c.ImGui_Button("screenshot");
    c.ImGui_TextDisabled("G panel   I/O save/load config   P screenshot");
    return actions;
}

fn slider(label: [*:0]const u8, value: *f64, range: Range) void {
    _ = c.ImGui_SliderScalarEx(label, c.ImGuiDataType_Double, value, &range.min, &range.max, "%.3f", c.ImGuiSliderFlags_AlwaysClamp);
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
        .vertical_fov_degrees = defaults.camera.vertical_fov_degrees,
        .aspect_scale = defaults.camera.aspect_scale,
        .exposure = defaults.exposure,
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
}

test "readout converts the frame rate to frame time and pixel samples" {
    const readout: Readout = .init(50, .{ 1920, 1080 }, 7);
    try testing.expectEqual(7, readout.sample_count);
    try testing.expectApproxEqAbs(20, readout.frame_ms, 1e-12);
    try testing.expectApproxEqAbs(50 * 1920 * 1080, readout.samples_per_second, 1e-6);
}

test "camera response names join into one NUL-separated list" {
    var panel: Panel = try .init(testing.allocator, &.{ "Neutral", "FP2900ZG" });
    defer panel.deinit(testing.allocator);
    try testing.expectEqualStrings("Neutral\x00FP2900ZG\x00", panel.camera_response_names);
}
