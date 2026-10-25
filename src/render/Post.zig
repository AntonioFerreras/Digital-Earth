//! The display pass of shaders/post/display.slang: accumulated samples to the rgba8
//! image that is presented or saved.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const uniforms = @import("uniforms.zig");
const GpuLuts = @import("GpuLuts.zig");

const Post = @This();

kernel: gpu.Kernel,
params: std.EnumArray(Param, gpu.Kernel.Param),

const Param = enum { post, accum, camera_responses, linear_clamp, image };

/// The user-facing controls of the display pass; none of them resets accumulation.
pub const Tone = struct {
    /// Stops on the original renderer's scale.
    exposure: f64,
    /// Oklab chroma scale ahead of OpenDRT; 1 keeps the upstream look.
    saturation: f64,
    /// Offset to the tonescale preset's contrast power; 0 keeps the preset.
    contrast: f64,
    tonescale: TonescalePreset,
    /// Off shows the exposed linear colour clipped to [0, 1] instead of OpenDRT's.
    tone_map: bool = true,
    /// One of `GpuLuts.camera_response_names`; "Neutral" is the identity response.
    camera_response: []const u8,
    gamma: f64,
};

/// OpenDRT's tonescale presets, in the order of `TONESCALE_PRESETS` in
/// shaders/post/opendrt.slang.
pub const TonescalePreset = enum(u32) {
    high_contrast,
    low_contrast,
    aces_1,
    aces_2,
    marvelous_tonescape,
    arriba_tonecall,
    dagrinchi_tone,
};

pub const RecordError = gpu.Error || error{UnknownCameraResponse};

/// The exposure for a sun of `cos_zenith` at the camera: `auto_exposure_table` by a
/// monotone cubic, smooth and without overshoot between its points, held past its ends.
pub fn autoExposure(cos_zenith: f64) f64 {
    const table = auto_exposure_table;
    if (cos_zenith <= table[0][0]) return table[0][1];
    if (cos_zenith >= table[table.len - 1][0]) return table[table.len - 1][1];
    var i: usize = 0;
    while (cos_zenith > table[i + 1][0]) i += 1;
    const h = table[i + 1][0] - table[i][0];
    const t = (cos_zenith - table[i][0]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * table[i][1] + (t3 - 2 * t2 + t) * h * auto_exposure_slopes[i] +
        (3 * t2 - 2 * t3) * table[i + 1][1] + (t3 - t2) * h * auto_exposure_slopes[i + 1];
}

/// (cos zenith, stops), cos zenith rising, from python/validation/auto_exposure.py: the
/// sky shows as bright as under a sun 0.3 high at 3 stops all the way down to sunset,
/// then dims smoothly to 10 stops at the lowest sun the viewer and validation allow.
pub const auto_exposure_table = [_][2]f64{
    .{ -0.07, 10.0000 },
    .{ -0.06, 9.0336 },
    .{ -0.05, 8.2263 },
    .{ -0.04, 7.4908 },
    .{ -0.03, 6.8790 },
    .{ -0.02, 6.3434 },
    .{ -0.01, 5.8588 },
    .{ 0, 5.4114 },
    .{ 0.02, 4.6601 },
    .{ 0.04, 4.1727 },
    .{ 0.06, 3.8436 },
    .{ 0.08, 3.6158 },
    .{ 0.1, 3.4600 },
    .{ 0.12, 3.3524 },
    .{ 0.14, 3.2730 },
    .{ 0.16, 3.2066 },
    .{ 0.18, 3.1550 },
    .{ 0.2, 3.1136 },
    .{ 0.22, 3.0790 },
    .{ 0.24, 3.0508 },
    .{ 0.26, 3.0314 },
    .{ 0.28, 3.0143 },
    .{ 0.3, 3.0000 },
    .{ 0.35, 2.9721 },
    .{ 0.4, 2.9557 },
    .{ 0.45, 2.9395 },
    .{ 0.5, 2.9286 },
    .{ 0.55, 2.9210 },
    .{ 0.6, 2.9128 },
    .{ 0.65, 2.9079 },
    .{ 0.7, 2.9017 },
    .{ 0.75, 2.9020 },
    .{ 0.8, 2.9086 },
    .{ 0.85, 2.9238 },
    .{ 0.9, 2.9941 },
    .{ 0.95, 3.0503 },
    .{ 1, 3.0843 },
};

/// The table's slopes by PCHIP (Fritsch and Butland): zero at its extrema, a weighted
/// harmonic mean of the neighbouring secants elsewhere, which keeps each span monotone.
const auto_exposure_slopes = slopes: {
    const table = auto_exposure_table;
    var secants: [table.len - 1]f64 = undefined;
    var widths: [table.len - 1]f64 = undefined;
    for (&secants, &widths, table[0 .. table.len - 1], table[1..]) |*secant, *width, low, high| {
        width.* = high[0] - low[0];
        secant.* = (high[1] - low[1]) / width.*;
    }
    var result: [table.len]f64 = undefined;
    result[0] = secants[0];
    result[table.len - 1] = secants[secants.len - 1];
    for (1..table.len - 1) |k| {
        const before = secants[k - 1];
        const after = secants[k];
        if (before * after <= 0) {
            result[k] = 0;
        } else {
            const w1 = 2 * widths[k] + widths[k - 1];
            const w2 = widths[k] + 2 * widths[k - 1];
            result[k] = (w1 + w2) / (w1 / before + w2 / after);
        }
    }
    break :slopes result;
};

pub fn init(device: gpu.Device) gpu.Kernel.LayoutError!Post {
    const kernel: gpu.Kernel = try .init(device, "post/display", "display_main", &.{});
    errdefer kernel.deinit();
    try kernel.checkLayout(uniforms.PostUniforms, "post");
    var params: std.EnumArray(Param, gpu.Kernel.Param) = undefined;
    inline for (comptime std.enums.values(Param)) |name| params.set(name, try kernel.param(@tagName(name)));
    return .{ .kernel = kernel, .params = params };
}

pub fn deinit(post: Post) void {
    post.kernel.deinit();
}

/// Records the display of the mean of `sample_count` samples in `accum` into `image`
/// (rgba8, same size).
pub fn record(
    post: Post,
    command_list: gpu.CommandList,
    luts: *const GpuLuts,
    accum: gpu.Texture,
    image: gpu.Texture,
    tone: Tone,
    sample_count: u32,
) RecordError!void {
    assert(sample_count > 0);
    assert(accum.desc.width == image.desc.width);
    assert(accum.desc.height == image.desc.height);
    assert(image.desc.format == .rgba8_unorm);
    const camera_response_index = try luts.cameraResponseIndex(tone.camera_response);
    const camera_response_count = luts.camera_responses.desc.height;

    const post_uniforms: uniforms.PostUniforms = .{
        .tone = .{ @floatCast(tone.exposure), @floatCast(tone.gamma), @floatCast(tone.saturation), @floatCast(tone.contrast) },
        .state = .{ sample_count, camera_response_index, camera_response_count, @backingInt(tone.tonescale) },
        .options = .{ @intFromBool(tone.tone_map), 0, 0, 0 },
    };
    const bindings = [_]gpu.CommandList.Binding{
        .{ .param = post.params.get(.post), .resource = .{ .data = std.mem.asBytes(&post_uniforms) } },
        .{ .param = post.params.get(.accum), .resource = .{ .texture = accum } },
        .{ .param = post.params.get(.camera_responses), .resource = .{ .texture = luts.camera_responses } },
        .{ .param = post.params.get(.linear_clamp), .resource = .{ .sampler = luts.linear_clamp } },
        .{ .param = post.params.get(.image), .resource = .{ .texture = image } },
    };
    try command_list.dispatch(post.kernel, &bindings, post.kernel.groupCount(.{ image.desc.width, image.desc.height, 1 }));
}

const testing = std.testing;

test "auto exposure passes through its table, stays within each span, and holds past its ends" {
    for (auto_exposure_table) |point| try testing.expectApproxEqAbs(point[1], autoExposure(point[0]), 1e-12);
    for (auto_exposure_table[0 .. auto_exposure_table.len - 1], auto_exposure_table[1..]) |low, high| {
        for (1..10) |step| {
            const exposure = autoExposure(std.math.lerp(low[0], high[0], @as(f64, @floatFromInt(step)) / 10));
            try testing.expect(exposure >= @min(low[1], high[1]) - 1e-12 and exposure <= @max(low[1], high[1]) + 1e-12);
        }
    }
    try testing.expectEqual(auto_exposure_table[0][1], autoExposure(-1));
    try testing.expectEqual(auto_exposure_table[auto_exposure_table.len - 1][1], autoExposure(2));
}
