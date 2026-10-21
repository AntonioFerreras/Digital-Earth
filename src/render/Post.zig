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
