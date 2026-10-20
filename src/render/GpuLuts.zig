//! The lookup tables of `assets.Luts` on the GPU, uploaded once, the column densities
//! of the atmosphere every preset shares, baked once from them, and the samplers every
//! kernel shares. Buffers keep the host layouts documented in `assets.Luts`.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const Luts = @import("../assets/Luts.zig");
const scene = @import("scene.zig");

const GpuLuts = @This();

cie: gpu.Buffer,
srgb_to_spectrum: gpu.Buffer,
ozone_cross_section: gpu.Buffer,
ozone_profile: gpu.Buffer,
/// The column-density table of shaders/atmosphere/scattering_luts.slang.
column_density: gpu.Texture,
/// rgba32f, `Luts.crf_texel_count` wide, one camera response per row in
/// `camera_response_names` order.
camera_responses: gpu.Texture,
/// A copy of `Luts.crf_names`.
camera_response_names: []const []const u8,
linear_clamp: gpu.Sampler,
/// For Earth maps: longitude wraps, latitude clamps so the poles do not bleed into each other.
linear_repeat_u: gpu.Sampler,

/// The host tables may be freed once this returns.
/// COLUMN_DENSITY_LUT_SIZE in scattering_luts.slang.
pub const column_density_size = [2]u32{ 256, 256 };

pub fn init(gpa: std.mem.Allocator, device: gpu.Device, luts: *const Luts) (gpu.Error || error{OutOfMemory})!GpuLuts {
    const cie = try initBuffer(device, luts.cie, "CIE");
    errdefer cie.deinit();
    const srgb_to_spectrum = try initBuffer(device, luts.srgb_to_spectrum, "srgb2spec");
    errdefer srgb_to_spectrum.deinit();
    const ozone_cross_section = try initBuffer(device, luts.ozone_cross_section, "ozone cross section");
    errdefer ozone_cross_section.deinit();
    const ozone_profile = try initBuffer(device, luts.ozone_profile, "ozone profile");
    errdefer ozone_profile.deinit();
    const column_density = try bakeColumnDensity(device, ozone_cross_section, ozone_profile);
    errdefer column_density.deinit();

    const crf_count: u32 = @intCast(luts.crf_names.len);
    assert(luts.crf.len == crf_count * Luts.crf_texel_count * 4);
    const camera_responses: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = Luts.crf_texel_count,
        .height = crf_count,
        .depth_or_layers = 1,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .label = "camera responses",
    });
    errdefer camera_responses.deinit();
    try camera_responses.upload(device, .{
        .x = 0,
        .y = 0,
        .z = 0,
        .width = Luts.crf_texel_count,
        .height = crf_count,
        .depth = 1,
    }, std.mem.sliceAsBytes(luts.crf), Luts.crf_texel_count * 4 * @sizeOf(f32));

    const camera_response_names = try gpa.alloc([]const u8, crf_count);
    var names_copied: usize = 0;
    errdefer {
        for (camera_response_names[0..names_copied]) |name| gpa.free(name);
        gpa.free(camera_response_names);
    }
    for (camera_response_names, luts.crf_names) |*copy, name| {
        copy.* = try gpa.dupe(u8, name);
        names_copied += 1;
    }

    const linear_clamp: gpu.Sampler = try .init(device, .{
        .filter = .linear,
        .address_u = .clamp,
        .address_v = .clamp,
        .address_w = .clamp,
    });
    errdefer linear_clamp.deinit();
    const linear_repeat_u: gpu.Sampler = try .init(device, .{
        .filter = .linear,
        .address_u = .repeat,
        .address_v = .clamp,
        .address_w = .clamp,
    });
    return .{
        .cie = cie,
        .srgb_to_spectrum = srgb_to_spectrum,
        .ozone_cross_section = ozone_cross_section,
        .ozone_profile = ozone_profile,
        .column_density = column_density,
        .camera_responses = camera_responses,
        .camera_response_names = camera_response_names,
        .linear_clamp = linear_clamp,
        .linear_repeat_u = linear_repeat_u,
    };
}

pub fn deinit(gpu_luts: *GpuLuts, gpa: std.mem.Allocator) void {
    for (gpu_luts.camera_response_names) |name| gpa.free(name);
    gpa.free(gpu_luts.camera_response_names);
    gpu_luts.linear_repeat_u.deinit();
    gpu_luts.linear_clamp.deinit();
    gpu_luts.camera_responses.deinit();
    gpu_luts.column_density.deinit();
    gpu_luts.ozone_profile.deinit();
    gpu_luts.ozone_cross_section.deinit();
    gpu_luts.srgb_to_spectrum.deinit();
    gpu_luts.cie.deinit();
    gpu_luts.* = undefined;
}

/// Blocking: waits for the bake.
fn bakeColumnDensity(device: gpu.Device, ozone_cross_section: gpu.Buffer, ozone_profile: gpu.Buffer) gpu.Error!gpu.Texture {
    // The shader filters the column densities itself, in f32.
    const texture: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = column_density_size[0],
        .height = column_density_size[1],
        .depth_or_layers = 1,
        // Copies out serve the tests.
        .usage = .{ .shader_resource = true, .unordered_access = true, .copy_source = true },
        .label = "column density LUT",
    });
    errdefer texture.deinit();
    const kernel: gpu.Kernel = try .init(device, "modes/raymarch_lut", "bake_column_density", &.{});
    defer kernel.deinit();
    const atmosphere = scene.atmosphereUniforms();
    const command_list: gpu.CommandList = try .begin(device);
    const recorded = command_list.dispatch(kernel, &.{
        .{ .param = try kernel.param("scene"), .resource = .{ .data = std.mem.asBytes(&atmosphere) } },
        .{ .param = try kernel.param("ozone_cross_section"), .resource = .{ .buffer = ozone_cross_section } },
        .{ .param = try kernel.param("ozone_profile"), .resource = .{ .buffer = ozone_profile } },
        .{ .param = try kernel.param("column_density_target"), .resource = .{ .texture = texture } },
    }, kernel.groupCount(.{ column_density_size[0], column_density_size[1], 1 }));
    try command_list.submit();
    try recorded;
    try device.waitIdle();
    return texture;
}

/// The row of `camera_responses` holding the response named `name`; the only place a
/// camera response name becomes an index.
pub fn cameraResponseIndex(gpu_luts: *const GpuLuts, name: []const u8) error{UnknownCameraResponse}!u32 {
    for (gpu_luts.camera_response_names, 0..) |candidate, index| {
        if (std.mem.eql(u8, candidate, name)) return @intCast(index);
    }
    const log = std.log.scoped(.render);
    log.err("unknown camera response '{s}'; available:", .{name});
    for (gpu_luts.camera_response_names) |candidate| log.err("  {s}", .{candidate});
    return error.UnknownCameraResponse;
}

/// A read-only `StructuredBuffer<float>` holding `values`.
pub fn initBuffer(device: gpu.Device, values: []const f32, label: [*:0]const u8) gpu.Error!gpu.Buffer {
    assert(values.len > 0);
    return .init(device, .{
        .size = values.len * @sizeOf(f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = label,
    }, std.mem.sliceAsBytes(values));
}
