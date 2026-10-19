//! A 2D, 2D-array, or 3D texture owned by the caller; destroy it before its device.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const Texture = @This();

handle: *c.en_texture,
desc: Desc,

pub const Kind = enum(c.en_texture_kind) { @"2d", @"2d_array", @"3d" };

pub const Format = enum(c.en_format) {
    r8_unorm,
    rgba8_unorm,
    r16_float,
    rgba16_float,
    r32_float,
    rgba32_float,

    pub fn texelSize(format: Format) u32 {
        return switch (format) {
            .r8_unorm => 1,
            .rgba8_unorm, .r32_float => 4,
            .r16_float => 2,
            .rgba16_float => 8,
            .rgba32_float => 16,
        };
    }
};

pub const Usage = packed struct(u32) {
    shader_resource: bool = false,
    unordered_access: bool = false,
    copy_source: bool = false,
    copy_destination: bool = false,
    _: u28 = 0,
};

comptime {
    gpu.assertMirrorsC(Kind, "EN_TEXTURE_");
    gpu.assertMirrorsC(Format, "EN_FORMAT_");
    gpu.assertMirrorsC(Usage, "EN_TEXTURE_");
}

pub const Desc = struct {
    kind: Kind,
    format: Format,
    width: u32,
    height: u32,
    /// Depth for 3D textures, layer count for arrays, 1 for plain 2D.
    depth_or_layers: u32,
    usage: Usage,
    label: [*:0]const u8,
};

/// `z` is the depth offset for 3D textures and the first layer for arrays.
pub const Region = c.en_texture_region;

pub fn init(device: gpu.Device, desc: Desc) gpu.Error!Texture {
    assert(desc.width > 0);
    assert(desc.height > 0);
    assert(desc.depth_or_layers > 0);
    if (desc.kind == .@"2d") assert(desc.depth_or_layers == 1);
    const native_desc: c.en_texture_desc = .{
        .kind = @backingInt(desc.kind),
        .format = @backingInt(desc.format),
        .width = desc.width,
        .height = desc.height,
        .depth_or_layers = desc.depth_or_layers,
        .usage = @bitCast(desc.usage),
        .label = desc.label,
    };
    var handle: ?*c.en_texture = null;
    try gpu.check(c.en_texture_create(device.handle, &native_desc, &handle), "en_texture_create");
    return .{ .handle = handle.?, .desc = desc };
}

pub fn deinit(texture: Texture) void {
    c.en_texture_destroy(texture.handle);
}

/// `data` holds `region.depth` slices of `region.height` rows, `row_pitch` bytes apart.
pub fn upload(
    texture: Texture,
    device: gpu.Device,
    region: Region,
    data: []const u8,
    row_pitch: u64,
) gpu.Error!void {
    assert(region.x + region.width <= texture.desc.width);
    assert(region.y + region.height <= texture.desc.height);
    assert(region.z + region.depth <= texture.desc.depth_or_layers);
    assert(row_pitch >= @as(u64, region.width) * texture.desc.format.texelSize());
    assert(data.len >= row_pitch * region.height * region.depth);
    try gpu.check(
        c.en_texture_upload(device.handle, texture.handle, &region, data.ptr, row_pitch),
        "en_texture_upload",
    );
}

/// Blocks, then writes one layer (or 3D slice) tightly packed into `target`.
pub fn read(texture: Texture, device: gpu.Device, layer: u32, target: []u8) gpu.Error!void {
    assert(layer < texture.desc.depth_or_layers);
    const layer_size = @as(u64, texture.desc.width) * texture.desc.height * texture.desc.format.texelSize();
    assert(target.len == layer_size);
    try gpu.check(
        c.en_texture_read(device.handle, texture.handle, layer, target.ptr, target.len),
        "en_texture_read",
    );
}
