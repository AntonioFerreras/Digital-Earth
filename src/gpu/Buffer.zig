//! A GPU buffer owned by the caller; destroy it before its device.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const Buffer = @This();

handle: *c.en_buffer,
size: u64,

pub const Usage = packed struct(u32) {
    constant: bool = false,
    shader_resource: bool = false,
    unordered_access: bool = false,
    copy_source: bool = false,
    copy_destination: bool = false,
    _: u27 = 0,
};

pub const Memory = enum(c.en_memory) { device, upload, readback };

comptime {
    gpu.assertMirrorsC(Usage, "EN_BUFFER_");
    gpu.assertMirrorsC(Memory, "EN_MEMORY_");
}

pub const Desc = struct {
    size: u64,
    /// Stride of a structured buffer; 0 for raw and constant buffers.
    element_size: u32,
    usage: Usage,
    memory: Memory,
    label: [*:0]const u8,
};

/// `initial_data`, when given, must hold exactly `desc.size` bytes.
pub fn init(device: gpu.Device, desc: Desc, initial_data: ?[]const u8) gpu.Error!Buffer {
    assert(desc.size > 0);
    if (initial_data) |data| assert(data.len == desc.size);
    const native_desc: c.en_buffer_desc = .{
        .size = desc.size,
        .element_size = desc.element_size,
        .usage = @bitCast(desc.usage),
        .memory = @backingInt(desc.memory),
        .label = desc.label,
    };
    var handle: ?*c.en_buffer = null;
    const data_ptr = if (initial_data) |data| data.ptr else null;
    try gpu.check(c.en_buffer_create(device.handle, &native_desc, data_ptr, &handle), "en_buffer_create");
    return .{ .handle = handle.?, .size = desc.size };
}

pub fn deinit(buffer: Buffer) void {
    c.en_buffer_destroy(buffer.handle);
}

pub fn upload(buffer: Buffer, device: gpu.Device, offset: u64, data: []const u8) gpu.Error!void {
    assert(offset + data.len <= buffer.size);
    try gpu.check(
        c.en_buffer_upload(device.handle, buffer.handle, offset, data.ptr, data.len),
        "en_buffer_upload",
    );
}

/// Blocks until the GPU has written `target.len` bytes starting at `offset`.
pub fn read(buffer: Buffer, device: gpu.Device, offset: u64, target: []u8) gpu.Error!void {
    assert(offset + target.len <= buffer.size);
    try gpu.check(
        c.en_buffer_read(device.handle, buffer.handle, offset, target.ptr, target.len),
        "en_buffer_read",
    );
}
