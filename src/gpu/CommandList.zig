//! A recording of compute dispatches and copies. `submit` consumes it, also on failure.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const CommandList = @This();

handle: *c.en_cmd,

/// A bounded stack array holds the converted bindings of one dispatch.
pub const bindings_max = 48;

pub const Binding = struct {
    param: gpu.Kernel.Param,
    resource: Resource,

    pub const Resource = union(enum) {
        /// Uniform bytes, copied at dispatch; typically `std.mem.asBytes(&uniforms)`.
        data: []const u8,
        buffer: gpu.Buffer,
        texture: gpu.Texture,
        sampler: gpu.Sampler,
    };

    fn toNative(binding: Binding) c.en_binding {
        var native: c.en_binding = .{ .param = binding.param.location };
        switch (binding.resource) {
            .data => |bytes| {
                assert(binding.param.kind == .data);
                assert(bytes.len == binding.param.size);
                native.kind = c.EN_BIND_DATA;
                native.data = bytes.ptr;
                native.size = bytes.len;
            },
            .buffer => |buffer| {
                assert(binding.param.kind == .buffer or binding.param.kind == .rw_buffer);
                native.kind = c.EN_BIND_BUFFER;
                native.buffer = buffer.handle;
            },
            .texture => |texture| {
                assert(binding.param.kind == .texture or binding.param.kind == .rw_texture);
                native.kind = c.EN_BIND_TEXTURE;
                native.texture = texture.handle;
            },
            .sampler => |sampler| {
                assert(binding.param.kind == .sampler);
                native.kind = c.EN_BIND_SAMPLER;
                native.sampler = sampler.handle;
            },
        }
        return native;
    }
};

pub fn begin(device: gpu.Device) gpu.Error!CommandList {
    var handle: ?*c.en_cmd = null;
    try gpu.check(c.en_cmd_begin(device.handle, &handle), "en_cmd_begin");
    return .{ .handle = handle.? };
}

/// Records one dispatch; a UAV barrier separates it from the previous one. The C API has
/// no discard, so even after a failure the list is released only by `submit`.
pub fn dispatch(
    command_list: CommandList,
    kernel: gpu.Kernel,
    bindings: []const Binding,
    groups: [3]u32,
) gpu.Error!void {
    assert(bindings.len <= bindings_max);
    for (groups) |group_count| assert(group_count > 0);
    var native: [bindings_max]c.en_binding = undefined;
    for (native[0..bindings.len], bindings) |*target, binding| target.* = binding.toNative();
    try gpu.check(c.en_cmd_dispatch(
        command_list.handle,
        kernel.handle,
        &native,
        @intCast(bindings.len),
        groups[0],
        groups[1],
        groups[2],
    ), "en_cmd_dispatch");
}

/// EN_COPY_ROW_PITCH_ALIGNMENT and EN_COPY_OFFSET_ALIGNMENT: what `copyBufferToTexture`
/// requires of its source's row pitch and offset.
pub const copy_row_pitch_alignment = c.EN_COPY_ROW_PITCH_ALIGNMENT;
pub const copy_offset_alignment = c.EN_COPY_OFFSET_ALIGNMENT;

/// Records a copy of `size` bytes, ordered with the list's other commands.
pub fn copyBuffer(command_list: CommandList, dst: gpu.Buffer, dst_offset: u64, src: gpu.Buffer, src_offset: u64, size: u64) gpu.Error!void {
    assert(size > 0 and dst_offset + size <= dst.size and src_offset + size <= src.size);
    try gpu.check(
        c.en_cmd_copy_buffer(command_list.handle, dst.handle, dst_offset, src.handle, src_offset, size),
        "en_cmd_copy_buffer",
    );
}

/// Records a copy of rows `row_pitch` apart from `src` at `src_offset` into `region` of a
/// 2D texture or of one layer (`region.z`) of a 2D array.
pub fn copyBufferToTexture(
    command_list: CommandList,
    dst: gpu.Texture,
    region: gpu.Texture.Region,
    src: gpu.Buffer,
    src_offset: u64,
    row_pitch: u64,
) gpu.Error!void {
    assert(region.depth == 1);
    assert(row_pitch % copy_row_pitch_alignment == 0 and src_offset % copy_offset_alignment == 0);
    assert(src_offset + row_pitch * region.height <= src.size);
    const native: c.en_texture_region = .{
        .x = region.x,
        .y = region.y,
        .z = region.z,
        .width = region.width,
        .height = region.height,
        .depth = region.depth,
    };
    try gpu.check(
        c.en_cmd_copy_buffer_to_texture(command_list.handle, dst.handle, &native, src.handle, src_offset, row_pitch),
        "en_cmd_copy_buffer_to_texture",
    );
}

/// Submits the list; with `signal`, its fence reaches the value once the work completes.
pub fn submit(command_list: CommandList, signal: ?gpu.Fence.Signal) gpu.Error!void {
    const fence = if (signal) |s| s.fence.handle else null;
    const value = if (signal) |s| s.value else 0;
    try gpu.check(c.en_cmd_submit(command_list.handle, fence, value), "en_cmd_submit");
}
