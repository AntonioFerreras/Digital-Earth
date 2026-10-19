//! A recording of compute dispatches. `submit` consumes it, also on failure.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const CommandList = @This();

handle: *c.en_cmd,

/// A bounded stack array holds the converted bindings of one dispatch.
pub const bindings_max = 32;

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

pub fn submit(command_list: CommandList) gpu.Error!void {
    try gpu.check(c.en_cmd_submit(command_list.handle), "en_cmd_submit");
}
