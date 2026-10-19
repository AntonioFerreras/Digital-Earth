//! A compiled compute entry point owned by the caller; destroy it before its device.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const Kernel = @This();

handle: *c.en_kernel,
thread_group_size: [3]u32,

const log = std.log.scoped(.gpu);

/// A global shader parameter located by name; valid only for the kernel it came from.
pub const Param = struct {
    location: c.en_param,
    kind: Kind,
    /// Byte size of the uniform data for `.data`, else 0.
    size: u64,

    pub const Kind = enum(c.en_param_kind) {
        data,
        buffer,
        rw_buffer,
        texture,
        rw_texture,
        sampler,
    };

    comptime {
        gpu.assertMirrorsC(Kind, "EN_PARAM_");
    }
};

pub const LayoutError = gpu.Error || error{LayoutMismatch};

/// Compiles `entry` of the Slang module `module`, found through the device's search paths.
/// A generic entry takes one Slang expression per generic parameter in `generic_args`,
/// e.g. "KPlanes<8>"; a plain one takes none.
pub fn init(device: gpu.Device, module: [*:0]const u8, entry: [*:0]const u8, generic_args: []const [*:0]const u8) gpu.Error!Kernel {
    var handle: ?*c.en_kernel = null;
    try gpu.check(c.en_kernel_create(
        device.handle,
        module,
        entry,
        @ptrCast(generic_args.ptr),
        @intCast(generic_args.len),
        &handle,
    ), "en_kernel_create");
    var thread_group_size: [3]u32 = undefined;
    c.en_kernel_thread_group_size(handle, &thread_group_size);
    for (thread_group_size) |threads| assert(threads > 0);
    return .{ .handle = handle.?, .thread_group_size = thread_group_size };
}

pub fn deinit(kernel: Kernel) void {
    c.en_kernel_destroy(kernel.handle);
}

pub fn param(kernel: Kernel, path: [*:0]const u8) gpu.Error!Param {
    return try kernel.paramOptional(path) orelse {
        log.err("en_kernel_find_param failed (NotFound): no shader parameter '{s}'", .{path});
        return error.NotFound;
    };
}

/// `param` for a global the kernel may not declare: null, and nothing logged, if absent.
pub fn paramOptional(kernel: Kernel, path: [*:0]const u8) gpu.Error!?Param {
    var info: c.en_param_info = .{};
    const status = c.en_kernel_find_param(kernel.handle, path, &info);
    if (status == c.EN_ERR_NOT_FOUND) return null;
    try gpu.check(status, "en_kernel_find_param");
    return .{ .location = info.param, .kind = @fromBackingInt(info.kind), .size = info.size };
}

/// Verifies that every field of the extern struct `T` has the offset and size Slang
/// reflection reports for the struct-typed parameter `path`. Call once after `init`.
pub fn checkLayout(kernel: Kernel, comptime T: type, path: [*:0]const u8) LayoutError!void {
    const info = @typeInfo(T).@"struct";
    comptime assert(info.layout == .@"extern");
    inline for (info.field_names, info.field_types) |name, FieldType| {
        var offset: u64 = undefined;
        var size: u64 = undefined;
        try gpu.check(
            c.en_kernel_find_field(kernel.handle, path, name, &offset, &size),
            "en_kernel_find_field",
        );
        if (offset != @offsetOf(T, name)) {
            log.err("{s}.{s}: Slang offset {d}, Zig offset {d}", .{ path, name, offset, @offsetOf(T, name) });
            return error.LayoutMismatch;
        }
        if (size != @sizeOf(FieldType)) {
            log.err("{s}.{s}: Slang size {d}, Zig size {d}", .{ path, name, size, @sizeOf(FieldType) });
            return error.LayoutMismatch;
        }
    }
}

/// Thread groups needed to cover `threads` invocations per axis.
pub fn groupCount(kernel: Kernel, threads: [3]u32) [3]u32 {
    var groups: [3]u32 = undefined;
    for (&groups, threads, kernel.thread_group_size) |*group, thread_count, group_size| {
        assert(thread_count > 0);
        group.* = std.math.divCeil(u32, thread_count, group_size) catch unreachable;
    }
    return groups;
}
