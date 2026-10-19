//! A GPU device. It must outlive every resource, kernel, and surface created from it.

const std = @import("std");
const c = @import("c");
const gpu = @import("../gpu.zig");

const Device = @This();

handle: *c.en_device,

pub const Backend = enum(c.en_backend) { default, vulkan, metal, d3d12, cuda };

comptime {
    gpu.assertMirrorsC(Backend, "EN_BACKEND_");
}

pub const Options = struct {
    backend: Backend,
    validation: bool,
    /// Directories searched for Slang modules, in order. Copied by the device.
    shader_search_paths: []const [*:0]const u8,
    /// Persistent shader cache directory; null disables the cache.
    shader_cache_path: ?[*:0]const u8,
};

/// Release with `deinit` after every object created from the device.
pub fn init(options: Options) gpu.Error!Device {
    const desc: c.en_device_desc = .{
        .backend = @backingInt(options.backend),
        .enable_validation = @intFromBool(options.validation),
        .shader_search_paths = @ptrCast(options.shader_search_paths.ptr),
        .shader_search_path_count = @intCast(options.shader_search_paths.len),
        .shader_cache_path = options.shader_cache_path,
    };
    var handle: ?*c.en_device = null;
    try gpu.check(c.en_device_create(&desc, &handle), "en_device_create");
    return .{ .handle = handle.? };
}

pub fn deinit(device: Device) void {
    c.en_device_destroy(device.handle);
}

pub fn info(device: Device) c.en_device_info {
    var result: c.en_device_info = .{};
    c.en_device_get_info(device.handle, &result);
    return result;
}

/// Blocks until all submitted work has finished.
pub fn waitIdle(device: Device) gpu.Error!void {
    try gpu.check(c.en_device_wait_idle(device.handle), "en_device_wait_idle");
}
