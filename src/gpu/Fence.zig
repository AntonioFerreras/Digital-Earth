//! A counter the GPU raises as submitted work completes (`CommandList.submit`), so the host
//! can tell when a copy or dispatch has finished without waiting for the whole device.

const c = @import("c");
const gpu = @import("../gpu.zig");

const Fence = @This();

handle: *c.en_fence,

/// What a submit raises the fence to once its work completes.
pub const Signal = struct {
    fence: Fence,
    /// Larger than every value signaled before.
    value: u64,
};

pub fn init(device: gpu.Device) gpu.Error!Fence {
    var handle: ?*c.en_fence = null;
    try gpu.check(c.en_fence_create(device.handle, &handle), "en_fence_create");
    return .{ .handle = handle.? };
}

pub fn deinit(fence: Fence) void {
    c.en_fence_destroy(fence.handle);
}

/// The largest value signaled so far; never blocks.
pub fn value(fence: Fence) gpu.Error!u64 {
    var result: u64 = 0;
    try gpu.check(c.en_fence_value(fence.handle, &result), "en_fence_value");
    return result;
}

/// Blocks until the fence reaches `target`.
pub fn wait(fence: Fence, device: gpu.Device, target: u64) gpu.Error!void {
    try gpu.check(c.en_fence_wait(device.handle, fence.handle, target), "en_fence_wait");
}
