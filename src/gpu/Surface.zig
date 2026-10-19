//! A swapchain for a native window; destroy it before its device and its window.

const std = @import("std");
const assert = std.debug.assert;
const c = @import("c");
const gpu = @import("../gpu.zig");

const Surface = @This();

handle: *c.en_surface,

pub fn init(device: gpu.Device, window: c.en_window_handle) gpu.Error!Surface {
    var handle: ?*c.en_surface = null;
    try gpu.check(c.en_surface_create(device.handle, &window, &handle), "en_surface_create");
    return .{ .handle = handle.? };
}

pub fn deinit(surface: Surface) void {
    c.en_surface_destroy(surface.handle);
}

/// (Re)creates the swapchain at `size` pixels; required before the first `present`.
pub fn configure(surface: Surface, size: [2]u32, vsync: bool) gpu.Error!void {
    assert(size[0] > 0);
    assert(size[1] > 0);
    try gpu.check(
        c.en_surface_configure(surface.handle, size[0], size[1], @intFromBool(vsync)),
        "en_surface_configure",
    );
}

/// Draws the rgba8 `image` stretched over the surface, then the ImGui frame if one was
/// begun. `error.SurfaceOutOfDate` asks the caller to `configure` again.
pub fn present(surface: Surface, device: gpu.Device, image: gpu.Texture) gpu.Error!void {
    assert(image.desc.format == .rgba8_unorm);
    try gpu.check(c.en_present(device.handle, surface.handle, image.handle), "en_present");
}
