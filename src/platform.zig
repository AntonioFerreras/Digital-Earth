//! Operating-system integration: windowing and input through SDL3.

const std = @import("std");

pub const Window = @import("platform/Window.zig");

test {
    std.testing.refAllDecls(@This());
}
