//! The process-wide general-purpose allocator: leak-checked in Debug builds,
//! `smp_allocator` otherwise.

const std = @import("std");
const builtin = @import("builtin");

const checked = builtin.mode == .debug;

var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

/// Thread-safe; valid from process start until `deinit`.
pub fn allocator() std.mem.Allocator {
    return if (checked) debug_allocator.allocator() else std.heap.smp_allocator;
}

/// Reports every live allocation (Debug only) and returns whether any leaked.
pub fn deinit() std.heap.Check {
    return if (checked) debug_allocator.deinit() else .ok;
}
