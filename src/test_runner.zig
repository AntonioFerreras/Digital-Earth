//! Test runner for `zig build test` (simple mode, no build-server protocol).
//!
//! `--test-filter <substring>` may repeat; a test runs when its name contains any filter.
//! Each test gets fresh `std.testing` state, mirroring lib/compiler/test_runner.zig: a
//! leak-checked allocator, an Io instance, and the environment. Log output is buffered
//! per test and shown only for failures and skips, where it explains the outcome.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const testing = std.testing;

pub const std_options: std.Options = .{ .logFn = log };

var log_buffer: [64 * 1024]u8 = undefined;
var log_writer: Io.Writer = .fixed(&log_buffer);
var log_truncated = false;

const Outcome = enum { pass, skip, fail };

pub fn main(init: std.process.Init.Minimal) void {
    var arena_instance: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    const filters = parseFilters(arena, init.args) catch |err| {
        std.process.fatal("test runner: {t}", .{err});
    };
    const clock_io = Io.Threaded.global_single_threaded.io();

    var counts = std.EnumArray(Outcome, usize).initFill(0);
    for (builtin.test_functions) |test_fn| {
        if (!matchesAny(test_fn.name, filters)) continue;

        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        testing.io_instance = .init(testing.allocator, .{
            .argv0 = .init(init.args),
            .environ = init.environ,
        });
        testing.environ = init.environ;
        testing.log_level = .warn;
        log_writer.end = 0;
        log_truncated = false;

        const start = Io.Timestamp.now(clock_io, .awake);
        const result = test_fn.func();
        const elapsed = start.durationTo(Io.Timestamp.now(clock_io, .awake));
        testing.io_instance.deinit();
        const leak_count = testing.allocator_instance.deinit();

        const failure: ?anyerror = if (result) |_| null else |err| switch (err) {
            error.SkipZigTest => null,
            else => err,
        };
        const outcome: Outcome = if (failure != null or leak_count != 0)
            .fail
        else if (result == error.SkipZigTest) .skip else .pass;
        counts.getPtr(outcome).* += 1;
        std.debug.print("{s} {s} ({f})", .{ statusLabel(outcome), test_fn.name, elapsed });
        if (failure) |err| std.debug.print(": {t}", .{err});
        if (leak_count != 0) std.debug.print(": {d} leaked allocations", .{leak_count});
        std.debug.print("\n", .{});
        if (outcome != .pass) printLogs();
        if (failure != null) {
            if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
        }
    }

    const ran = counts.get(.pass) + counts.get(.skip) + counts.get(.fail);
    std.debug.print("{d} passed, {d} skipped, {d} failed\n", .{
        counts.get(.pass), counts.get(.skip), counts.get(.fail),
    });
    if (ran == 0) {
        std.debug.print("no test matched the filters\n", .{});
        std.process.exit(1);
    }
    if (counts.get(.fail) != 0) std.process.exit(1);
}

fn parseFilters(arena: std.mem.Allocator, args: std.process.Args) ![]const []const u8 {
    const argv = try args.toSlice(arena);
    var filters: std.ArrayList([]const u8) = .empty;
    var index: usize = 1;
    while (index < argv.len) : (index += 1) {
        if (!std.mem.eql(u8, argv[index], "--test-filter")) {
            std.debug.print("unknown argument '{s}'; expected --test-filter <substring>\n", .{argv[index]});
            return error.InvalidArguments;
        }
        index += 1;
        if (index == argv.len) return error.MissingFilter;
        try filters.append(arena, argv[index]);
    }
    return filters.items;
}

fn matchesAny(name: []const u8, filters: []const []const u8) bool {
    if (filters.len == 0) return true;
    for (filters) |filter| {
        if (std.mem.find(u8, name, filter) != null) return true;
    }
    return false;
}

fn statusLabel(outcome: Outcome) []const u8 {
    return switch (outcome) {
        .pass => "PASS",
        .skip => "SKIP",
        .fail => "FAIL",
    };
}

fn printLogs() void {
    var lines = std.mem.splitScalar(u8, log_writer.buffered(), '\n');
    while (lines.next()) |line| {
        if (line.len != 0) std.debug.print("    {s}\n", .{line});
    }
    if (log_truncated) std.debug.print("    (log truncated)\n", .{});
}

fn log(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@backingInt(level) > @backingInt(testing.log_level)) return;
    log_writer.print("[" ++ @tagName(scope) ++ "] (" ++ @tagName(level) ++ "): " ++ format ++ "\n", args) catch {
        log_truncated = true;
    };
}
