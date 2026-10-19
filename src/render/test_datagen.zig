//! `Datagen` reproducibility: a seed fixes every record bit for bit, whatever the
//! batching, and points spanning several chunks average every path. Skipped without
//! a GPU.

const std = @import("std");
const testing = std.testing;
const Luts = @import("../assets/Luts.zig");
const datagen_format = @import("../assets/datagen_format.zig");
const createTestDevice = @import("../gpu/test_gpu.zig").createTestDevice;
const Datagen = @import("Datagen.zig");

const point_count = 6;

/// Points across the domain, including the ground half (w < 0.5) and the sun-disk corner.
fn testRecords() [point_count]datagen_format.Record {
    const uvwz = [point_count][4]f32{
        .{ 0.5, 0.6, 0.7, 0.1 },
        .{ 0.2, 0.9, 0.3, 0.5 },
        .{ 0.9, 0.3, 0.55, 0.9 },
        .{ 1.0, 0.5, 0.75, 0.2 },
        .{ 0.0, 0.1, 0.45, 0.0 },
        .{ 0.6, 0.45, 0.52, 0.99 },
    };
    var records: [point_count]datagen_format.Record = undefined;
    for (&records, uvwz) |*record, point| record.* = .{ .uvwz = point, .rgb = @splat(-1) };
    return records;
}

fn generate(path_count: u32, seed: u32, batch_size: usize) ![point_count]datagen_format.Record {
    const device = try createTestDevice();
    defer device.deinit();
    var lut_dir = try std.Io.Dir.cwd().openDir(testing.io, "LUT", .{});
    defer lut_dir.close(testing.io);
    var luts: Luts = try .load(testing.allocator, testing.io, lut_dir);
    defer luts.deinit(testing.allocator);
    var datagen: Datagen = try .init(testing.allocator, device, &luts, .{
        .path_count = path_count,
        .seed = seed,
        .exclude_sun_disk = false,
    });
    defer datagen.deinit(testing.allocator);

    var records = testRecords();
    var first: usize = 0;
    while (first < records.len) : (first += batch_size) {
        const batch = records[first..@min(first + batch_size, records.len)];
        _ = try datagen.evaluate(@intCast(first), batch);
    }
    return records;
}

test "datagen output depends on the seed alone" {
    // More than one chunk per point, the last one partial.
    const path_count = 2 * 16384 + 300;
    const whole = try generate(path_count, 7, point_count);
    const split = try generate(path_count, 7, 4);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&whole), std.mem.sliceAsBytes(&split));
    for (whole) |record| {
        for (record.rgb) |channel| try testing.expect(std.math.isFinite(channel) and channel != -1);
    }
    const reseeded = try generate(path_count, 8, point_count);
    try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(&whole), std.mem.sliceAsBytes(&reseeded)));
}
