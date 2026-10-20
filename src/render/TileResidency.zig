//! Which tiles of one layer occupy which pool slots, and the indirection that lets a
//! shader find, for any tile of any face, its finest resident ancestor in one lookup.
//!
//! The indirection holds one u32 per (face, level, x, y) to the last level, indexed like
//! the tile files (`TileSet.Key.index` after the face's `entryCount(levels)` entries).
//! An entry is `slot << level_bits | level` of the finest resident tile containing it.
//! Morton order keeps the descendants of a tile contiguous on each finer level, so a
//! residency change rewrites one range per level.

const TileResidency = @This();

const std = @import("std");
const assert = std.debug.assert;
const TileSet = @import("../assets/TileSet.zig");

/// Mirror of the GPU indirection buffer, `face_count * entries_per_face` entries.
indirection: []u32,
/// The tile in each slot, or null for a free slot.
slots: []?Tile,
/// Levels 0 to `last_level`.
last_level: u32,

pub const face_count = 6;
pub const level_bits = 4;

pub const Tile = struct {
    face: u32,
    key: TileSet.Key,
};

/// A contiguous run of indirection entries that changed and must reach the GPU.
pub const Range = struct { first: usize, count: usize };

pub fn init(gpa: std.mem.Allocator, last_level: u32, slot_count: u32) error{OutOfMemory}!TileResidency {
    assert(last_level < 1 << level_bits);
    const indirection = try gpa.alloc(u32, face_count * TileSet.entryCount(last_level + 1));
    errdefer gpa.free(indirection);
    // Unset until level 0 of every face is resident; shaders never see this state.
    @memset(indirection, std.math.maxInt(u32));
    const slots = try gpa.alloc(?Tile, slot_count);
    @memset(slots, null);
    return .{ .indirection = indirection, .slots = slots, .last_level = last_level };
}

pub fn deinit(residency: *TileResidency, gpa: std.mem.Allocator) void {
    gpa.free(residency.slots);
    gpa.free(residency.indirection);
    residency.* = undefined;
}

pub fn entriesPerFace(residency: TileResidency) usize {
    return TileSet.entryCount(residency.last_level + 1);
}

pub fn entryIndex(residency: TileResidency, tile: Tile) usize {
    assert(tile.face < face_count and tile.key.level <= residency.last_level);
    return tile.face * residency.entriesPerFace() + tile.key.index();
}

pub fn encode(slot: u32, level: u32) u32 {
    assert(level < 1 << level_bits);
    return slot << level_bits | level;
}

/// The level of the finest resident tile containing `tile`, or null before any is.
pub fn residentLevel(residency: TileResidency, tile: Tile) ?u32 {
    const entry = residency.indirection[residency.entryIndex(tile)];
    if (entry == std.math.maxInt(u32)) return null;
    return entry & ((1 << level_bits) - 1);
}

/// Makes `slot` hold `tile` and points the tile and those of its descendants without a
/// finer resident tile at it. Calls `changed` with each rewritten range. The slot must
/// be free and the tile's parent resident (or the tile at level 0).
pub fn insert(residency: *TileResidency, slot: u32, tile: Tile, context: anytype, comptime changed: fn (@TypeOf(context), Range) void) void {
    assert(residency.slots[slot] == null);
    if (tile.key.level > 0) assert(residency.residentLevel(parentOf(tile)) == tile.key.level - 1);
    residency.slots[slot] = tile;
    residency.rewriteSubtree(tile, encode(slot, tile.key.level), tile.key.level, context, changed);
}

/// Frees `slot` and points the entries that used it back at its tile's parent. The tile
/// must not be level 0 and none of its children may be resident.
pub fn remove(residency: *TileResidency, slot: u32, context: anytype, comptime changed: fn (@TypeOf(context), Range) void) void {
    const tile = residency.slots[slot].?;
    assert(tile.key.level > 0);
    const parent_entry = residency.indirection[residency.entryIndex(parentOf(tile))];
    residency.slots[slot] = null;
    residency.rewriteSubtree(tile, parent_entry, tile.key.level, context, changed);
}

/// Sets the entries of `tile` and its descendants that point at a tile coarser than
/// `owner_level`, or at the tile being removed, to `value`.
fn rewriteSubtree(residency: *TileResidency, tile: Tile, value: u32, owner_level: u32, context: anytype, comptime changed: fn (@TypeOf(context), Range) void) void {
    const face_base = tile.face * residency.entriesPerFace();
    var level = tile.key.level;
    var first = TileSet.Key.index(tile.key) - TileSet.entryCount(level);
    var count: usize = 1;
    while (level <= residency.last_level) : (level += 1) {
        const base = face_base + TileSet.entryCount(level) + first;
        const entries = residency.indirection[base..][0..count];
        // A finer resident tile owns its own subtree; its entries keep pointing at it.
        for (entries) |*entry| {
            if (entry.* == std.math.maxInt(u32) or entry.* & ((1 << level_bits) - 1) <= owner_level) entry.* = value;
        }
        changed(context, .{ .first = base, .count = count });
        first *= 4;
        count *= 4;
    }
}

pub fn parentOf(tile: Tile) Tile {
    assert(tile.key.level > 0);
    return .{ .face = tile.face, .key = .{ .level = tile.key.level - 1, .x = tile.key.x / 2, .y = tile.key.y / 2 } };
}

const testing = std.testing;

fn ignore(_: void, _: Range) void {}

test "every entry points at its finest resident ancestor as tiles come and go" {
    var residency: TileResidency = try .init(testing.allocator, 3, 8);
    defer residency.deinit(testing.allocator);
    for (0..face_count) |face| residency.insert(@intCast(face), .{ .face = @intCast(face), .key = .{ .level = 0, .x = 0, .y = 0 } }, {}, ignore);

    const leaf: Tile = .{ .face = 2, .key = .{ .level = 3, .x = 5, .y = 6 } };
    try testing.expectEqual(0, residency.residentLevel(leaf));
    const child: Tile = .{ .face = 2, .key = .{ .level = 1, .x = 1, .y = 1 } };
    residency.insert(6, child, {}, ignore);
    try testing.expectEqual(1, residency.residentLevel(leaf));
    try testing.expectEqual(encode(6, 1), residency.indirection[residency.entryIndex(leaf)]);
    // A sibling subtree and other faces are untouched.
    try testing.expectEqual(0, residency.residentLevel(.{ .face = 2, .key = .{ .level = 3, .x = 1, .y = 6 } }));
    try testing.expectEqual(0, residency.residentLevel(.{ .face = 3, .key = .{ .level = 3, .x = 5, .y = 6 } }));

    const grandchild: Tile = .{ .face = 2, .key = .{ .level = 2, .x = 2, .y = 3 } };
    residency.insert(7, grandchild, {}, ignore);
    try testing.expectEqual(2, residency.residentLevel(leaf));
    try testing.expectEqual(1, residency.residentLevel(.{ .face = 2, .key = .{ .level = 3, .x = 7, .y = 7 } }));

    residency.remove(7, {}, ignore);
    try testing.expectEqual(encode(6, 1), residency.indirection[residency.entryIndex(leaf)]);
    residency.remove(6, {}, ignore);
    try testing.expectEqual(encode(2, 0), residency.indirection[residency.entryIndex(leaf)]);
}

test "a change reports one contiguous range per level" {
    var residency: TileResidency = try .init(testing.allocator, 2, 8);
    defer residency.deinit(testing.allocator);
    var ranges: std.ArrayList(Range) = .empty;
    defer ranges.deinit(testing.allocator);
    const Collect = struct {
        fn add(list: *std.ArrayList(Range), range: Range) void {
            list.append(testing.allocator, range) catch unreachable;
        }
    };
    residency.insert(0, .{ .face = 1, .key = .{ .level = 0, .x = 0, .y = 0 } }, &ranges, Collect.add);
    const per_face = residency.entriesPerFace();
    try testing.expectEqualSlices(Range, &.{
        .{ .first = per_face, .count = 1 },
        .{ .first = per_face + 1, .count = 4 },
        .{ .first = per_face + 5, .count = 16 },
    }, ranges.items);
}
