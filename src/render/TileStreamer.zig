//! Streams tiles into a `TilePool` from what its lookups asked for, without ever waiting
//! for the GPU. Each frame that finds a ring position free copies the request bits to its
//! readback buffer behind the frame's fence and clears them; otherwise the bits gather
//! until one is. Every later frame reads the readbacks whose fences it finds reached,
//! queues each wanted tile with its ancestors below the resident one, decodes up to
//! `tiles_per_frame` of them, coarsest first, across the threads of the files' `io`, and
//! records their uploads, with the indirection ranges they rewrite, ahead of its own
//! dispatches. A full pool gives up the least recently wanted tile that has no resident
//! children.

const TileStreamer = @This();

const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const TileSet = @import("../assets/TileSet.zig");
const TilePool = @import("TilePool.zig");
const TileResidency = TilePool.TileResidency;

fence: gpu.Fence,
/// The value the next submitted frame signals; frame n signals n.
frame: u64,
/// Per ring position: the request bits of the frame that used it last, and the uploads
/// that frame recorded.
readbacks: [ring_size]gpu.Buffer,
staging: [ring_size]gpu.Buffer,
/// The frame that last used each ring position, 0 for none.
ring_frames: [ring_size]u64,
/// Whether each ring position holds request bits not read yet.
unread: [ring_size]bool,
/// The ring position of the frame being recorded; null while all are in flight or unread.
position: ?usize,
/// Clears the request bits.
zeros: gpu.Buffer,
slot_use: std.EnumArray(TileSet.Layer, []SlotUse),
free_slots: std.EnumArray(TileSet.Layer, std.ArrayList(u32)),
/// This frame's recorded uploads into `staging`.
copies: std.ArrayList(Copy),
staged_bytes: u64,
/// This frame's decodes, `tiles_per_frame` of them.
decodes: []Decode,
/// Tiles wanted by the readbacks read so far, by layer and entry index, to load next.
wanted: std.AutoArrayHashMapUnmanaged(Wanted, void),
/// The last frame that loaded tiles.
last_load_frame: u64,
/// Whether a readback of a frame that saw the last loads wanted nothing more to load.
quiet: bool,
/// Tiles the last readback wanted next, present or not in the files.
wanted_last: u32,

/// More than the frames a GPU-bound loop runs ahead, so that one position is free.
const ring_size = 8;
pub const tiles_per_frame = 128;

const SlotUse = struct {
    /// The last frame whose lookups wanted the tile or one below it.
    frame: u64 = 0,
    resident_children: u8 = 0,
};

const Wanted = struct { layer: TileSet.Layer, tile: TileResidency.Tile };

const Decode = struct {
    wanted: Wanted,
    payload: []const u8,
    scratch: TilePool.Scratch,
    data: TileSet.DecodeError!TilePool.TileData,

    fn run(decode: *Decode) void {
        decode.data = decode.scratch.decode(decode.wanted.layer, decode.payload);
    }
};

const Copy = union(enum) {
    tile: struct { texture: gpu.Texture, slot: u32, side: u32, offset: u64, row_pitch: u64 },
    indirection: BufferRange,
    block_max: BufferRange,

    const BufferRange = struct { dst_offset: u64, offset: u64, size: u64 };
};

/// Byte layout of one staged tile: rows of `side` texels `texel_size` bytes each.
fn rowPitch(side: u32, texel_size: u32) u64 {
    return std.mem.alignForward(u64, side * texel_size, gpu.CommandList.copy_row_pitch_alignment);
}

fn stagedSize(side: u32, texel_size: u32) u64 {
    return std.mem.alignForward(u64, rowPitch(side, texel_size) * side, gpu.CommandList.copy_offset_alignment);
}

/// Staging per frame: the largest tile's planes `tiles_per_frame` times, and the
/// indirection ranges an insertion and an eviction rewrite below the base levels.
fn stagingSize(pool: *const TilePool) u64 {
    var subtree: usize = 0;
    for (pool.residency.values) |residency| subtree = @max(subtree, residency.subtreeEntries(TilePool.base_last_level + 1));
    // Each plane may start after up to one alignment's padding.
    const padding = 2 * gpu.CommandList.copy_offset_alignment;
    const height = stagedSize(TileSet.height_side, 2) + stagedSize(TileSet.height_side, 1) + @sizeOf(@FieldType(TilePool.HeightData, "block_max"));
    const tile = padding + @max(height, stagedSize(TileSet.surface_side, 4));
    return tiles_per_frame * (tile + 2 * subtree * @sizeOf(u32));
}

/// Copies per tile: two planes, the block maxima, and one range per level for an
/// insertion and an eviction.
const copies_per_tile = 3 + 2 * (1 << TileResidency.level_bits);

pub fn init(gpa: std.mem.Allocator, device: gpu.Device, pool: *const TilePool) (gpu.Error || error{OutOfMemory})!TileStreamer {
    const fence: gpu.Fence = try .init(device);
    errdefer fence.deinit();
    const request_size = pool.requests.size;
    var readbacks: [ring_size]gpu.Buffer = undefined;
    var staging: [ring_size]gpu.Buffer = undefined;
    var created: usize = 0;
    errdefer for (readbacks[0..created], staging[0..created]) |readback, buffer| {
        readback.deinit();
        buffer.deinit();
    };
    for (&readbacks, &staging) |*readback, *buffer| {
        readback.* = try .init(device, .{ .size = request_size, .element_size = 0, .usage = .{ .copy_destination = true }, .memory = .readback, .label = "tile request readback" }, null);
        errdefer readback.deinit();
        buffer.* = try .init(device, .{ .size = stagingSize(pool), .element_size = 0, .usage = .{ .copy_source = true }, .memory = .upload, .label = "tile staging" }, null);
        created += 1;
    }
    const zero_bytes = try gpa.alloc(u8, request_size);
    defer gpa.free(zero_bytes);
    @memset(zero_bytes, 0);
    const zeros: gpu.Buffer = try .init(device, .{ .size = request_size, .element_size = 0, .usage = .{ .copy_source = true }, .memory = .device, .label = "tile request zeros" }, zero_bytes);
    errdefer zeros.deinit();
    try pool.requests.upload(device, 0, zero_bytes);

    var slot_use: std.EnumArray(TileSet.Layer, []SlotUse) = .initUndefined();
    var free_slots: std.EnumArray(TileSet.Layer, std.ArrayList(u32)) = .initFill(.empty);
    errdefer for (std.enums.values(TileSet.Layer)) |layer| free_slots.getPtr(layer).deinit(gpa);
    var allocated: usize = 0;
    errdefer for (slot_use.values[0..allocated]) |uses| gpa.free(uses);
    for (std.enums.values(TileSet.Layer)) |layer| {
        const residency = pool.residency.get(layer);
        const uses = try gpa.alloc(SlotUse, residency.slots.len);
        slot_use.set(layer, uses);
        allocated += 1;
        @memset(uses, .{});
        for (residency.slots, 0..) |occupant, slot| {
            if (occupant) |tile| {
                if (tile.key.level > 0) uses[residency.slotOf(TileResidency.parentOf(tile)).?].resident_children += 1;
            } else {
                try free_slots.getPtr(layer).append(gpa, @intCast(slot));
            }
        }
        // Lower slots are taken first.
        std.mem.reverse(u32, free_slots.getPtr(layer).items);
    }
    var copies: std.ArrayList(Copy) = try .initCapacity(gpa, tiles_per_frame * copies_per_tile);
    errdefer copies.deinit(gpa);
    const decodes = try gpa.alloc(Decode, tiles_per_frame);
    return .{
        .fence = fence,
        .frame = 1,
        .readbacks = readbacks,
        .staging = staging,
        .ring_frames = @splat(0),
        .unread = @splat(false),
        .position = null,
        .zeros = zeros,
        .slot_use = slot_use,
        .free_slots = free_slots,
        .copies = copies,
        .staged_bytes = 0,
        .decodes = decodes,
        .wanted = .empty,
        .last_load_frame = 0,
        .quiet = false,
        .wanted_last = 0,
    };
}

pub fn deinit(streamer: *TileStreamer, gpa: std.mem.Allocator) void {
    gpa.free(streamer.decodes);
    streamer.wanted.deinit(gpa);
    streamer.copies.deinit(gpa);
    for (std.enums.values(TileSet.Layer)) |layer| {
        streamer.free_slots.getPtr(layer).deinit(gpa);
        gpa.free(streamer.slot_use.get(layer));
    }
    streamer.zeros.deinit();
    for (streamer.readbacks, streamer.staging) |readback, staging| {
        readback.deinit();
        staging.deinit();
    }
    streamer.fence.deinit();
    streamer.* = undefined;
}

pub const UpdateError = gpu.Error || TileSet.DecodeError || Io.Cancelable || error{OutOfMemory};

/// Reads the requests of finished frames and stages this frame's uploads; returns
/// whether any tile changed, which changes what the renderer accumulates.
pub fn update(streamer: *TileStreamer, gpa: std.mem.Allocator, device: gpu.Device, pool: *TilePool, files: *const TilePool.Files) UpdateError!bool {
    streamer.copies.clearRetainingCapacity();
    streamer.staged_bytes = 0;
    const reached = try streamer.fence.value();
    var newest_read: ?u64 = null;
    for (streamer.ring_frames, &streamer.unread, 0..) |ring_frame, *unread, position| {
        if (!unread.* or ring_frame > reached) continue;
        try streamer.readRequests(gpa, device, pool, position);
        unread.* = false;
        newest_read = @max(newest_read orelse 0, ring_frame);
    }
    if (newest_read != null) streamer.wanted_last = @intCast(streamer.wanted.count());
    // A position's buffers are reused once its frame is done and its bits are read.
    streamer.position = for (streamer.ring_frames, streamer.unread, 0..) |ring_frame, unread, position| {
        if (!unread and ring_frame <= reached) break position;
    } else null;
    const position = streamer.position orelse return false;
    if (streamer.wanted.count() == 0) {
        if (newest_read) |frame| streamer.quiet = frame >= streamer.last_load_frame;
        return false;
    }

    const Order = struct {
        fn lessThan(keys: []const Wanted, a: usize, b: usize) bool {
            return keys[a].tile.key.level < keys[b].tile.key.level;
        }
    };
    streamer.wanted.sortUnstable(struct {
        keys: []const Wanted,
        pub fn lessThan(context: @This(), a: usize, b: usize) bool {
            return Order.lessThan(context.keys, a, b);
        }
    }{ .keys = streamer.wanted.keys() });

    var decode_count: usize = 0;
    for (streamer.wanted.keys()) |wanted| {
        if (decode_count == streamer.decodes.len) break;
        // Wanted by readbacks from before it arrived.
        if (pool.residency.get(wanted.layer).residentLevel(wanted.tile).? >= wanted.tile.key.level) continue;
        const set = files.sets.get(wanted.layer)[wanted.tile.face];
        const entry = set.entry(wanted.tile.key) orelse continue;
        const decode = &streamer.decodes[decode_count];
        decode.wanted = wanted;
        decode.payload = set.payload(entry);
        decode_count += 1;
    }
    var group: Io.Group = .init;
    for (streamer.decodes[0..decode_count]) |*decode| group.async(files.io, Decode.run, .{decode});
    try group.await(files.io);

    const staging = try streamer.staging[position].map(device);
    defer streamer.staging[position].unmap(device) catch {};
    var loaded: usize = 0;
    for (streamer.decodes[0..decode_count]) |*decode| {
        const wanted = decode.wanted;
        // Decoding comes before a slot is taken, so a corrupt tile changes nothing.
        const data = try decode.data;
        const residency = pool.residency.getPtr(wanted.layer);
        // The parent arrives earlier in the batch, or is missing from the files.
        if (residency.residentLevel(wanted.tile) != wanted.tile.key.level - 1) continue;
        const slot = streamer.takeSlot(wanted.layer, pool, staging) orelse break;
        streamer.stageTile(pool, slot, data, staging);
        const parent_slot = residency.slotOf(TileResidency.parentOf(wanted.tile)).?;
        streamer.slot_use.get(wanted.layer)[parent_slot].resident_children += 1;
        streamer.slot_use.get(wanted.layer)[slot] = .{ .frame = streamer.frame };
        var stager: Stager = .{ .streamer = streamer, .staging = staging, .entries = residency.indirection, .base = indirectionBase(pool, wanted.layer) };
        residency.insert(slot, wanted.tile, &stager, Stager.stage);
        loaded += 1;
    }
    streamer.wanted.clearRetainingCapacity();
    if (loaded > 0) {
        streamer.last_load_frame = streamer.frame;
        streamer.quiet = false;
    } else if (newest_read) |frame| {
        streamer.quiet = frame >= streamer.last_load_frame;
    }
    return loaded > 0;
}

/// Records this frame's uploads; ahead of the dispatches that read them.
pub fn recordUploads(streamer: *const TileStreamer, command_list: gpu.CommandList, pool: *const TilePool) gpu.Error!void {
    const position = streamer.position orelse return;
    const staging = streamer.staging[position];
    for (streamer.copies.items) |copy| switch (copy) {
        .tile => |tile| try command_list.copyBufferToTexture(tile.texture, .{ .x = 0, .y = 0, .z = tile.slot, .width = tile.side, .height = tile.side, .depth = 1 }, staging, tile.offset, tile.row_pitch),
        .indirection => |range| try command_list.copyBuffer(pool.indirection, range.dst_offset, staging, range.offset, range.size),
        .block_max => |range| try command_list.copyBuffer(pool.block_max, range.dst_offset, staging, range.offset, range.size),
    };
}

/// Records the copy of this frame's request bits for a later frame, and their clearing,
/// when the frame has a ring position.
pub fn recordFeedback(streamer: *const TileStreamer, command_list: gpu.CommandList, pool: *const TilePool) gpu.Error!void {
    const position = streamer.position orelse return;
    const readback = streamer.readbacks[position];
    try command_list.copyBuffer(readback, 0, pool.requests, 0, pool.requests.size);
    try command_list.copyBuffer(pool.requests, 0, streamer.zeros, 0, pool.requests.size);
}

/// Resident tiles per layer.
pub fn residentCount(streamer: *const TileStreamer, pool: *const TilePool, layer: TileSet.Layer) usize {
    return pool.residency.get(layer).slots.len - streamer.free_slots.get(layer).items.len;
}

/// Whether the lookups want no tile that could still arrive.
pub fn settled(streamer: *const TileStreamer) bool {
    return streamer.quiet;
}

/// The fence signal of the frame being recorded; call `submitted` after submitting it.
pub fn signal(streamer: *const TileStreamer) gpu.Fence.Signal {
    return .{ .fence = streamer.fence, .value = streamer.frame };
}

pub fn submitted(streamer: *TileStreamer) void {
    if (streamer.position) |position| {
        streamer.ring_frames[position] = streamer.frame;
        streamer.unread[position] = true;
    }
    streamer.frame += 1;
}

/// Marks the resident tiles on the way to each wanted tile as used, and queues the
/// missing ones.
fn readRequests(streamer: *TileStreamer, gpa: std.mem.Allocator, device: gpu.Device, pool: *TilePool, position: usize) (gpu.Error || error{OutOfMemory})!void {
    const bytes = try streamer.readbacks[position].map(device);
    defer streamer.readbacks[position].unmap(device) catch {};
    const words = std.mem.bytesAsSlice(u32, bytes);
    const height_entries = pool.residency.get(.height).indirection.len;
    for (words, 0..) |word, word_index| {
        var bits = word;
        while (bits != 0) : (bits &= bits - 1) {
            const index = word_index * 32 + @ctz(bits);
            const layer: TileSet.Layer = if (index < height_entries) .height else .surface;
            const residency = pool.residency.getPtr(layer);
            if (index - (if (layer == .height) 0 else height_entries) >= residency.indirection.len) continue;
            const tile = residency.tileAt(index - (if (layer == .height) 0 else height_entries));
            const resident_level = residency.residentLevel(tile).?;
            // The resident ancestor and those above it serve this tile's lookups.
            var ancestor = tile;
            while (ancestor.key.level > resident_level) ancestor = TileResidency.parentOf(ancestor);
            while (true) {
                streamer.slot_use.get(layer)[residency.slotOf(ancestor).?].frame = streamer.frame;
                if (ancestor.key.level == 0) break;
                ancestor = TileResidency.parentOf(ancestor);
            }
            var missing = tile;
            while (missing.key.level > resident_level) : (missing = TileResidency.parentOf(missing)) {
                try streamer.wanted.put(gpa, .{ .layer = layer, .tile = missing }, {});
            }
        }
    }
}

/// A free slot, or the slot of the least recently wanted tile without resident children
/// that no frame in flight reads, evicted; null when there is none.
fn takeSlot(streamer: *TileStreamer, layer: TileSet.Layer, pool: *TilePool, staging: []u8) ?u32 {
    if (streamer.free_slots.getPtr(layer).pop()) |slot| return slot;
    const residency = pool.residency.getPtr(layer);
    const uses = streamer.slot_use.get(layer);
    var victim: ?u32 = null;
    for (residency.slots, uses, 0..) |occupant, use, slot| {
        const tile = occupant orelse continue;
        if (tile.key.level <= TilePool.base_last_level or use.resident_children > 0) continue;
        if (use.frame + ring_size >= streamer.frame) continue;
        if (victim == null or use.frame < uses[victim.?].frame) victim = @intCast(slot);
    }
    const slot = victim orelse return null;
    uses[residency.slotOf(TileResidency.parentOf(residency.slots[slot].?)).?].resident_children -= 1;
    var stager: Stager = .{ .streamer = streamer, .staging = staging, .entries = residency.indirection, .base = indirectionBase(pool, layer) };
    residency.remove(slot, &stager, Stager.stage);
    return slot;
}

fn indirectionBase(pool: *const TilePool, layer: TileSet.Layer) u64 {
    return switch (layer) {
        .height => 0,
        .surface => pool.residency.get(.height).indirection.len,
    };
}

/// Copies rewritten indirection ranges into the staging buffer as they change; the
/// staging size and copy capacity cover every change one frame can make.
const Stager = struct {
    streamer: *TileStreamer,
    staging: []u8,
    /// The layer's host indirection, already rewritten when `stage` runs.
    entries: []const u32,
    /// The layer's first entry in the indirection buffer.
    base: u64,

    fn stage(stager: *Stager, range: TileResidency.Range) void {
        const streamer = stager.streamer;
        const bytes = std.mem.sliceAsBytes(stager.entries[range.first..][0..range.count]);
        const offset = streamer.staged_bytes;
        @memcpy(stager.staging[offset..][0..bytes.len], bytes);
        streamer.staged_bytes += bytes.len;
        streamer.copies.appendAssumeCapacity(.{ .indirection = .{
            .dst_offset = (stager.base + range.first) * @sizeOf(u32),
            .offset = offset,
            .size = bytes.len,
        } });
    }
};

fn stageTile(streamer: *TileStreamer, pool: *const TilePool, slot: u32, data: TilePool.TileData, staging: []u8) void {
    switch (data) {
        .height => |height| {
            streamer.stagePlane(pool.heights, slot, TileSet.height_side, 2, std.mem.sliceAsBytes(&height.tile.height), staging);
            streamer.stagePlane(pool.water, slot, TileSet.height_side, 1, &height.tile.water, staging);
            const bytes = std.mem.asBytes(&height.block_max);
            const offset = streamer.staged_bytes;
            @memcpy(staging[offset..][0..bytes.len], bytes);
            streamer.staged_bytes += bytes.len;
            streamer.copies.appendAssumeCapacity(.{ .block_max = .{ .dst_offset = TilePool.blockMaxOffset(slot), .offset = offset, .size = bytes.len } });
        },
        .surface => |rgba| streamer.stagePlane(pool.surface, slot, TileSet.surface_side, 4, rgba, staging),
    }
}

fn stagePlane(streamer: *TileStreamer, texture: gpu.Texture, slot: u32, side: u32, texel_size: u32, texels: []const u8, staging: []u8) void {
    const row_pitch = rowPitch(side, texel_size);
    // Indirection ranges may have left the end unaligned.
    const offset = std.mem.alignForward(u64, streamer.staged_bytes, gpu.CommandList.copy_offset_alignment);
    const row_size = side * texel_size;
    assert(offset + stagedSize(side, texel_size) <= staging.len);
    for (0..side) |row| @memcpy(staging[offset + row * row_pitch ..][0..row_size], texels[row * row_size ..][0..row_size]);
    streamer.staged_bytes = offset + stagedSize(side, texel_size);
    streamer.copies.appendAssumeCapacity(.{ .tile = .{ .texture = texture, .slot = slot, .side = side, .offset = offset, .row_pitch = row_pitch } });
}
