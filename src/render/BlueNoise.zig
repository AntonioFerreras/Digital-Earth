//! A blue noise tile for the cloud bench's jitters (modes/cloud_bench.slang): per
//! channel a dither array by void-and-cluster (Ulichney 1993), ranking the texels so
//! that the texels below any threshold spread evenly, without clusters or voids. Tiled
//! over the screen, a value per pixel then carries no low frequencies, which the eye
//! forgives far sooner than white noise's clumps.

const std = @import("std");
const assert = std.debug.assert;

pub const size = 64;
pub const channels = 4;
const texel_count = size * size;
/// Of the Gaussian that measures how crowded a texel is, in texels; its support is cut
/// where it falls below a thousandth.
const sigma = 1.9;
const radius = 7;
/// The fraction of texels the relaxed starting pattern holds.
const seed_fraction = 0.1;

/// The tile, `channels` bytes a texel, rows of `size`.
pub fn generate(seed: u64) [texel_count * channels]u8 {
    var tile: [texel_count * channels]u8 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    for (0..channels) |channel| {
        var ranks: [texel_count]u16 = undefined;
        rank(prng.random(), &ranks);
        for (&ranks, 0..) |r, i| tile[i * channels + channel] = @intCast(r * 256 / texel_count);
    }
    return tile;
}

/// Each texel's rank in [0, texel_count): the order in which the void-and-cluster
/// filling places it.
fn rank(random: std.Random, ranks: *[texel_count]u16) void {
    var energy: [texel_count]f32 = @splat(0);
    var pattern: [texel_count]bool = @splat(false);
    // A random start, relaxed until its tightest cluster is also its largest void.
    const seed_count = @as(usize, @intFromFloat(seed_fraction * texel_count));
    var order: [texel_count]u16 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    random.shuffle(u16, &order);
    for (order[0..seed_count]) |i| place(&energy, &pattern, i, true);
    while (true) {
        const cluster = extreme(&energy, &pattern, true, true);
        place(&energy, &pattern, cluster, false);
        const void_texel = extreme(&energy, &pattern, false, false);
        place(&energy, &pattern, void_texel, true);
        if (void_texel == cluster) break;
    }
    // Phase one empties the start from its tightest clusters down, ranking backwards.
    var phase_energy = energy;
    var phase_pattern = pattern;
    var next: u16 = @intCast(seed_count);
    while (next > 0) {
        next -= 1;
        const cluster = extreme(&phase_energy, &phase_pattern, true, true);
        place(&phase_energy, &phase_pattern, cluster, false);
        ranks[cluster] = next;
    }
    // Phase two fills the largest voids from the start on. Past half full the usual
    // inversion changes nothing: the crowding of the empty texels is the kernel's
    // constant sum less that of the full ones, so their tightest cluster is the full
    // ones' largest void.
    next = @intCast(seed_count);
    while (next < texel_count) : (next += 1) {
        const void_texel = extreme(&energy, &pattern, false, false);
        place(&energy, &pattern, void_texel, true);
        ranks[void_texel] = next;
    }
}

/// Fills or empties texel `i` and adds or removes its crowding around it, on the torus.
fn place(energy: *[texel_count]f32, pattern: *[texel_count]bool, i: usize, fill: bool) void {
    assert(pattern[i] != fill);
    pattern[i] = fill;
    const sign: f32 = if (fill) 1 else -1;
    const x: i32 = @intCast(i % size);
    const y: i32 = @intCast(i / size);
    var dy: i32 = -radius;
    while (dy <= radius) : (dy += 1) {
        var dx: i32 = -radius;
        while (dx <= radius) : (dx += 1) {
            const distance_squared: f32 = @floatFromInt(dx * dx + dy * dy);
            const weight = @exp(-distance_squared / (2 * sigma * sigma));
            const tx: usize = @intCast(@mod(x + dx, size));
            const ty: usize = @intCast(@mod(y + dy, size));
            energy[tx + size * ty] += sign * weight;
        }
    }
}

/// Of the texels whose state is `filled`, the most crowded with `most`, else the least.
fn extreme(energy: *const [texel_count]f32, pattern: *const [texel_count]bool, filled: bool, most: bool) usize {
    var best: usize = texel_count;
    for (energy, pattern, 0..) |e, p, i| {
        if (p != filled) continue;
        if (best == texel_count or (if (most) e > energy[best] else e < energy[best])) best = i;
    }
    assert(best < texel_count);
    return best;
}

test "each channel ranks every texel once, evenly over the byte range" {
    const tile = generate(1);
    for (0..channels) |channel| {
        var counts: [256]u32 = @splat(0);
        for (0..texel_count) |i| counts[tile[i * channels + channel]] += 1;
        for (counts) |count| try std.testing.expectEqual(@as(u32, texel_count / 256), count);
    }
}

test "neighbouring texels differ more than white noise's would" {
    // Two independent uniform values differ by a third on average; blue noise's
    // neighbours are anticorrelated and differ by more.
    const tile = generate(1);
    for (0..channels) |channel| {
        var total: f64 = 0;
        for (0..texel_count) |i| {
            const x = i % size;
            const y = i / size;
            const here: f64 = @floatFromInt(tile[i * channels + channel]);
            const right: f64 = @floatFromInt(tile[((x + 1) % size + size * y) * channels + channel]);
            const below: f64 = @floatFromInt(tile[(x + size * ((y + 1) % size)) * channels + channel]);
            total += @abs(here - right) + @abs(here - below);
        }
        const mean_difference = total / (2 * texel_count) / 255;
        try std.testing.expect(mean_difference > 0.37);
    }
}
