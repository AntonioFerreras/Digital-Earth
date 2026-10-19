//! A trained network for the `neural` render mode, in the EMLP file that
//! `python/export_weights.py` writes: an input encoding of uvwz, then a stack of
//! dense layers to linear Rec.709. Every field is a little-endian 32-bit word.
//!
//! Version 3:
//!   magic "EMLP" | version u32 = 3 | layer_count u32 | output_scale f32
//!   flags u32 (bit 0: the network takes u_mu rotated by half, frac(u_mu + 0.5))
//!   encoding u32 (`Encoding.Kind`) | encoding_word_count u32 | encoding words
//!   layer_count x (in_dim u32, out_dim u32, activation u32)
//!   per layer, in forward order: weight f32[out_dim][in_dim], bias f32[out_dim]
//!
//! Encoding words, as shaders/neural/ evaluates them:
//!   raw        none: the network takes uvwz.
//!   fourier    octaves u32, frequency f32[4] (octave-0 cycles per unit per axis).
//!   hash_grid  level_count u32, level_count x `HashLevel` (the shader's
//!              `hash_levels` verbatim), then the table: f32 feature pairs.
//!   k_planes   features u32, `k_planes_scales` x vertices u32[4] per axis, then
//!              the planes: f32 by scale, plane (axis pairs (0,1) (0,2) (0,3) (1,2)
//!              (1,3) (2,3)), row along the second axis, vertex along the first,
//!              feature.
//!
//! Versions 1 and 2 hold a plain network on raw uvwz without rotation: version 2
//! is version 3 without the flags and encoding words. Version 1 also lacks
//! `output_scale`; its networks learned the original Taichi data, whose scale
//! `legacy_output_scale` restores. `output_scale` takes the network's output to
//! the renderer's radiance: 1 for networks trained on `earth datagen` data.

const Network = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// The file, owned; the slices below view it.
words: []const u32,
output_scale: f32,
mu_rotation: bool,
encoding: Encoding,
/// In forward order.
layers: []Layer,

comptime {
    // Fields are read as raw host words.
    assert(builtin.cpu.arch.endian() == .little);
}

pub const magic: [4]u8 = "EMLP".*;
pub const version: u32 = 3;
/// 2^-ORIGINAL_ESTIMATOR_STOPS (shaders/spectral/cie.slang): the original estimator's
/// output, and so the version 1 networks', is this much below the renderer's.
pub const legacy_output_scale: f32 = std.math.exp2(7.406);
pub const layer_count_max = 64;
pub const dim_max = 4096;
/// KPLANES_SCALES in shaders/neural/kplanes.slang.
pub const k_planes_scales = 3;
pub const k_planes_plane_axes = [6][2]u32{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
/// Where `earth` looks without `--weights`, relative to the data root: the export
/// of the checkpoint the original viewer loaded.
pub const default_path = "cache/mlp/mlp_e120.emlp";
/// Where the viewer lists networks to pick from, relative to the data root.
pub const directory_path = "cache/neural";
pub const file_extension = ".emlp";
/// Above the largest study export (a 2^20-entry, 16-level hash grid, 77 MB); guards
/// against loading an unrelated file.
const file_size_max = 256 << 20;
const v2_header_word_count = 4;
const v3_header_word_count = 7;
const layer_header_word_count = 3;
const flag_mu_rotation: u32 = 1;
const octaves_max = 16;
const hash_levels_max = 32;
const k_planes_features_max = 64;

/// Applied to a layer's output. Codes match `ACTIVATIONS` in `export_weights.py`.
pub const Activation = enum(u32) {
    identity = 0,
    /// Exact GELU, `x * (1 + erf(x / sqrt(2))) / 2`.
    gelu = 1,
    relu = 2,
};

pub const Layer = struct {
    in_dim: u32,
    out_dim: u32,
    activation: Activation,
    /// Row-major `[out_dim][in_dim]`.
    weight: []const f32,
    bias: []const f32,
};

/// `hash_levels` of shaders/neural/hashgrid.slang: one level's cells per axis, then
/// where its entries start, the hash mask (table size - 1), and whether it hashes.
pub const HashLevel = extern struct {
    cells: [4]u32,
    first: u32,
    mask: u32,
    hashed: u32,
    zero: u32,
};

pub const Encoding = union(Kind) {
    raw,
    fourier: struct { octaves: u32, frequency: [4]f32 },
    hash_grid: struct { levels: []const HashLevel, table: []const f32 },
    k_planes: struct { features: u32, vertices: [k_planes_scales][4]u32, planes: []const f32 },

    pub const Kind = enum(u32) { raw, fourier, hash_grid, k_planes };

    /// The encoding's output width, the first layer's input.
    pub fn featureCount(encoding: Encoding) u32 {
        return switch (encoding) {
            .raw => 4,
            .fourier => |fourier| 4 + 8 * fourier.octaves,
            .hash_grid => |hash_grid| 2 * @as(u32, @intCast(hash_grid.levels.len)),
            .k_planes => |k_planes| k_planes_scales * k_planes.features,
        };
    }
};

pub const ParseError = error{
    Truncated,
    InvalidMagic,
    UnsupportedVersion,
    InvalidOutputScale,
    InvalidFlags,
    InvalidEncoding,
    InvalidLayerCount,
    InvalidDim,
    LayerDimMismatch,
    InvalidActivation,
    TrailingBytes,
} || Allocator.Error;

pub const LoadError = Io.Dir.ReadFileAllocError || ParseError;

pub fn load(gpa: Allocator, io: Io, dir: Io.Dir, path: []const u8) LoadError!Network {
    const file = try dir.readFileAllocOptions(io, path, gpa, .limited(file_size_max), .of(u32), null);
    errdefer gpa.free(file);
    if (file.len % 4 != 0) return error.Truncated;
    return parse(gpa, @ptrCast(file));
}

/// Validates `words` and takes ownership of them on success.
pub fn parse(gpa: Allocator, words: []const u32) ParseError!Network {
    var reader: Reader = .{ .words = words };
    if (words.len < 3) return error.Truncated;
    if (!std.mem.eql(u8, std.mem.asBytes(&words[0]), &magic)) return error.InvalidMagic;
    const file_version = words[1];
    if (file_version < 1 or file_version > version) return error.UnsupportedVersion;
    reader.position = 2;
    const layer_count = try reader.word();
    if (layer_count == 0 or layer_count > layer_count_max) return error.InvalidLayerCount;
    const output_scale: f32 = if (file_version == 1) legacy_output_scale else @bitCast(try reader.word());
    if (!(output_scale > 0 and std.math.isFinite(output_scale))) return error.InvalidOutputScale;

    var mu_rotation = false;
    var encoding: Encoding = .raw;
    if (file_version == version) {
        const flags = try reader.word();
        if (flags & ~flag_mu_rotation != 0) return error.InvalidFlags;
        mu_rotation = flags & flag_mu_rotation != 0;
        const kind = std.enums.fromInt(Encoding.Kind, try reader.word()) orelse return error.InvalidEncoding;
        const encoding_word_count = try reader.word();
        var encoding_reader: Reader = .{ .words = try reader.slice(encoding_word_count) };
        encoding = try parseEncoding(&encoding_reader, kind);
        if (encoding_reader.position != encoding_reader.words.len) return error.InvalidEncoding;
    }

    const layers = try gpa.alloc(Layer, layer_count);
    errdefer gpa.free(layers);
    const layer_table = try reader.slice(layer_count * layer_header_word_count);
    for (layers, 0..) |*layer, index| {
        const in_dim, const out_dim, const activation_code = layer_table[index * layer_header_word_count ..][0..layer_header_word_count].*;
        if (in_dim == 0 or in_dim > dim_max) return error.InvalidDim;
        if (out_dim == 0 or out_dim > dim_max) return error.InvalidDim;
        const expected_in_dim = if (index == 0) encoding.featureCount() else layers[index - 1].out_dim;
        if (in_dim != expected_in_dim) return error.LayerDimMismatch;
        layer.* = .{
            .in_dim = in_dim,
            .out_dim = out_dim,
            .activation = std.enums.fromInt(Activation, activation_code) orelse return error.InvalidActivation,
            .weight = undefined,
            .bias = undefined,
        };
    }
    for (layers) |*layer| {
        layer.weight = try reader.floats(layer.in_dim * layer.out_dim);
        layer.bias = try reader.floats(layer.out_dim);
    }
    if (reader.position != words.len) return error.TrailingBytes;
    return .{ .words = words, .output_scale = output_scale, .mu_rotation = mu_rotation, .encoding = encoding, .layers = layers };
}

fn parseEncoding(reader: *Reader, kind: Encoding.Kind) error{ Truncated, InvalidEncoding }!Encoding {
    switch (kind) {
        .raw => return .raw,
        .fourier => {
            const octaves = try reader.word();
            if (octaves == 0 or octaves > octaves_max) return error.InvalidEncoding;
            const frequency = (try reader.floats(4))[0..4].*;
            for (frequency) |value| if (!std.math.isFinite(value)) return error.InvalidEncoding;
            return .{ .fourier = .{ .octaves = octaves, .frequency = frequency } };
        },
        .hash_grid => {
            const level_count = try reader.word();
            if (level_count == 0 or level_count > hash_levels_max) return error.InvalidEncoding;
            const level_words = try reader.slice(level_count * @sizeOf(HashLevel) / 4);
            const levels: []const HashLevel = @ptrCast(level_words);
            var entry_count: u64 = 0;
            for (levels) |level| {
                const table_size = @as(u64, level.mask) + 1;
                if (!std.math.isPowerOfTwo(table_size) or level.mask != levels[0].mask) return error.InvalidEncoding;
                if (level.first != entry_count or level.hashed > 1 or level.zero != 0) return error.InvalidEncoding;
                var vertex_count: u64 = 1;
                for (level.cells) |cells| {
                    if (cells == 0 or cells > std.math.maxInt(u16)) return error.InvalidEncoding;
                    vertex_count *= cells + 1;
                }
                const hashed = vertex_count > table_size;
                if (level.hashed != @intFromBool(hashed)) return error.InvalidEncoding;
                entry_count += if (hashed) table_size else vertex_count;
                if (entry_count > std.math.maxInt(u32)) return error.InvalidEncoding;
            }
            return .{ .hash_grid = .{ .levels = levels, .table = try reader.floats(2 * entry_count) } };
        },
        .k_planes => {
            const features = try reader.word();
            if (features == 0 or features % 4 != 0 or features > k_planes_features_max) return error.InvalidEncoding;
            var vertices: [k_planes_scales][4]u32 = undefined;
            var value_count: u64 = 0;
            for (&vertices) |*scale| {
                scale.* = (try reader.slice(4))[0..4].*;
                for (scale.*) |count| if (count < 2 or count > std.math.maxInt(u16)) return error.InvalidEncoding;
                for (k_planes_plane_axes) |axes| value_count += @as(u64, scale[axes[0]]) * scale[axes[1]] * features;
            }
            return .{ .k_planes = .{ .features = features, .vertices = vertices, .planes = try reader.floats(value_count) } };
        },
    }
}

/// Sequential reads of whole words, failing with `Truncated` past the end.
const Reader = struct {
    words: []const u32,
    position: usize = 0,

    fn slice(reader: *Reader, count: u64) error{Truncated}![]const u32 {
        if (count > reader.words.len - reader.position) return error.Truncated;
        const result = reader.words[reader.position..][0..@intCast(count)];
        reader.position += @intCast(count);
        return result;
    }

    fn word(reader: *Reader) error{Truncated}!u32 {
        return (try reader.slice(1))[0];
    }

    fn floats(reader: *Reader, count: u64) error{Truncated}![]const f32 {
        return @ptrCast(try reader.slice(count));
    }
};

pub fn deinit(network: *Network, gpa: Allocator) void {
    gpa.free(network.layers);
    gpa.free(network.words);
    network.* = undefined;
}

/// Weights, biases, and encoding tables.
pub fn parameterCount(network: *const Network) u64 {
    var count: u64 = switch (network.encoding) {
        .raw, .fourier => 0,
        .hash_grid => |hash_grid| hash_grid.table.len,
        .k_planes => |k_planes| k_planes.planes.len,
    };
    for (network.layers) |layer| count += layer.weight.len + layer.bias.len;
    return count;
}

const testing = std.testing;

/// A version 2 file: 2 -> 3 (gelu) -> 1 (relu), parameters counting up from 1.
fn testWordsV2() [v2_header_word_count + 2 * layer_header_word_count + (2 * 3 + 3) + (3 * 1 + 1)]u32 {
    var words: [v2_header_word_count + 2 * layer_header_word_count + (2 * 3 + 3) + (3 * 1 + 1)]u32 = undefined;
    words[0..v2_header_word_count].* = .{ @bitCast(magic), 2, 2, @bitCast(@as(f32, 0.5)) };
    words[4..10].* = .{ 2, 3, @backingInt(Activation.gelu), 3, 1, @backingInt(Activation.relu) };
    for (words[10..], 1..) |*word, value| word.* = @bitCast(@as(f32, @floatFromInt(value)));
    return words;
}

fn parseCopy(words: []const u32) ParseError!Network {
    return parse(testing.allocator, try testing.allocator.dupe(u32, words));
}

/// `parse`, freeing the words on failure as `load` does.
fn expectParseError(expected: ParseError, words: []const u32) !void {
    const copy = try testing.allocator.dupe(u32, words);
    defer testing.allocator.free(copy);
    try testing.expectError(expected, parse(testing.allocator, copy));
}

test "version 2 files hold a plain network on raw uvwz, laid out in forward order" {
    // A 2-input network does not fit raw uvwz.
    var words = testWordsV2();
    try expectParseError(error.LayerDimMismatch, &words);
    words[4] = 4;
    try expectParseError(error.Truncated, &words);

    const one: u32 = @bitCast(@as(f32, 1));
    const weights: [19]u32 = @splat(one);
    const extended = [_]u32{ @bitCast(magic), 2, 2, @bitCast(@as(f32, 0.5)), 4, 3, 1, 3, 1, 2 } ++ weights;
    var network = try parseCopy(&extended);
    defer network.deinit(testing.allocator);
    try testing.expectEqual(0.5, network.output_scale);
    try testing.expect(!network.mu_rotation);
    try testing.expectEqual(.raw, std.meta.activeTag(network.encoding));
    try testing.expectEqual(2, network.layers.len);
    try testing.expectEqual(Activation.gelu, network.layers[0].activation);
    try testing.expectEqual(12, network.layers[0].weight.len);
    try testing.expectEqual(1, network.layers[1].bias.len);
    try testing.expectEqual(4 * 3 + 3 + 3 + 1, network.parameterCount());
}

test "version 1 files take the legacy output scale" {
    const parameters: [4 * 3 + 3]u32 = @splat(0);
    const words = [_]u32{ @bitCast(magic), 1, 1, 4, 3, 2 } ++ parameters;
    var network = try parseCopy(&words);
    defer network.deinit(testing.allocator);
    try testing.expectEqual(legacy_output_scale, network.output_scale);
    try testing.expectEqual(3, network.layers[0].out_dim);
}

/// A version 3 file with `encoding_words` and one relu layer to 3 outputs.
fn testWordsV3(gpa: Allocator, flags: u32, kind: Encoding.Kind, encoding_words: []const u32, in_dim: u32) ![]u32 {
    var words: std.ArrayList(u32) = .empty;
    errdefer words.deinit(gpa);
    try words.appendSlice(gpa, &.{ @bitCast(magic), version, 1, @bitCast(@as(f32, 1)), flags, @backingInt(kind), @intCast(encoding_words.len) });
    try words.appendSlice(gpa, encoding_words);
    try words.appendSlice(gpa, &.{ in_dim, 3, @backingInt(Activation.relu) });
    try words.appendNTimes(gpa, 0, in_dim * 3 + 3);
    return words.toOwnedSlice(gpa);
}

test "version 3 files name their encoding" {
    const gpa = testing.allocator;
    const fourier = try testWordsV3(gpa, 1, .fourier, &.{ 2, @bitCast(@as(f32, 0.5)), @bitCast(@as(f32, 0.5)), @bitCast(@as(f32, 0.5)), @bitCast(@as(f32, 0.5)) }, 20);
    var network = try parse(gpa, fourier);
    try testing.expect(network.mu_rotation);
    try testing.expectEqual(2, network.encoding.fourier.octaves);
    network.deinit(gpa);

    // Two levels sharing a 16-entry table: 1^4 cells stay dense (16 vertices), 2^4 hash.
    const level_words = [_]u32{ 1, 1, 1, 1, 0, 15, 0, 0, 2, 2, 2, 2, 16, 15, 1, 0 };
    var hash_words: std.ArrayList(u32) = .empty;
    defer hash_words.deinit(gpa);
    try hash_words.append(gpa, 2);
    try hash_words.appendSlice(gpa, &level_words);
    try hash_words.appendNTimes(gpa, 0, 2 * 32);
    var hash = try parse(gpa, try testWordsV3(gpa, 0, .hash_grid, hash_words.items, 4));
    try testing.expectEqual(2, hash.encoding.hash_grid.levels.len);
    try testing.expectEqual(64, hash.encoding.hash_grid.table.len);
    try testing.expectEqual(64 + 4 * 3 + 3, hash.parameterCount());
    hash.deinit(gpa);
    hash_words.items[1 + 8 + 6] = 0;
    const misflagged = try testWordsV3(gpa, 0, .hash_grid, hash_words.items, 4);
    defer gpa.free(misflagged);
    try expectParseError(error.InvalidEncoding, misflagged);

    var plane_words: std.ArrayList(u32) = .empty;
    defer plane_words.deinit(gpa);
    try plane_words.append(gpa, 4);
    for (0..k_planes_scales) |_| try plane_words.appendSlice(gpa, &.{ 2, 3, 4, 2 });
    // Per scale: 2*3 + 2*4 + 2*2 + 3*4 + 3*2 + 4*2 = 44 vertices of 4 features.
    try plane_words.appendNTimes(gpa, 0, k_planes_scales * 44 * 4);
    var planes = try parse(gpa, try testWordsV3(gpa, 0, .k_planes, plane_words.items, 12));
    try testing.expectEqual(k_planes_scales * 44 * 4, planes.encoding.k_planes.planes.len);
    planes.deinit(gpa);

    _ = plane_words.pop();
    const short = try testWordsV3(gpa, 0, .k_planes, plane_words.items, 12);
    defer gpa.free(short);
    try expectParseError(error.Truncated, short);
}

test "parse rejects malformed files" {
    const gpa = testing.allocator;
    const valid = try testWordsV3(gpa, 0, .raw, &.{}, 4);
    defer gpa.free(valid);
    const words = try gpa.dupe(u32, valid);
    defer gpa.free(words);

    try expectParseError(error.Truncated, valid[0 .. valid.len - 1]);
    try expectParseError(error.Truncated, valid[0..2]);
    const longer = try std.mem.concat(gpa, u32, &.{ valid, &.{0} });
    defer gpa.free(longer);
    try expectParseError(error.TrailingBytes, longer);

    const cases = [_]struct { usize, u32, ParseError }{
        .{ 0, 0, error.InvalidMagic },
        .{ 1, version + 1, error.UnsupportedVersion },
        .{ 2, 0, error.InvalidLayerCount },
        .{ 3, @bitCast(@as(f32, 0)), error.InvalidOutputScale },
        .{ 4, 2, error.InvalidFlags },
        .{ 5, 9, error.InvalidEncoding },
        .{ 7, 5, error.LayerDimMismatch },
        .{ 8, 0, error.InvalidDim },
        .{ 9, 7, error.InvalidActivation },
    };
    for (cases) |case| {
        @memcpy(words, valid);
        words[case[0]] = case[1];
        try expectParseError(case[2], words);
    }
}
