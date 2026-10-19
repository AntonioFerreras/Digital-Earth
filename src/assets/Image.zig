//! 8-bit images decoded by stb_image (PNG, JPEG) and PNGs encoded by
//! stb_image_write. Decoded pixels live in stb's malloc heap until `deinit`.

const Image = @This();

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const stb = @import("stb");

width: u32,
height: u32,
/// Bytes per texel in `pixels`.
channels: u32,
/// Channels stored in the encoded file; differs from `channels` when decoding forced a count.
file_channels: u32,
/// Row-major, top row first.
pixels: []u8,

pub const DecodeError = error{ ImageTooLarge, ImageDecode };

/// Decodes `encoded` with `channels_forced` channels per texel (1-4), or the
/// file's own count when null. stb expands palettes and narrows 16-bit samples.
/// On `error.ImageDecode`, `stb.stbi_failure_reason()` names the cause on this thread.
pub fn decode(encoded: []const u8, channels_forced: ?u32) DecodeError!Image {
    if (channels_forced) |channels| assert(channels >= 1 and channels <= 4);
    const encoded_size = std.math.cast(c_int, encoded.len) orelse return error.ImageTooLarge;
    var width: c_int = 0;
    var height: c_int = 0;
    var file_channels: c_int = 0;
    const pixels = stb.stbi_load_from_memory(
        encoded.ptr,
        encoded_size,
        &width,
        &height,
        &file_channels,
        @intCast(channels_forced orelse 0),
    ) orelse return error.ImageDecode;
    assert(width > 0 and height > 0);
    assert(file_channels >= 1 and file_channels <= 4);

    const channels: u32 = channels_forced orelse @intCast(file_channels);
    const size = @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * channels;
    return .{
        .width = @intCast(width),
        .height = @intCast(height),
        .channels = channels,
        .file_channels = @intCast(file_channels),
        .pixels = pixels[0..size],
    };
}

pub fn deinit(image: *Image) void {
    stb.stbi_image_free(image.pixels.ptr);
    image.* = undefined;
}

pub const WritePngError = error{ ImageTooLarge, PngEncode } || Io.File.OpenError || Io.File.Writer.Error;

/// Writes row-major, top-row-first `pixels` as a PNG file.
pub fn writePng(
    io: Io,
    dir: Io.Dir,
    sub_path: []const u8,
    width: u32,
    height: u32,
    channels: u32,
    pixels: []const u8,
) WritePngError!void {
    assert(channels >= 1 and channels <= 4);
    assert(pixels.len == @as(usize, width) * height * channels);
    const stride = std.math.cast(c_int, @as(usize, width) * channels) orelse return error.ImageTooLarge;
    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var file_writer = file.writer(io, &buffer);
    const encoded = stb.stbi_write_png_to_func(
        writePngCallback,
        &file_writer,
        std.math.cast(c_int, width) orelse return error.ImageTooLarge,
        std.math.cast(c_int, height) orelse return error.ImageTooLarge,
        @intCast(channels),
        pixels.ptr,
        stride,
    );
    if (file_writer.err) |err| return err;
    if (encoded == 0) return error.PngEncode;
    try file_writer.flush();
}

fn writePngCallback(context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void {
    const file_writer: *Io.File.Writer = @ptrCast(@alignCast(context));
    const bytes: [*]const u8 = @ptrCast(data);
    // A failure is kept in `file_writer.err` for `writePng` to return.
    file_writer.interface.writeAll(bytes[0..@intCast(size)]) catch {};
}

test "PNG round trip keeps pixels and forces channel counts" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const width = 3;
    const height = 2;
    var rgb: [width * height * 3]u8 = undefined;
    for (&rgb, 0..) |*value, index| value.* = @intCast(index * 13);
    try writePng(io, tmp.dir, "rgb.png", width, height, 3, &rgb);
    const encoded = try tmp.dir.readFileAlloc(io, "rgb.png", std.testing.allocator, .limited(1 << 16));
    defer std.testing.allocator.free(encoded);

    var native: Image = try .decode(encoded, null);
    defer native.deinit();
    try std.testing.expectEqual(width, native.width);
    try std.testing.expectEqual(height, native.height);
    try std.testing.expectEqual(3, native.channels);
    try std.testing.expectEqualSlices(u8, &rgb, native.pixels);

    var rgba: Image = try .decode(encoded, 4);
    defer rgba.deinit();
    try std.testing.expectEqual(3, rgba.file_channels);
    try std.testing.expectEqual(4, rgba.channels);
    try std.testing.expectEqualSlices(u8, rgb[3..6], rgba.pixels[4..7]);
    try std.testing.expectEqual(255, rgba.pixels[7]);
}

test "decode rejects bytes that are not an image" {
    try std.testing.expectError(error.ImageDecode, decode("not a png", null));
}
