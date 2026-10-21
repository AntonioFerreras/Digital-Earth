//! Downloads the tile files that python/tiles/publish.py uploads into <data>/tiles/.
//! A file is written under a `.part` name and takes its own name only once its size
//! and SHA-256 match the manifest, so a file under its own name is complete, and an
//! interrupted download resumes from its `.part`.

const std = @import("std");
const Io = std.Io;
const http = std.http;
const Sha256 = std.crypto.hash.sha2.Sha256;

const log = std.log.scoped(.assets);

pub const dataset_url = "https://huggingface.co/datasets/AntonioFerreras/digital-earth-tiles/resolve/main/";
const manifest_name = "manifest.zon";
const manifest_size_max = 64 * 1024;
/// Each attempt resumes from the bytes already written, so a dropped connection costs
/// one retry rather than the file.
const attempts_max = 8;
const progress_interval_ns = std.time.ns_per_s;
/// The CDN's signed redirect URLs run to a few kilobytes.
const redirect_buffer_size = 16 * 1024;
const chunk_size = 1 << 20;

pub const Manifest = struct {
    files: []const File,

    pub const File = struct {
        /// Relative to <data>/tiles/, with `/` separators.
        path: []const u8,
        size: u64,
        /// Lowercase hex.
        sha256: []const u8,
    };
};

pub fn run(gpa: std.mem.Allocator, io: Io, tiles_dir: Io.Dir) !void {
    var client: http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const manifest = try fetchManifest(&client, arena.allocator());

    var progress: Progress = .{ .start = .now(io, .awake), .last_print = .now(io, .awake) };
    for (manifest.files) |file| progress.total += file.size;
    const buffers = try gpa.create(Buffers);
    defer gpa.destroy(buffers);
    for (manifest.files) |file| {
        var attempt: usize = 1;
        while (true) : (attempt += 1) {
            downloadFile(&client, io, tiles_dir, file, buffers, &progress) catch |err| switch (err) {
                error.ChecksumMismatch, error.OutOfMemory, error.Canceled => return err,
                else => if (attempt < attempts_max) {
                    log.warn("{s}: {t}, retrying", .{ file.path, err });
                    continue;
                } else return err,
            };
            break;
        }
    }
    std.debug.print("\r\x1b[Ktiles: {d:.2} GB in {d} files, all verified\n", .{
        gigabytes(progress.total), manifest.files.len,
    });
}

fn fetchManifest(client: *http.Client, arena: std.mem.Allocator) !Manifest {
    var body: Io.Writer.Allocating = .init(arena);
    const result = client.fetch(.{
        .location = .{ .url = dataset_url ++ manifest_name },
        .response_writer = &body.writer,
    }) catch |err| {
        log.err("{s}: {t}", .{ dataset_url ++ manifest_name, err });
        return err;
    };
    if (result.status != .ok) {
        log.err("{s}: HTTP {d}", .{ dataset_url ++ manifest_name, @backingInt(result.status) });
        return error.DownloadFailed;
    }
    if (body.written().len > manifest_size_max) return error.InvalidManifest;
    const source = try body.toOwnedSliceSentinel(0);
    var diagnostics: std.zon.parse.Diagnostics = undefined;
    return std.zon.parse.fromSlice(Manifest, .{
        .gpa = arena,
        .arena = arena,
        .source = source,
        .diagnostics = &diagnostics,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseZon => {
            diagnostics.log(manifest_name);
            return error.InvalidManifest;
        },
    };
}

const Buffers = struct {
    chunk: [chunk_size]u8,
    file: [64 * 1024]u8,
    transfer: [64 * 1024]u8,
    redirect: [redirect_buffer_size]u8,
};

const Progress = struct {
    start: Io.Timestamp,
    last_print: Io.Timestamp,
    /// Bytes of every file, and those on disk, resumed ones included.
    total: u64 = 0,
    done: u64 = 0,
    /// Bytes this run received, for the rate.
    received: u64 = 0,

    fn advance(progress: *Progress, io: Io, path: []const u8, bytes: u64) void {
        progress.done += bytes;
        progress.received += bytes;
        const now: Io.Timestamp = .now(io, .awake);
        if (progress.last_print.durationTo(now).toNanoseconds() < progress_interval_ns) return;
        progress.last_print = now;
        const seconds = @as(f64, @floatFromInt(progress.start.durationTo(now).toNanoseconds())) / std.time.ns_per_s;
        const rate = @as(f64, @floatFromInt(progress.received)) / seconds;
        const left = @as(f64, @floatFromInt(progress.total - progress.done)) / rate;
        std.debug.print("\r\x1b[Ktiles: {d:.2} / {d:.2} GB, {d:.0} MB/s, {d:.0} min left, {s}", .{
            gigabytes(progress.done), gigabytes(progress.total), rate / 1e6, left / 60.0, path,
        });
    }
};

fn gigabytes(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

fn downloadFile(client: *http.Client, io: Io, dir: Io.Dir, file: Manifest.File, buffers: *Buffers, progress: *Progress) !void {
    // A retry counts the file's bytes on disk again.
    const done_before = progress.done;
    errdefer progress.done = done_before;
    if (dir.statFile(io, file.path, .{})) |stat| {
        if (stat.size == file.size) {
            progress.done += file.size;
            return;
        }
        log.warn("{s}: {d} bytes, expected {d}; downloading again", .{ file.path, stat.size, file.size });
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }
    if (std.fs.path.dirnamePosix(file.path)) |parent| try dir.createDirPath(io, parent);
    var part_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const part_path = try std.fmt.bufPrint(&part_path_buffer, "{s}.part", .{file.path});

    const part = try dir.createFile(io, part_path, .{ .read = true, .truncate = false });
    defer part.close(io);
    var hash: Sha256 = .init(.{});
    var have = try part.length(io);
    if (have > file.size) {
        try part.setLength(io, 0);
        have = 0;
    }
    var part_reader = part.reader(io, &buffers.file);
    var hashed: u64 = 0;
    while (hashed < have) {
        const length: usize = @intCast(@min(have - hashed, chunk_size));
        try part_reader.interface.readSliceAll(buffers.chunk[0..length]);
        hash.update(buffers.chunk[0..length]);
        hashed += length;
    }
    progress.done += have;

    if (have < file.size) {
        var url_buffer: [dataset_url.len + std.fs.max_path_bytes]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, dataset_url ++ "{s}", .{file.path});
        var range_buffer: [64]u8 = undefined;
        const range = try std.fmt.bufPrint(&range_buffer, "bytes={d}-", .{have});
        var request = try client.request(.GET, try .parse(url), .{
            // Kept across the redirect to the CDN, which serves the bytes.
            .extra_headers = &.{.{ .name = "range", .value = range }},
            // Ranges count bytes of the file, not of a compressed stream.
            .headers = .{ .accept_encoding = .omit },
        });
        defer request.deinit();
        try request.sendBodiless();
        var response = try request.receiveHead(&buffers.redirect);
        switch (response.head.status) {
            .partial_content => {},
            // The server ignored the range, so the body starts at byte 0.
            .ok => {
                try part.setLength(io, 0);
                progress.done -= have;
                have = 0;
                hash = .init(.{});
            },
            else => |status| {
                log.err("{s}: HTTP {d}", .{ url, @backingInt(status) });
                return error.DownloadFailed;
            },
        }
        var part_writer = part.writer(io, &buffers.file);
        try part_writer.seekTo(have);
        const body = response.reader(&buffers.transfer);
        while (have < file.size) {
            const length = body.readSliceShort(buffers.chunk[0..@intCast(@min(file.size - have, chunk_size))]) catch |err| switch (err) {
                error.ReadFailed => return response.bodyErr().?,
            };
            if (length == 0) return error.DownloadFailed;
            hash.update(buffers.chunk[0..length]);
            try part_writer.interface.writeAll(buffers.chunk[0..length]);
            have += length;
            progress.advance(io, file.path, length);
        }
        try part_writer.interface.flush();
    }

    const digest = std.fmt.bytesToHex(hash.finalResult(), .lower);
    if (!std.mem.eql(u8, &digest, file.sha256)) {
        log.err("{s}: SHA-256 {s}, expected {s}; deleted it", .{ file.path, &digest, file.sha256 });
        try dir.deleteFile(io, part_path);
        return error.ChecksumMismatch;
    }
    try dir.rename(part_path, dir, file.path, io);
}
