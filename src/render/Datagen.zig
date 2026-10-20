//! `earth datagen` on the GPU (shaders/modes/datagen.slang): the symmetric scene's
//! linear Rec.709 radiance at batches of uvwz points, each the mean of `path_count`
//! spectral paths, in the normalized estimator's scale (`datagen_format`).
//!
//! A point's paths split into chunks of at most `chunk_paths_max`, one thread group
//! each. A batch records its groups as dispatches of about `dispatch_paths` paths, so
//! no dispatch holds the GPU long enough to stall the desktop or trip a watchdog,
//! submits them at once, and reads every group's sum back in a single blocking read;
//! the host then adds each point's chunks in f64. Paths are keyed by (point index,
//! path index, seed), so on one device a record depends on the seed alone.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const Luts = @import("../assets/Luts.zig");
const datagen_format = @import("../assets/datagen_format.zig");
const uniforms = @import("uniforms.zig");
const scene = @import("scene.zig");
const GpuLuts = @import("GpuLuts.zig");
const GpuEarthMaps = @import("GpuEarthMaps.zig");
const TilePool = @import("TilePool.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const Datagen = @This();

shared: Renderer.Shared,
kernel: ModeKernel,
datagen_param: gpu.Kernel.Param,
points_param: gpu.Kernel.Param,
options: Options,
chunk: Chunking,
/// Room for `batch_points_max` uvwz points.
points: gpu.Buffer,
/// One texel per thread group of a batch: its paths' rgb sum and sun-disk hits.
accum: gpu.Texture,
/// Host copy of `accum`.
group_sums: [][4]f32,
/// The uvwz of a batch, packed for upload.
staged_points: [][4]f32,
/// Points per `evaluate` call.
batch_points_max: u32,

/// GROUP_THREADS in shaders/modes/datagen.slang.
const group_threads = 256;
/// Each thread adds at most 64 paths in f32 before the group's pairwise tree.
const chunk_paths_max = 64 * group_threads;
/// Thread groups per `accum` row.
const accum_width = 1024;
/// Paths per dispatch; at the measured path cost a dispatch stays far below 200 ms.
const dispatch_paths = 1 << 26;
/// Paths per batch: enough GPU time to hide the host work between batches.
const batch_paths = 1 << 31;
/// Bounds the host and GPU memory of a batch at few paths per point.
const batch_points_cap = 1 << 21;

pub const Options = struct {
    /// Paths averaged per point.
    path_count: u32,
    seed: u32,
    /// Leaves the sun disk out of escaping rays; hits are counted either way.
    exclude_sun_disk: bool,
};

const Chunking = struct {
    paths: u32,
    count: u32,

    fn init(path_count: u32) Chunking {
        const paths = @min(path_count, chunk_paths_max);
        return .{ .paths = paths, .count = std.math.divCeil(u32, path_count, paths) catch unreachable };
    }
};

/// What a batch saw of the sun disk.
pub const SunDiskHits = struct {
    /// Points with at least one path that escaped into the sun disk.
    points: u64 = 0,
    paths: u64 = 0,

    pub fn add(hits: *SunDiskHits, other: SunDiskHits) void {
        hits.points += other.points;
        hits.paths += other.paths;
    }
};

pub const InitError = gpu.Kernel.LayoutError || error{OutOfMemory};

/// Compiles the kernel and uploads `luts`, which the caller may free afterwards.
pub fn init(gpa: std.mem.Allocator, device: gpu.Device, luts: *const Luts, options: Options) InitError!Datagen {
    assert(options.path_count > 0);
    var gpu_luts: GpuLuts = try .init(gpa, device, luts);
    errdefer gpu_luts.deinit(gpa);
    var earth_maps: GpuEarthMaps = try .placeholder(device);
    errdefer earth_maps.deinit();
    var tiles: TilePool = try .placeholder(gpa, device);
    errdefer tiles.deinit(gpa);
    const kernel: ModeKernel = try .init(device, "modes/datagen", "datagen_main", &.{});
    errdefer kernel.deinit();
    assert(std.mem.eql(u32, &kernel.kernel.thread_group_size, &.{ group_threads, 1, 1 }));
    try kernel.kernel.checkLayout(uniforms.DatagenUniforms, "datagen");
    // The symmetric scene samples no Earth map.
    const flags: uniforms.SceneFlags = @bitCast(scene.Preset.symmetric.sceneUniforms().config[0]);
    assert(!flags.earth_maps);

    const chunk: Chunking = .init(options.path_count);
    const point_paths = @as(u64, chunk.count) * chunk.paths;
    const batch_points_max: u32 = @intCast(std.math.clamp(batch_paths / point_paths, 1, batch_points_cap));
    const rows = std.math.divCeil(u32, batch_points_max * chunk.count, accum_width) catch unreachable;

    const points: gpu.Buffer = try .init(device, .{
        .size = @as(u64, batch_points_max) * @sizeOf([4]f32),
        .element_size = @sizeOf([4]f32),
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = "datagen points",
    }, null);
    errdefer points.deinit();
    const accum: gpu.Texture = try .init(device, .{
        .kind = .@"2d",
        .format = .rgba32_float,
        .width = accum_width,
        .height = rows,
        .depth_or_layers = 1,
        .usage = .{ .unordered_access = true, .copy_source = true },
        .label = "datagen group sums",
    });
    errdefer accum.deinit();
    const group_sums = try gpa.alloc([4]f32, @as(usize, accum_width) * rows);
    errdefer gpa.free(group_sums);
    const staged_points = try gpa.alloc([4]f32, batch_points_max);
    errdefer gpa.free(staged_points);
    const datagen_param = try kernel.kernel.param("datagen");
    const points_param = try kernel.kernel.param("points");
    return .{
        .shared = .{ .device = device, .luts = gpu_luts, .network = null, .lut4d = null, .earth_maps = earth_maps, .tiles = tiles },
        .kernel = kernel,
        .datagen_param = datagen_param,
        .points_param = points_param,
        .options = options,
        .chunk = chunk,
        .points = points,
        .accum = accum,
        .group_sums = group_sums,
        .staged_points = staged_points,
        .batch_points_max = batch_points_max,
    };
}

/// Waits for the GPU, then releases everything `init` created.
pub fn deinit(datagen: *Datagen, gpa: std.mem.Allocator) void {
    datagen.shared.device.waitIdle() catch {};
    gpa.free(datagen.staged_points);
    gpa.free(datagen.group_sums);
    datagen.accum.deinit();
    datagen.points.deinit();
    datagen.kernel.deinit();
    datagen.shared.tiles.deinit(gpa);
    datagen.shared.earth_maps.deinit();
    datagen.shared.luts.deinit(gpa);
    datagen.* = undefined;
}

/// Fills the `rgb` of each record from its `uvwz`; `records[i]` is point
/// `first_point + i` of the run, which keys its paths. Blocks until the GPU is done.
pub fn evaluate(datagen: *Datagen, first_point: u32, records: []datagen_format.Record) gpu.Error!SunDiskHits {
    assert(records.len > 0 and records.len <= datagen.batch_points_max);
    assert(records.len - 1 <= std.math.maxInt(u32) - first_point);
    const device = datagen.shared.device;
    const staged = datagen.staged_points[0..records.len];
    for (staged, records) |*point, record| point.* = record.uvwz;
    try datagen.points.upload(device, 0, std.mem.sliceAsBytes(staged));

    const chunk = datagen.chunk;
    const group_count: u32 = @intCast(records.len * chunk.count);
    const row_count = std.math.divCeil(u32, group_count, accum_width) catch unreachable;
    const dispatch_rows: u32 = @max(1, dispatch_paths / (accum_width * chunk.paths));
    const exclude: uniforms.DatagenFlags = .{ .exclude_sun_disk = datagen.options.exclude_sun_disk };
    var frame: Renderer.Frame = .{
        .uniforms = std.mem.zeroes(uniforms.FrameUniforms),
        .scene = scene.Preset.symmetric.sceneUniforms(),
        .accum = datagen.accum,
    };
    frame.uniforms.frame = .{ datagen.options.seed, accum_width, datagen.accum.desc.height, 0 };

    const command_list: gpu.CommandList = try .begin(device);
    const recorded: gpu.Error!void = record: {
        var row: u32 = 0;
        while (row < row_count) : (row += dispatch_rows) {
            const batch_uniforms: uniforms.DatagenUniforms = .{
                .batch = .{ first_point, datagen.options.path_count, chunk.paths, chunk.count },
                .dispatch = .{ row, group_count, @bitCast(exclude), 0 },
            };
            datagen.kernel.dispatchGroups(command_list, &datagen.shared, &frame, &.{
                .{ .param = datagen.datagen_param, .resource = .{ .data = std.mem.asBytes(&batch_uniforms) } },
                .{ .param = datagen.points_param, .resource = .{ .buffer = datagen.points } },
            }, .{ accum_width, @min(dispatch_rows, row_count - row), 1 }) catch |err| break :record err;
        }
        break :record {};
    };
    try command_list.submit(null);
    try recorded;
    try datagen.accum.read(device, 0, std.mem.sliceAsBytes(datagen.group_sums));

    var hits: SunDiskHits = .{};
    const path_count: f64 = @floatFromInt(datagen.options.path_count);
    for (records, 0..) |*record, index| {
        var sum: [4]f64 = @splat(0);
        for (datagen.group_sums[index * chunk.count ..][0..chunk.count]) |group_sum| {
            for (&sum, group_sum) |*total, value| total.* += value;
        }
        for (&record.rgb, sum[0..3]) |*channel, total| channel.* = @floatCast(total / path_count);
        const sun_disk_paths: u64 = @intFromFloat(sum[3]);
        hits.paths += sun_disk_paths;
        hits.points += @intFromBool(sun_disk_paths > 0);
    }
    return hits;
}

/// Where the uvwz points of a run come from.
pub const Sampling = union(enum) {
    uniform: Uniform,
    /// The vertices of a 4D table of this many vertices along (u, v, w, z), one record
    /// per vertex in `datagen_format.gridPoint` order.
    grid: [4]u32,
    /// Each record's uvwz, re-evaluated in order.
    points: []const datagen_format.Record,

    /// uvwz ~ U[0, 1)^4, except that a `sunset_fraction` of the points scale v (the
    /// sun-angle coordinate) by `sunset_scale`: the original data_prep.py recipe,
    /// which spent that share of its files on sunsets.
    pub const Uniform = struct {
        random: std.Random.DefaultPrng,
        sunset_fraction: f32,
        sunset_scale: f32,
        /// Index of the next point; points are drawn in order.
        next: u64 = 0,

        pub fn init(seed: u64, sunset_fraction: f32, sunset_scale: f32) Uniform {
            return .{ .random = .init(seed), .sunset_fraction = sunset_fraction, .sunset_scale = sunset_scale };
        }
    };

    /// Points `first .. first + records.len` into the records' `uvwz`. Uniform
    /// points must be asked for in order.
    pub fn fill(sampling: *Sampling, first: u64, records: []datagen_format.Record) void {
        switch (sampling.*) {
            .uniform => |*uniform| {
                assert(first == uniform.next);
                const random = uniform.random.random();
                for (records) |*record| {
                    for (&record.uvwz) |*coordinate| coordinate.* = random.float(f32);
                    if (random.float(f32) < uniform.sunset_fraction) record.uvwz[1] *= uniform.sunset_scale;
                }
                uniform.next += records.len;
            },
            .grid => |shape| for (records, first..) |*record, index| {
                record.uvwz = datagen_format.gridPoint(shape, index);
            },
            .points => |points| for (records, points[first..][0..records.len]) |*record, point| {
                record.uvwz = point.uvwz;
            },
        }
    }
};
