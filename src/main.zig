//! `earth`: the interactive viewer and the headless commands.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const heap = @import("heap.zig");
const gpu = @import("gpu.zig");
const viewer = @import("viewer.zig");
const render = @import("render.zig");
const assets = @import("assets.zig");

const log = std.log.scoped(.earth);

const usage =
    \\usage: earth [command] [options]
    \\
    \\commands:
    \\  view            interactive viewer (default)
    \\  render          offline render
    \\  datagen         MLP training data
    \\  bake-textures   decode Earth maps into the raw cache
    \\  bake-probes     bake the cloud bench's probe field into a file, for validation
    \\  download        fetch the streamed terrain tiles (13 GB) into <data>/tiles/
    \\
    \\options:
    \\  --backend vulkan|metal|d3d12   GPU backend (default: the platform's choice)
    \\  --validation                   GPU validation layers (always on in Debug builds)
    \\  --data <dir>                   holds LUT/, textures/, cache/, presets/, screenshots/,
    \\                                 and the viewer's config.zon (default: the working directory)
    \\  --config <file.zon>            initial settings, e.g. <data>/presets/florida.zon
    \\                                 (default: built-in)
    \\  --weights <file.emlp>          network of the neural mode, from python/export_weights.py
    \\                                 (default: <data>/
++ assets.Network.default_path ++ ", if present; the viewer\n" ++
    \\                                 also lists <data>/
++ assets.Network.directory_path ++ "/*" ++ assets.Network.file_extension ++ ")\n" ++
    \\  --lut <file.edgn>              scattering table of the bruneton mode, from
    \\                                 `datagen --sampling grid` (default: <data>/
++ assets.Lut4d.default_path ++ ", if present)\n" ++
    \\  --cloud-volume <file.ecld>     volume of the cloud_bench mode, from python/clouds/convert_vdb.py
    \\                                 (default: <data>/
++ assets.CloudVolume.default_path ++ ", if present)\n" ++
    \\  --texture-quality q1|q2        Earth map set to render or bake (default: the config's)
    \\  --help                         this text
    \\
    \\render options:
    \\  --mode debug|path_tracer|raymarch|neural|bruneton|cloud_bench
    \\                                 render mode (default: the config's)
    \\  --preset textured|symmetric    scene preset (default: the config's)
    \\  --width <px> --height <px>     image size (default: 1920 x 1080)
    \\  --spp <n>                      samples per pixel (default: 64)
    \\  --out <file.png>               output image (default: render.png)
    \\  --out-linear <file.pfm>        also the mean linear Rec.709 samples, before display
    \\  --single-scattering            light scatters once only (one bounce; raymarch drops
    \\                                 its multiple-scattering table), for accuracy studies
    \\  --no-clouds                    leave the cloud shell out of presets that have one
    \\  --clouds-only                  the clouds alone: no air and a black surface
    \\  --bench-view reference|approximation|split|error
    \\                                 what the cloud_bench mode shows (default: the config's)
    \\  --seed <n>                     first frame of the random-number key (default: 0);
    \\                                 renders seeded --spp apart are independent
    \\  --probes <file.eprb>           the cloud bench's probe field, from bake-probes, instead
    \\                                 of the probe cache, which is left alone
    \\  --jobs <file.zon>              several images on one renderer: a list of
    \\                                 .{ .config, .out, .out_linear, .bench_view, .spp }, each
    \\                                 field optional, the mode and preset the command's
    \\
    \\bake-probes options (the config's walks, without the probe cache; see TESTING.md):
    \\  --probes <file.eprb>           field to go on from (default: none)
    \\  --out <file.eprb>              the baked field (required)
    \\  --half <file.eprb>             also the field at half the walks, for the split-half noise
    \\
    \\datagen options (the symmetric scene's radiance at uvwz points):
    \\  --sampling uniform|grid        point set (default: uniform)
    \\  --count <n>                    uniform: points to draw (required); --points: use the first n
    \\  --sunset-fraction <f>          uniform: share of points with v scaled down (default: 0.1)
    \\  --sunset-scale <f>             uniform: that scale of v, the sun angle (default: 0.4)
    \\  --grid <nu,nv,nw,nz>           grid: vertices of a 4D table spanning the whole domain,
    \\                                 u fastest; nw splits into ground and sky rays
    \\  --points <file.edgn>           re-evaluate the uvwz of these records instead
    \\  --paths <n>                    spectral paths averaged per point (default: 16384)
    \\  --seed <n>                     keys the points and the paths (default: 0)
    \\  --exclude-sun-disk             leave the sun disk out of rays that escape unscattered
    \\  --out <file.edgn>              output records (default: datagen.edgn)
    \\
    \\bake-textures options (writes <data>/cache/):
    \\  --source <dir>                 NASA PNG directory (default: <data>/textures)
    \\
;

const Command = enum { view, render, datagen, @"bake-textures", @"bake-probes", download };

const Arguments = struct {
    command: Command = .view,
    backend: gpu.Device.Backend = .default,
    validation: bool = builtin.mode == .debug,
    data_path: []const u8 = ".",
    config_path: ?[]const u8 = null,
    weights_path: ?[]const u8 = null,
    lut_path: ?[]const u8 = null,
    cloud_volume_path: ?[]const u8 = null,
    /// `--probes`, for both render and bake-probes.
    probes_path: ?[]const u8 = null,
    /// `--half`, for bake-probes.
    half_path: ?[]const u8 = null,
    /// `--jobs`, for render.
    jobs_path: ?[]const u8 = null,
    texture_quality: ?assets.EarthMaps.Quality = null,
    texture_source_path: ?[]const u8 = null,
    /// `--out`, whose default depends on the command.
    out_path: ?[]const u8 = null,
    /// `--seed`, for both render and datagen.
    seed: u32 = 0,
    render: RenderArguments = .{},
    datagen: DatagenArguments = .{},
};

const RenderArguments = struct {
    mode: ?render.Renderer.ModeKind = null,
    preset: ?render.scene.Preset = null,
    size: [2]u32 = .{ 1920, 1080 },
    sample_count: u32 = 64,
    out_linear_path: ?[]const u8 = null,
    single_scattering: bool = false,
    no_clouds: bool = false,
    clouds_only: bool = false,
    bench_view: ?render.scene.CloudBench.View = null,
};

const DatagenArguments = struct {
    sampling: DatagenSampling = .uniform,
    point_count: ?u32 = null,
    sunset_fraction: f32 = 0.1,
    sunset_scale: f32 = 0.4,
    grid: ?[4]u32 = null,
    points_path: ?[]const u8 = null,
    path_count: u32 = 16384,
    exclude_sun_disk: bool = false,
};

const DatagenSampling = enum { uniform, grid };

/// Bounds the streaming before an offline render when tiles keep arriving.
const tile_warmup_frames_max = 512;

/// Generous bound on an offline render; keeps a typo from queueing hours of GPU work.
const render_sample_count_max = 1 << 16;
const render_size_max = 16384;

const ParseError = error{ InvalidArguments, HelpRequested };

pub fn main(init: std.process.Init.Minimal) u8 {
    const gpa = heap.allocator();
    defer if (heap.deinit() == .leak) log.err("memory leaked", .{});
    var threaded: Io.Threaded = .init(gpa, .{ .argv0 = .init(init.args), .environ = init.environ });
    defer threaded.deinit();
    const io = threaded.io();

    var args_arena: std.heap.ArenaAllocator = .init(gpa);
    defer args_arena.deinit();
    const args = init.args.toSlice(args_arena.allocator()) catch {
        log.err("out of memory reading the command line", .{});
        return 1;
    };
    const arguments = parseArguments(args[1..]) catch |err| {
        std.debug.print("{s}", .{usage});
        return switch (err) {
            error.HelpRequested => 0,
            error.InvalidArguments => 2,
        };
    };
    switch (arguments.command) {
        .view => view(gpa, io, arguments) catch |err| {
            log.err("viewer failed: {t}", .{err});
            return 1;
        },
        .render => renderImage(gpa, io, arguments) catch |err| {
            log.err("render failed: {t}", .{err});
            return 1;
        },
        .@"bake-probes" => bakeProbes(gpa, io, arguments) catch |err| {
            log.err("bake-probes failed: {t}", .{err});
            return 1;
        },
        .@"bake-textures" => bakeTextures(gpa, io, arguments) catch |err| {
            log.err("bake-textures failed: {t}", .{err});
            return 1;
        },
        .download => downloadTiles(gpa, io, arguments) catch |err| {
            log.err("download failed: {t}; running it again resumes", .{err});
            return 1;
        },
        .datagen => generateData(gpa, io, arguments) catch |err| {
            log.err("datagen failed: {t}", .{err});
            return 1;
        },
    }
    return 0;
}

fn parseArguments(args: []const [:0]const u8) ParseError!Arguments {
    var arguments: Arguments = .{};
    var command_seen = false;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--help")) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--validation")) {
            arguments.validation = true;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const backend = try parseEnum(gpu.Device.Backend, arg, try nextValue(args, &index));
            if (backend != .vulkan and backend != .metal and backend != .d3d12) {
                log.err("unknown backend '{t}'", .{backend});
                return error.InvalidArguments;
            }
            arguments.backend = backend;
        } else if (std.mem.eql(u8, arg, "--data")) {
            arguments.data_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--weights")) {
            arguments.weights_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--lut")) {
            arguments.lut_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--cloud-volume")) {
            arguments.cloud_volume_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--probes")) {
            arguments.probes_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--jobs")) {
            arguments.jobs_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--half")) {
            arguments.half_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--bench-view")) {
            arguments.render.bench_view = try parseEnum(render.scene.CloudBench.View, arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--config")) {
            arguments.config_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--mode")) {
            arguments.render.mode = try parseEnum(render.Renderer.ModeKind, arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--preset")) {
            arguments.render.preset = try parseEnum(render.scene.Preset, arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--width")) {
            arguments.render.size[0] = try parseCount(arg, try nextValue(args, &index), render_size_max);
        } else if (std.mem.eql(u8, arg, "--height")) {
            arguments.render.size[1] = try parseCount(arg, try nextValue(args, &index), render_size_max);
        } else if (std.mem.eql(u8, arg, "--spp")) {
            arguments.render.sample_count = try parseCount(arg, try nextValue(args, &index), render_sample_count_max);
        } else if (std.mem.eql(u8, arg, "--out")) {
            arguments.out_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--out-linear")) {
            arguments.render.out_linear_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--single-scattering")) {
            arguments.render.single_scattering = true;
        } else if (std.mem.eql(u8, arg, "--no-clouds")) {
            arguments.render.no_clouds = true;
        } else if (std.mem.eql(u8, arg, "--clouds-only")) {
            arguments.render.clouds_only = true;
        } else if (std.mem.eql(u8, arg, "--texture-quality")) {
            arguments.texture_quality = try parseEnum(assets.EarthMaps.Quality, arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--sampling")) {
            arguments.datagen.sampling = try parseEnum(DatagenSampling, arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--count")) {
            arguments.datagen.point_count = try parseCount(arg, try nextValue(args, &index), std.math.maxInt(u32));
        } else if (std.mem.eql(u8, arg, "--sunset-fraction")) {
            arguments.datagen.sunset_fraction = try parseUnit(arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--sunset-scale")) {
            arguments.datagen.sunset_scale = try parseUnit(arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--grid")) {
            arguments.datagen.grid = try parseGrid(arg, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--points")) {
            arguments.datagen.points_path = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--paths")) {
            arguments.datagen.path_count = try parseCount(arg, try nextValue(args, &index), std.math.maxInt(u32));
        } else if (std.mem.eql(u8, arg, "--seed")) {
            arguments.seed = std.fmt.parseInt(u32, try nextValue(args, &index), 10) catch {
                log.err("{s} takes an integer in [0, {d}]", .{ arg, std.math.maxInt(u32) });
                return error.InvalidArguments;
            };
        } else if (std.mem.eql(u8, arg, "--exclude-sun-disk")) {
            arguments.datagen.exclude_sun_disk = true;
        } else if (std.mem.eql(u8, arg, "--source")) {
            arguments.texture_source_path = try nextValue(args, &index);
        } else if (!command_seen and !std.mem.startsWith(u8, arg, "-")) {
            arguments.command = try parseEnum(Command, "command", arg);
            command_seen = true;
        } else {
            log.err("unexpected argument '{s}'", .{arg});
            return error.InvalidArguments;
        }
    }
    return arguments;
}

/// The value after the option at `index.*`, which advances past it.
fn nextValue(args: []const [:0]const u8, index: *usize) ParseError![]const u8 {
    const option = args[index.*];
    index.* += 1;
    if (index.* < args.len) return args[index.*];
    log.err("{s} needs a value", .{option});
    return error.InvalidArguments;
}

fn parseEnum(comptime T: type, what: []const u8, value: []const u8) ParseError!T {
    return std.meta.stringToEnum(T, value) orelse {
        log.err("unknown {s} '{s}'", .{ what, value });
        return error.InvalidArguments;
    };
}

/// A count in [1, max].
fn parseCount(option: []const u8, value: []const u8, max: u32) ParseError!u32 {
    const count = std.fmt.parseInt(u32, value, 10) catch 0;
    if (count >= 1 and count <= max) return count;
    log.err("{s} takes an integer in [1, {d}], not '{s}'", .{ option, max, value });
    return error.InvalidArguments;
}

/// A number in [0, 1].
fn parseUnit(option: []const u8, value: []const u8) ParseError!f32 {
    const number = std.fmt.parseFloat(f32, value) catch std.math.nan(f32);
    if (number >= 0 and number <= 1) return number;
    log.err("{s} takes a number in [0, 1], not '{s}'", .{ option, value });
    return error.InvalidArguments;
}

/// Four comma-separated vertex counts that `datagen_format.gridShapeValid` accepts.
fn parseGrid(option: []const u8, value: []const u8) ParseError![4]u32 {
    var shape: [4]u32 = undefined;
    var sizes = std.mem.splitScalar(u8, value, ',');
    for (&shape) |*size| size.* = std.fmt.parseInt(u32, sizes.next() orelse "", 10) catch 0;
    if (sizes.next() == null and assets.datagen_format.gridShapeValid(shape)) return shape;
    log.err("{s} takes four vertex counts nu,nv,nw,nz of at least 2, nw even and at least 4, " ++
        "multiplying to at most {d}, not '{s}'", .{ option, std.math.maxInt(u32), value });
    return error.InvalidArguments;
}

/// Directories of the shaders that build.zig installs under <prefix>/share/earth beside
/// <prefix>/bin: ours, then libearth_native's.
const ShaderPaths = struct {
    shaders: [:0]u8,
    native: [:0]u8,

    fn init(gpa: std.mem.Allocator, io: Io) !ShaderPaths {
        const exe_dir = try std.process.executableDirPathAlloc(io, gpa);
        defer gpa.free(exe_dir);
        const shaders = try std.fs.path.joinZ(gpa, &.{ exe_dir, "..", "share", "earth", "shaders" });
        errdefer gpa.free(shaders);
        const native = try std.fs.path.joinZ(gpa, &.{ shaders, "native" });
        return .{ .shaders = shaders, .native = native };
    }

    fn deinit(paths: ShaderPaths, gpa: std.mem.Allocator) void {
        gpa.free(paths.native);
        gpa.free(paths.shaders);
    }
};

/// `--config`, relative to the working directory like any command-line path, with
/// `--texture-quality` applied. The caller owns the result.
fn initialConfig(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !viewer.Config {
    var config: viewer.Config = if (arguments.config_path) |path|
        viewer.Config.load(gpa, io, .cwd(), path) catch |err| {
            log.err("loading {s} failed: {t}", .{ path, err });
            return err;
        }
    else
        try (viewer.Config{}).clone(gpa);
    if (arguments.texture_quality) |quality| config.texture_quality = quality;
    return config;
}

/// A mode's input file: `path` from the command line (relative to the working
/// directory), else `default_path` under the data root when it exists; without either
/// the mode reports the missing file when it starts. The mode reads it then, so a
/// broken file fails that mode alone.
fn modeInput(io: Io, data_root: assets.DataRoot, path: ?[]const u8, default_path: []const u8) ?render.Renderer.InputFile {
    if (path) |given| return .{ .io = io, .dir = .cwd(), .path = given };
    data_root.dir.access(io, default_path, .{}) catch return null;
    return .{ .io = io, .dir = data_root.dir, .path = default_path };
}

fn view(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    var config = try initialConfig(gpa, io, arguments);
    defer config.deinit(gpa);
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);
    // Without baked maps the viewer still runs the presets that need none.
    var cache_dir: ?Io.Dir = openCacheDir(io, data_root) catch null;
    defer if (cache_dir) |*dir| dir.close(io);
    var tile_files = openTileFiles(io, data_root);
    defer if (tile_files) |*files| files.close();
    const shader_paths: ShaderPaths = try .init(gpa, io);
    defer shader_paths.deinit(gpa);
    var earth_viewer: viewer.Viewer = try .init(gpa, io, .{
        .backend = arguments.backend,
        .validation = arguments.validation,
        .shader_search_paths = &.{ shader_paths.shaders, shader_paths.native },
        .data_root = data_root,
        .config = config,
        .earth_map_cache = if (cache_dir) |dir| .{ .io = io, .dir = dir } else null,
        .network = modeInput(io, data_root, arguments.weights_path, assets.Network.default_path),
        .lut4d = modeInput(io, data_root, arguments.lut_path, assets.Lut4d.default_path),
        .cloud_volume = modeInput(io, data_root, arguments.cloud_volume_path, assets.CloudVolume.default_path),
        .tile_files = if (tile_files) |*files| files else null,
    });
    defer earth_viewer.deinit();
    try earth_viewer.run();
}

/// One image of `render`: its config (default: the command's), outputs and samples.
const RenderJob = struct {
    config: ?[]const u8 = null,
    out: []const u8 = "render.png",
    out_linear: ?[]const u8 = null,
    bench_view: ?render.scene.CloudBench.View = null,
    spp: u32 = 64,
};

/// Headless: renders `--spp` frames on a device without a window, then writes the
/// display image as PNG and prints the timing; with `--jobs`, each job's image in
/// turn on the one renderer, which spares the setup of a process per image.
fn renderImage(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    const options = arguments.render;
    var config = try initialConfig(gpa, io, arguments);
    defer config.deinit(gpa);
    applyRenderOverrides(&config, options, null);
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);
    var jobs_arena: std.heap.ArenaAllocator = .init(gpa);
    defer jobs_arena.deinit();
    const single = [_]RenderJob{.{ .out = arguments.out_path orelse "render.png", .out_linear = options.out_linear_path, .spp = options.sample_count }};
    const jobs: []const RenderJob = if (arguments.jobs_path) |path| try loadJobs(gpa, jobs_arena.allocator(), io, path) else &single;

    const start = Io.Timestamp.now(io, .awake);
    const device = try initHeadlessDevice(gpa, io, arguments);
    defer device.deinit();
    var tile_files = openTileFiles(io, data_root);
    defer if (tile_files) |*files| files.close();
    const probe_field: render.Renderer.ProbeField = if (arguments.probes_path) |path| .{ .file = .{ .io = io, .dir = .cwd(), .path = path } } else .cache;
    var renderer = try initOfflineRenderer(gpa, io, arguments, &config, options.size, data_root, device, if (tile_files) |*files| files else null, probe_field);
    defer renderer.deinit();
    const pixels = try gpa.alloc(u8, @as(usize, options.size[0]) * options.size[1] * 4);
    defer gpa.free(pixels);
    log.info("setup {d:.0} ms", .{milliseconds(start.durationTo(Io.Timestamp.now(io, .awake)))});

    for (jobs) |job| {
        if (job.spp == 0 or job.spp > render_sample_count_max) {
            log.err("{s}: spp must lie in [1, {d}]", .{ job.out, render_sample_count_max });
            return error.InvalidArguments;
        }
        var job_config = if (job.config) |path| viewer.Config.load(gpa, io, .cwd(), path) catch |err| {
            log.err("loading {s} failed: {t}", .{ path, err });
            return err;
        } else try config.clone(gpa);
        defer job_config.deinit(gpa);
        applyRenderOverrides(&job_config, options, job.bench_view);
        if (job_config.mode != config.mode or job_config.preset != config.preset) {
            log.err("{s}: every job renders the command's mode and preset", .{job.out});
            return error.InvalidArguments;
        }
        renderer.cloud_bench = job_config.cloud_bench;
        renderer.restart();
        const frame_view = job_config.camera.view(job_config.sunDirection());
        try streamTiles(io, device, &renderer, frame_view, job_config.tone());
        const render_start = Io.Timestamp.now(io, .awake);
        for (0..job.spp) |_| try renderer.render(frame_view, job_config.tone());
        try device.waitIdle();
        const render_ms = milliseconds(render_start.durationTo(Io.Timestamp.now(io, .awake)));
        std.debug.assert(renderer.sampleCount() == job.spp);

        try renderer.image.read(device, 0, pixels);
        try assets.Image.writePng(io, .cwd(), job.out, options.size[0], options.size[1], 4, pixels);
        if (job.out_linear) |path| try writeLinear(gpa, io, &renderer, path);
        std.debug.print("auto exposure {d:.4} -> {s}\n", .{ render.Post.autoExposure(job_config.camera.cosZenith(job_config.sunDirection())), job.out });
        std.debug.print("{t} {t} {d}x{d}, {d} spp: {d:.1} ms ({d:.3} ms/spp) -> {s}\n", .{
            job_config.mode, job_config.preset, options.size[0],                              options.size[1],
            job.spp,         render_ms,         render_ms / @as(f64, @floatFromInt(job.spp)), job.out,
        });
    }
}

fn loadJobs(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: Io, path: []const u8) ![]const RenderJob {
    const source = try Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 20), .of(u8), 0);
    defer gpa.free(source);
    var diagnostics: std.zon.parse.Diagnostics = undefined;
    return std.zon.parse.fromSlice([]const RenderJob, .{
        .gpa = gpa,
        .arena = arena,
        .source = source,
        .diagnostics = &diagnostics,
    }) catch |err| {
        if (err == error.ParseZon) diagnostics.log(path);
        return err;
    };
}

/// The command line's choices over a config's.
fn applyRenderOverrides(config: *viewer.Config, options: RenderArguments, bench_view: ?render.scene.CloudBench.View) void {
    if (options.mode) |mode| config.mode = mode;
    if (options.preset) |preset| config.preset = preset;
    if (bench_view orelse options.bench_view) |view_kind| config.cloud_bench.view = view_kind;
}

/// Streams in the tiles the view wants before its samples accumulate on them.
fn streamTiles(io: Io, device: gpu.Device, renderer: *render.Renderer, frame_view: render.Renderer.View, tone: render.Post.Tone) !void {
    const warmup_start = Io.Timestamp.now(io, .awake);
    var warmup_frames: u32 = 0;
    while (!renderer.tilesSettled() and warmup_frames < tile_warmup_frames_max) : (warmup_frames += 1) {
        try renderer.render(frame_view, tone);
        // Requests come back only once their frame is done; ahead of it they would not.
        try device.waitIdle();
    }
    if (warmup_frames > 0) {
        renderer.tiles_frozen = true;
        renderer.restart();
        const warmup_ms = warmup_start.durationTo(Io.Timestamp.now(io, .awake)).toMilliseconds();
        log.info("tiles streamed in {d} frames, {d} ms", .{ warmup_frames, warmup_ms });
    }
}

/// Bakes the cloud bench's probes to the config's walks, going on from the field in
/// `--probes` or from none, into `--out`, without the probe cache, and prints the
/// bake's time: its frames until the GPU is done, not the setup or the writes. With
/// `--half`, the field at half the walks is written there on the way (TESTING.md).
fn bakeProbes(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    const out_path = arguments.out_path orelse {
        log.err("bake-probes needs --out", .{});
        return error.InvalidArguments;
    };
    var config = try initialConfig(gpa, io, arguments);
    defer config.deinit(gpa);
    config.mode = .cloud_bench;
    config.cloud_bench.view = .approximation;
    // Without a viewer to keep responsive, longer dispatches overlap more photon rounds;
    // the field does not depend on them.
    config.cloud_bench.probe_rounds_per_frame = @max(config.cloud_bench.probe_rounds_per_frame, bake_rounds_per_frame);
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);
    const device = try initHeadlessDevice(gpa, io, arguments);
    defer device.deinit();
    const start_field: ?render.Renderer.InputFile = if (arguments.probes_path) |path| .{ .io = io, .dir = .cwd(), .path = path } else null;
    var renderer = try initOfflineRenderer(gpa, io, arguments, &config, bake_size, data_root, device, null, .{ .file = start_field });
    defer renderer.deinit();
    const mode = &renderer.mode.cloud_bench;
    const frame_view = config.camera.view(config.sunDirection());
    const walks = config.cloud_bench.probe_walks;
    // A frame that bakes nothing lays the probes out, so that the timings leave the
    // layout's build out.
    renderer.cloud_bench.probe_walks = 0;
    try renderer.render(frame_view, config.tone());
    try device.waitIdle();
    var bake_ms: f64 = 0;
    var half_ms: ?f64 = null;
    const phases = [_]struct { path: ?[]const u8, walks: u32 }{
        .{ .path = arguments.half_path, .walks = walks / 2 },
        .{ .path = out_path, .walks = walks },
    };
    for (phases, 0..) |phase, index| {
        const field_path = phase.path orelse continue;
        renderer.cloud_bench.probe_walks = phase.walks;
        const phase_start = Io.Timestamp.now(io, .awake);
        try renderer.render(frame_view, config.tone());
        while (mode.bakeProgress() != null) try renderer.render(frame_view, config.tone());
        try device.waitIdle();
        bake_ms += milliseconds(phase_start.durationTo(Io.Timestamp.now(io, .awake)));
        if (index == 0) half_ms = bake_ms;
        try mode.saveProbes(gpa, io, .cwd(), field_path);
    }
    std.debug.print("bake-probes: {d} walks in {d:.1} ms", .{ mode.probe_walks, bake_ms });
    if (half_ms) |ms| std.debug.print(", {d} at {d:.1} ms -> {s}", .{ walks / 2, ms, arguments.half_path.? });
    std.debug.print(" -> {s}\n", .{out_path});
}

/// The bake renders an image only to drive its frames.
const bake_size = [2]u32{ 16, 16 };
/// Short enough for a dispatch not to stall a display GPU for long.
const bake_rounds_per_frame = 32;

/// The renderer of the offline commands, for `config` at `size`.
fn initOfflineRenderer(
    gpa: std.mem.Allocator,
    io: Io,
    arguments: Arguments,
    config: *const viewer.Config,
    size: [2]u32,
    data_root: assets.DataRoot,
    device: gpu.Device,
    tile_files: ?*const render.TilePool.Files,
    probe_field: render.Renderer.ProbeField,
) !render.Renderer {
    var lut_dir = try data_root.openSubdir(io, .lut, .{});
    defer lut_dir.close(io);
    var luts: assets.Luts = try .load(gpa, io, lut_dir);
    defer luts.deinit(gpa);
    // Only a preset that samples the maps needs their cache.
    const scene_flags: render.uniforms.SceneFlags = @bitCast(config.preset.sceneUniforms().config[0]);
    var cache_dir: ?Io.Dir = if (scene_flags.earth_maps) try openCacheDir(io, data_root) else null;
    defer if (cache_dir) |*dir| dir.close(io);
    return render.Renderer.init(gpa, device, &luts, .{
        .size = size,
        .mode = config.mode,
        .preset = config.preset,
        .single_scattering = arguments.render.single_scattering,
        .clouds = config.clouds and !arguments.render.no_clouds,
        .water = config.water,
        .cloud_scattering = config.cloud_scattering,
        .cloud_bench = config.cloud_bench,
        .clouds_only = config.clouds_only or arguments.render.clouds_only,
        .seed = arguments.seed,
        .texture_quality = config.texture_quality,
        .earth_map_cache = if (cache_dir) |dir| .{ .io = io, .dir = dir } else null,
        .network = modeInput(io, data_root, arguments.weights_path, assets.Network.default_path),
        .lut4d = modeInput(io, data_root, arguments.lut_path, assets.Lut4d.default_path),
        .cloud_volume = modeInput(io, data_root, arguments.cloud_volume_path, assets.CloudVolume.default_path),
        .probe_field = probe_field,
        .tile_files = tile_files,
        .tiles = config.tiles,
    });
}

/// A device without a window, for the offline commands.
fn initHeadlessDevice(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !gpu.Device {
    const shader_paths: ShaderPaths = try .init(gpa, io);
    defer shader_paths.deinit(gpa);
    return .init(.{
        .backend = arguments.backend,
        .validation = arguments.validation,
        .shader_search_paths = &.{ shader_paths.shaders, shader_paths.native },
        .shader_cache_path = null,
    });
}

/// Seconds between progress lines.
const datagen_progress_interval_s = 10;

/// Streams the radiance at the requested uvwz points to `--out`, logging progress,
/// throughput, and how many points saw the sun disk.
fn generateData(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    const options = arguments.datagen;
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);

    const points_bytes: ?[]align(@alignOf(assets.datagen_format.Record)) u8 = if (options.points_path) |path|
        Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .unlimited, .of(assets.datagen_format.Record), null) catch |err| {
            log.err("points {s}: {t}", .{ path, err });
            return err;
        }
    else
        null;
    defer if (points_bytes) |bytes| gpa.free(bytes);
    var sampling: render.Datagen.Sampling, const point_count: u32 = if (points_bytes) |bytes| points: {
        const points = assets.datagen_format.parse(bytes) catch |err| {
            log.err("points {s}: {t}", .{ options.points_path.?, err });
            return err;
        };
        break :points .{ .{ .points = points }, @intCast(@min(options.point_count orelse points.len, points.len)) };
    } else switch (options.sampling) {
        .uniform => .{
            .{ .uniform = .init(arguments.seed, options.sunset_fraction, options.sunset_scale) },
            options.point_count orelse {
                log.err("uniform sampling needs --count", .{});
                return error.InvalidArguments;
            },
        },
        .grid => grid: {
            const shape = options.grid orelse {
                log.err("grid sampling needs --grid", .{});
                return error.InvalidArguments;
            };
            break :grid .{ .{ .grid = shape }, shape[0] * shape[1] * shape[2] * shape[3] };
        },
    };
    if (point_count == 0) {
        log.err("no points to evaluate", .{});
        return error.InvalidArguments;
    }

    const device = try initHeadlessDevice(gpa, io, arguments);
    defer device.deinit();
    var datagen = init: {
        var lut_dir = try data_root.openSubdir(io, .lut, .{});
        defer lut_dir.close(io);
        var luts: assets.Luts = try .load(gpa, io, lut_dir);
        defer luts.deinit(gpa);
        break :init try render.Datagen.init(gpa, device, &luts, .{
            .path_count = options.path_count,
            .seed = arguments.seed,
            .exclude_sun_disk = options.exclude_sun_disk,
        });
    };
    defer datagen.deinit(gpa);

    const out_path = arguments.out_path orelse "datagen.edgn";
    const file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var writer: assets.datagen_format.Writer = try .init(io, file, &write_buffer);
    const records = try gpa.alloc(assets.datagen_format.Record, @min(datagen.batch_points_max, point_count));
    defer gpa.free(records);

    const start = Io.Timestamp.now(io, .awake);
    var last_report = start;
    var sun_disk_hits: render.Datagen.SunDiskHits = .{};
    var done: u32 = 0;
    while (done < point_count) {
        const batch = records[0..@min(records.len, point_count - done)];
        sampling.fill(done, batch);
        sun_disk_hits.add(try datagen.evaluate(done, batch));
        try writer.write(batch);
        done += @intCast(batch.len);

        const now = Io.Timestamp.now(io, .awake);
        if (done == point_count or last_report.durationTo(now).toSeconds() >= datagen_progress_interval_s) {
            last_report = now;
            const seconds = milliseconds(start.durationTo(now)) / std.time.ms_per_s;
            const points_per_s = @as(f64, @floatFromInt(done)) / seconds;
            const remaining_s = @as(f64, @floatFromInt(point_count - done)) / points_per_s;
            std.debug.print("datagen: {d}/{d} points, {d:.0} s, {d:.4} points/s, {e:.3} paths/s, {d:.0} s left\n", .{
                done,         point_count,                                                seconds,
                points_per_s, points_per_s * @as(f64, @floatFromInt(options.path_count)), remaining_s,
            });
        }
    }
    try writer.finish();
    std.debug.print("datagen: {d} points x {d} paths, seed {d} -> {s}; sun disk seen by {d} points ({d} paths){s}\n", .{
        point_count,                                        options.path_count,   arguments.seed,
        out_path,                                           sun_disk_hits.points, sun_disk_hits.paths,
        if (options.exclude_sun_disk) ", excluded" else "",
    });
}

/// The baked Earth maps' directory, or log lines naming the command that creates it.
fn openCacheDir(io: Io, data_root: assets.DataRoot) Io.Dir.OpenError!Io.Dir {
    return data_root.openSubdir(io, .cache, .{}) catch |err| {
        log.err("bake the Earth maps with `earth bake-textures`", .{});
        return err;
    };
}

fn downloadTiles(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);
    var tiles_dir = try data_root.createSubdir(io, .tiles);
    defer tiles_dir.close(io);
    try assets.tile_download.run(gpa, io, tiles_dir);
}

/// The cube-face tile files under `<data>/tiles`, or null when there are none; broken
/// files are logged and left out.
fn openTileFiles(io: Io, data_root: assets.DataRoot) ?render.TilePool.Files {
    var dir = data_root.dir.openDir(io, assets.DataRoot.Subdir.tiles.path(), .{}) catch {
        log.info("no tile files under {s}; `earth download` fetches them", .{assets.DataRoot.Subdir.tiles.path()});
        return null;
    };
    defer dir.close(io);
    return render.TilePool.Files.open(io, dir) catch |err| {
        log.err("tile files under {s}: {t}; rendering with the Earth maps", .{ assets.DataRoot.Subdir.tiles.path(), err });
        return null;
    };
}

/// The mean of the accumulated samples as a PFM (RGB f32, bottom row first, as `accum`).
fn writeLinear(gpa: std.mem.Allocator, io: Io, renderer: *const render.Renderer, path: []const u8) !void {
    const image_size = renderer.size();
    const texel_count = @as(usize, image_size[0]) * image_size[1];
    const texels = try gpa.alloc([4]f32, texel_count);
    defer gpa.free(texels);
    try renderer.accum.read(renderer.shared.device, 0, std.mem.sliceAsBytes(texels));
    const rgb = try gpa.alloc([3]f32, texel_count);
    defer gpa.free(rgb);
    const sample_count: f32 = @floatFromInt(renderer.sampleCount());
    for (rgb, texels) |*target, texel| target.* = .{ texel[0] / sample_count, texel[1] / sample_count, texel[2] / sample_count };

    var header_buffer: [64]u8 = undefined;
    // A negative scale marks little-endian floats.
    const header = std.fmt.bufPrint(&header_buffer, "PF\n{d} {d}\n-1.0\n", .{ image_size[0], image_size[1] }) catch unreachable;
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [1 << 16]u8 = undefined;
    var file_writer = file.writer(io, &buffer);
    try file_writer.interface.writeAll(header);
    try file_writer.interface.writeAll(std.mem.sliceAsBytes(rgb));
    try file_writer.interface.flush();
}

/// Decodes the NASA PNGs of the texture quality from `--source` into <data>/cache/.
fn bakeTextures(gpa: std.mem.Allocator, io: Io, arguments: Arguments) !void {
    var config = try initialConfig(gpa, io, arguments);
    defer config.deinit(gpa);
    const data_root: assets.DataRoot = try .open(io, arguments.data_path);
    defer data_root.close(io);
    var source_dir = if (arguments.texture_source_path) |path|
        Io.Dir.cwd().openDir(io, path, .{}) catch |err| {
            log.err("texture directory '{s}': {t}", .{ path, err });
            return err;
        }
    else
        try data_root.openSubdir(io, .textures, .{});
    defer source_dir.close(io);
    var cache_dir = try data_root.createSubdir(io, .cache);
    defer cache_dir.close(io);
    try assets.EarthMaps.bake(gpa, io, source_dir, cache_dir, config.texture_quality);
}

fn milliseconds(duration: Io.Duration) f64 {
    return @as(f64, @floatFromInt(duration.toNanoseconds())) / std.time.ns_per_ms;
}

test "bake-probes takes its fields' paths" {
    const bake = try parseArguments(&.{ "bake-probes", "--probes", "a.eprb", "--out", "b.eprb", "--half", "c.eprb" });
    try std.testing.expectEqual(Command.@"bake-probes", bake.command);
    try std.testing.expectEqualStrings("a.eprb", bake.probes_path.?);
    try std.testing.expectEqualStrings("b.eprb", bake.out_path.?);
    try std.testing.expectEqualStrings("c.eprb", bake.half_path.?);
}

test "command line parsing" {
    try std.testing.expectEqual(Command.view, (try parseArguments(&.{})).command);
    const full = try parseArguments(&.{ "datagen", "--backend", "metal", "--validation" });
    try std.testing.expectEqual(Command.datagen, full.command);
    try std.testing.expectEqual(gpu.Device.Backend.metal, full.backend);
    try std.testing.expect(full.validation);
    try std.testing.expectEqual(Command.@"bake-textures", (try parseArguments(&.{"bake-textures"})).command);
    try std.testing.expectEqual(Command.download, (try parseArguments(&.{"download"})).command);
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{"--backend"}));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "--backend", "cuda" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "view", "render" }));
    try std.testing.expectError(error.HelpRequested, parseArguments(&.{ "view", "--help" }));

    const offline = try parseArguments(&.{ "render", "--mode", "debug", "--preset", "symmetric", "--width", "640", "--spp", "16", "--seed", "32", "--out", "x.png" });
    try std.testing.expectEqual(Command.render, offline.command);
    try std.testing.expectEqual(render.scene.Preset.symmetric, offline.render.preset.?);
    try std.testing.expectEqual([2]u32{ 640, 1080 }, offline.render.size);
    try std.testing.expectEqual(16, offline.render.sample_count);
    try std.testing.expectEqual(32, offline.seed);
    try std.testing.expectEqualStrings("x.png", offline.out_path.?);
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "render", "--spp", "0" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "render", "--width", "-3" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "render", "--preset", "flat" }));

    const data = try parseArguments(&.{ "datagen", "--sampling", "grid", "--grid", "8,32,128,32", "--paths", "1024", "--seed", "4294967295", "--exclude-sun-disk", "--sunset-scale", "0.5" });
    try std.testing.expectEqual(DatagenSampling.grid, data.datagen.sampling);
    try std.testing.expectEqual([4]u32{ 8, 32, 128, 32 }, data.datagen.grid.?);
    try std.testing.expectEqual(1024, data.datagen.path_count);
    try std.testing.expectEqual(std.math.maxInt(u32), data.seed);
    try std.testing.expect(data.datagen.exclude_sun_disk);
    try std.testing.expectEqual(0.5, data.datagen.sunset_scale);
    try std.testing.expectEqual(0.1, data.datagen.sunset_fraction);
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--grid", "8,32,128" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--grid", "8,32,128,32,2" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--grid", "8,32,127,32" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--grid", "1,32,128,32" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--grid", "65536,65536,2,1" }));
    try std.testing.expectError(error.InvalidArguments, parseArguments(&.{ "datagen", "--sunset-fraction", "1.5" }));

    const inputs = try parseArguments(&.{ "render", "--mode", "bruneton", "--lut", "grid.edgn", "--weights", "net.emlp" });
    try std.testing.expectEqual(render.Renderer.ModeKind.bruneton, inputs.render.mode.?);
    try std.testing.expectEqualStrings("grid.edgn", inputs.lut_path.?);
    try std.testing.expectEqualStrings("net.emlp", inputs.weights_path.?);

    const rooted = try parseArguments(&.{ "--data", "/srv/earth", "--config", "presets/florida.zon" });
    try std.testing.expectEqual(Command.view, rooted.command);
    try std.testing.expectEqualStrings("/srv/earth", rooted.data_path);
    try std.testing.expectEqualStrings("presets/florida.zon", rooted.config_path.?);
    try std.testing.expectEqualStrings(".", offline.data_path);
}

test {
    _ = gpu;
    _ = viewer;
    _ = @import("platform.zig");
    _ = @import("assets.zig");
    _ = @import("render.zig");
}
