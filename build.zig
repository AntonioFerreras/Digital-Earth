const std = @import("std");
const builtin = @import("builtin");

const native_build_dir = "build/native";
const native_prefix = native_build_dir ++ "/install";

pub fn build(b: *std.Build) void {
    // Zig's bundled glibc stubs: host libc detection fails inside the conda
    // environment, and an old glibc floor keeps the binary portable.
    const target = b.standardTargetOptions(.{ .default_target = if (builtin.os.tag == .linux)
        .{ .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 28, .patch = 0 } }
    else
        .{} });
    const optimize = b.standardOptimizeOption(.{});
    const conda_prefix = b.graph.environ_map.get("CONDA_PREFIX") orelse
        @panic("CONDA_PREFIX unset: run through `pixi run`");

    const native = addNativeBuild(b, conda_prefix, optimize);

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.addIncludePath(b.path(native_prefix ++ "/include"));
    translate_c.addSystemIncludePath(b.graph.cwdRelativePath(b.pathJoin(&.{ conda_prefix, "include" })));
    translate_c.step.dependOn(native);

    const app = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    app.addImport("c", translate_c.createModule());
    app.addImport("stb", addStb(b, target));
    app.addLibraryPath(b.path(native_prefix ++ "/lib"));
    app.addLibraryPath(b.graph.cwdRelativePath(b.pathJoin(&.{ conda_prefix, "lib" })));
    app.linkSystemLibrary("earth_native", .{});
    app.linkSystemLibrary("SDL3", .{});
    app.addRPath(.{ .cwd_relative = "$ORIGIN/../lib" });
    app.addRPath(b.graph.cwdRelativePath(b.pathJoin(&.{ conda_prefix, "lib" })));

    const exe = b.addExecutable(.{ .name = "earth", .root_module = app });
    exe.step.dependOn(native);
    b.installArtifact(exe);
    b.installDirectory(.{
        .source_dir = b.path("shaders"),
        .install_dir = .prefix,
        .install_subdir = "share/earth/shaders",
    });
    // The native install tree only exists once CMake has run, even on a clean checkout.
    for ([_]std.Build.Step.InstallDir.Options{ .{
        .source_dir = b.path(native_prefix ++ "/lib"),
        .install_dir = .lib,
        .install_subdir = "",
        .exclude_extensions = &.{".a"},
    }, .{
        .source_dir = b.path(native_prefix ++ "/share/earth_native/shaders"),
        .install_dir = .prefix,
        .install_subdir = "share/earth/shaders/native",
    } }) |options| {
        const install_native = b.addInstallDirectory(options);
        install_native.step.dependOn(native);
        b.getInstallStep().dependOn(&install_native.step);
    }

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    b.step("run", "Run the viewer or a subcommand (`zig build run -- <args>`)").dependOn(&run.step);

    const tests = b.addTest(.{
        .root_module = app,
        .test_runner = .{ .path = b.path("src/test_runner.zig"), .mode = .simple },
    });
    tests.step.dependOn(native);
    const run_tests = b.addRunArtifact(tests);
    run_tests.step.dependOn(b.getInstallStep());
    // Tests resolve shaders and assets relative to the source tree.
    run_tests.setCwd(b.path("."));
    run_tests.addPassthruArgs();
    b.step("test", "Run tests (`zig build test -- --test-filter <substr>`)").dependOn(&run_tests.step);

    addShaderChecks(b, native);
}

/// Entry points the host specializes at run time (`gpu.Kernel.init`'s generic
/// arguments), with the specializations `zig build shaders` compiles instead of the
/// whole file: one per encoding and per weight source of shaders/neural/.
const specialized_entries = [_]struct { file: []const u8, entry: []const u8, generic_args: []const []const u8 }{
    .{ .file = "shaders/modes/neural.slang", .entry = "neural_main", .generic_args = &.{ "Raw", "Mlp<64, 4, Identity, Gelu, GroupStaged<1040>>" } },
    .{ .file = "shaders/modes/neural.slang", .entry = "neural_main", .generic_args = &.{ "Fourier<4>", "Mlp<32, 2, Relu, Relu, GroupStaged<296>>" } },
    .{ .file = "shaders/modes/neural.slang", .entry = "neural_main", .generic_args = &.{ "HashGrid<16>", "Mlp<16, 1, Relu, Relu, Constant>" } },
    .{ .file = "shaders/modes/neural.slang", .entry = "neural_main", .generic_args = &.{ "KPlanes<8>", "Mlp<1, 0, Identity, Identity, Constant>" } },
    .{ .file = "shaders/tests/test_neural.slang", .entry = "test_network", .generic_args = &.{ "KPlanes<4>", "Mlp<16, 1, Relu, Relu, Constant>" } },
    .{ .file = "shaders/tests/test_neural.slang", .entry = "test_erf", .generic_args = &.{} },
};

/// `zig build shaders`: compiles every file with a `[shader(...)]` entry point in
/// the entry directories for SPIR-V and Metal, or its `specialized_entries`.
/// Library modules are checked through the entries that import them; slangc's
/// depfile tracks those imports. The scan registers what it reads so the configure
/// cache stays valid.
fn addShaderChecks(b: *std.Build, native: *std.Build.Step) void {
    const step = b.step("shaders", "Compile-check every shader entry point for SPIR-V and Metal");
    const io = b.graph.io;
    b.dependOnDirectoryContents(b.path("shaders"));
    for ([_][]const u8{ "shaders/modes", "shaders/post", "shaders/tests" }) |dir_path| {
        var dir = b.root.openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => std.debug.panic("open {s}: {t}", .{ dir_path, err }),
        };
        defer dir.close(io);
        b.dependOnDirectoryContents(b.path(dir_path));
        var entries = dir.iterate();
        while (entries.next(io) catch |err| std.debug.panic("list {s}: {t}", .{ dir_path, err })) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".slang")) continue;
            const file_path = b.pathJoin(&.{ dir_path, entry.name });
            b.dependOnFileContents(b.path(file_path));
            const source = dir.readFileAlloc(io, entry.name, b.allocator, .limited(1 << 20)) catch |err|
                std.debug.panic("read {s}/{s}: {t}", .{ dir_path, entry.name, err });
            if (std.mem.indexOf(u8, source, "[shader(") == null) continue;
            const stem = entry.name[0 .. entry.name.len - ".slang".len];
            var specialized = false;
            for (specialized_entries, 0..) |specialization, index| {
                if (!std.mem.eql(u8, specialization.file, file_path)) continue;
                specialized = true;
                var args: std.ArrayList([]const u8) = .empty;
                args.appendSlice(b.allocator, &.{ "-entry", specialization.entry }) catch @panic("OOM");
                for (specialization.generic_args) |arg| args.appendSlice(b.allocator, &.{ "-specialize", arg }) catch @panic("OOM");
                addShaderCheck(b, step, native, file_path, b.fmt("{s}_{d}", .{ stem, index }), args.items);
            }
            if (!specialized) addShaderCheck(b, step, native, file_path, stem, &.{});
        }
    }
}

fn addShaderCheck(b: *std.Build, step: *std.Build.Step, native: *std.Build.Step, file_path: []const u8, stem: []const u8, args: []const []const u8) void {
    for ([_][2][]const u8{ .{ "spirv", "spv" }, .{ "metal", "metal" } }) |target| {
        const compile = b.addRunFile(b.path(native_prefix ++ "/bin/slangc"));
        compile.setName(b.fmt("slangc {s} {s} ({s})", .{ file_path, std.mem.join(b.allocator, " ", args) catch @panic("OOM"), target[0] }));
        compile.addFileArg(b.path(file_path));
        compile.addPrefixedDirectoryArg("-I", b.path("shaders"));
        compile.addArgs(args);
        compile.addArgs(&.{ "-target", target[0], "-o" });
        _ = compile.addOutputFileArg(b.fmt("{s}.{s}", .{ stem, target[1] }));
        compile.addArg("-depfile");
        _ = compile.addDepFileOutputArg(b.fmt("{s}.{s}.d", .{ stem, target[1] }));
        compile.step.dependOn(native);
        step.dependOn(&compile.step);
    }
}

/// CMake owns every C++ translation unit so the process holds one C++ runtime;
/// Ninja keeps the always-run steps incremental.
fn addNativeBuild(b: *std.Build, conda_prefix: []const u8, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    const configure = b.addSystemCommand(&.{
        "cmake",                                                                                   "-S",                                       "native",
        "-B",                                                                                      native_build_dir,                           "-G",
        "Ninja",                                                                                   "-DCMAKE_INSTALL_PREFIX=" ++ native_prefix, b.fmt("-DCMAKE_PREFIX_PATH={s}", .{conda_prefix}),
        b.fmt("-DCMAKE_BUILD_TYPE={s}", .{if (optimize == .debug) "Debug" else "RelWithDebInfo"}),
    });
    configure.has_side_effects = true;
    const compile = b.addSystemCommand(&.{ "cmake", "--build", native_build_dir });
    compile.has_side_effects = true;
    compile.step.dependOn(&configure.step);
    const install = b.addSystemCommand(&.{ "cmake", "--install", native_build_dir });
    install.has_side_effects = true;
    install.step.dependOn(&compile.step);
    return &install.step;
}

/// stb_image and stb_image_write compiled from the pinned dependency. Always
/// optimized: Debug C is UBSan-instrumented and unoptimized, too slow for
/// decoding 21600-wide Earth maps.
fn addStb(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const stb = b.dependency("stb", .{});
    const translate_stb = b.addTranslateC(.{
        .root_source_file = b.path("src/assets/stb.h"),
        .target = target,
        .optimize = .fast,
    });
    translate_stb.addIncludePath(stb.path(""));
    const module = translate_stb.createModule();
    module.addIncludePath(stb.path(""));
    module.addIncludePath(b.path("src/assets"));
    module.addCSourceFile(.{ .file = b.path("src/assets/stb_impl.c"), .flags = &.{"-std=c11"} });
    return module;
}
