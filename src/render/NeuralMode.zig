//! The `neural` render mode (shaders/modes/neural.slang): a trained network
//! evaluated per camera ray in place of the symmetric path tracer. The network
//! is read from `Renderer.Shared.network` when the mode starts and again on
//! `select`; the per-pixel kernel is specialized for its architecture, and each
//! specialization is kept, so returning to an architecture compiles nothing.

const std = @import("std");
const gpu = @import("../gpu.zig");
const Network = @import("../assets/Network.zig");
const uniforms = @import("uniforms.zig");
const GpuNetwork = @import("GpuNetwork.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const NeuralMode = @This();

/// The entry of `variants` for `network`'s architecture.
kernel: ModeKernel,
variants: std.ArrayList(Variant),
network: GpuNetwork,

const Variant = struct {
    /// `GpuNetwork.generic_args` joined; owned.
    key: []const u8,
    kernel: ModeKernel,
};

pub const InitError = gpu.Kernel.LayoutError || GpuNetwork.InitError || Network.LoadError || error{NetworkMissing};

const log = std.log.scoped(.neural);

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) InitError!NeuralMode {
    var mode: NeuralMode = .{ .kernel = undefined, .variants = .empty, .network = undefined };
    errdefer mode.deinitVariants(gpa);
    mode.network, mode.kernel = try mode.load(gpa, shared);
    return mode;
}

pub fn deinit(mode: *NeuralMode, gpa: std.mem.Allocator) void {
    mode.network.deinit(gpa);
    mode.deinitVariants(gpa);
    mode.* = undefined;
}

fn deinitVariants(mode: *NeuralMode, gpa: std.mem.Allocator) void {
    for (mode.variants.items) |variant| {
        variant.kernel.deinit();
        gpa.free(variant.key);
    }
    mode.variants.deinit(gpa);
}

/// Replaces the network with the one `shared.network` names now; on failure the
/// previous network stays active.
pub fn select(mode: *NeuralMode, gpa: std.mem.Allocator, shared: *const Renderer.Shared) InitError!void {
    var network, const kernel = try mode.load(gpa, shared);
    errdefer network.deinit(gpa);
    try shared.device.waitIdle();
    mode.network.deinit(gpa);
    mode.network = network;
    mode.kernel = kernel;
}

/// Reads and uploads the network `shared.network` names, and finds or compiles the
/// kernel for its architecture.
fn load(mode: *NeuralMode, gpa: std.mem.Allocator, shared: *const Renderer.Shared) InitError!struct { GpuNetwork, ModeKernel } {
    const file = shared.network orelse {
        log.err("the neural mode needs a network: pass --weights <file" ++ Network.file_extension ++ "> or export one with " ++
            "python/export_weights.py to " ++ Network.default_path ++ " or " ++ Network.directory_path, .{});
        return error.NetworkMissing;
    };
    var network = Network.load(gpa, file.io, file.dir, file.path) catch |err| {
        log.err("{s}: {t}", .{ file.path, err });
        return err;
    };
    defer network.deinit(gpa);
    var gpu_network: GpuNetwork = try .init(gpa, shared.device, &network);
    errdefer gpu_network.deinit(gpa);
    const kernel = try mode.findVariant(gpa, shared.device, gpu_network.generic_args);
    log.info("{s}: {s}", .{ file.path, gpu_network.description });
    return .{ gpu_network, kernel };
}

fn findVariant(mode: *NeuralMode, gpa: std.mem.Allocator, device: gpu.Device, generic_args: [2][:0]const u8) InitError!ModeKernel {
    const key = try std.mem.join(gpa, "|", &generic_args);
    errdefer gpa.free(key);
    for (mode.variants.items) |existing| {
        if (!std.mem.eql(u8, existing.key, key)) continue;
        gpa.free(key);
        return existing.kernel;
    }
    try mode.variants.ensureUnusedCapacity(gpa, 1);
    const kernel: ModeKernel = try .init(device, "modes/neural", "neural_main", &.{ generic_args[0], generic_args[1] });
    mode.variants.appendAssumeCapacity(.{ .key = key, .kernel = kernel });
    return kernel;
}

pub fn record(
    mode: *NeuralMode,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    var buffer: [GpuNetwork.binding_count_max]gpu.CommandList.Binding = undefined;
    try mode.kernel.dispatch(command_list, shared, frame, try mode.network.bindings(mode.kernel.kernel, &buffer));
}

/// The network ignores the preset: it learned the symmetric scene.
pub fn invalidate(mode: *NeuralMode, scene: *const uniforms.SceneUniforms) void {
    _ = mode;
    _ = scene;
}
