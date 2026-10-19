//! A `Network` on the GPU as shaders/neural/ reads it, and the generic arguments that
//! specialize a kernel calling `network_radiance` (shaders/neural/network.slang) for
//! its architecture. Kernels with equal `generic_args` are interchangeable.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const Network = @import("../assets/Network.zig");

const GpuNetwork = @This();

/// The encoding type, then the network type; owned.
generic_args: [2][:0]const u8,
/// One line naming the architecture, for the viewer; owned.
description: [:0]const u8,
uniforms: Uniforms,
weights: Weights,
encoding: Encoding,

/// `NetworkUniforms` in shaders/neural/network.slang.
pub const Uniforms = extern struct {
    output_scale: f32,
    mu_rotation: u32,
};

/// The float4 rows of shaders/neural/mlp.slang, in the head's weight source.
const Weights = union(enum) {
    /// `mlp_constants`, padded to the whole parameter; owned.
    constant: *[constant_rows][4]f32,
    /// `mlp_rows`.
    staged: gpu.Buffer,
};

const Encoding = union(Network.Encoding.Kind) {
    raw,
    fourier: [4]f32,
    hash_grid: struct { levels: gpu.Buffer, table: gpu.Buffer },
    k_planes: struct { vertices: [Network.k_planes_scales][4]u32, scales: [Network.k_planes_scales]gpu.Texture },
};

/// MLP_CONSTANT_ROWS in shaders/neural/mlp.slang.
const constant_rows = 4096;
/// MLP_OUTPUTS in shaders/neural/mlp.slang.
const output_count = 3;
/// Registers hold a layer's activations; ROLLED_WIDTH_MIN layers evaluate four
/// outputs at a time and need whole float4s.
const width_max = 64;
/// Widest head whose weights stay in the constant buffer: the inference benchmark
/// found constants fastest for linear and one 16-wide hidden layer, and up to 9x
/// slower than groupshared staging beyond.
const constant_width_max = 16;

pub const InitError = gpu.Error || error{ OutOfMemory, NetworkShapeUnsupported, NetworkOutOfRange };

const log = std.log.scoped(.neural);

/// Uploads `network`, which may be freed afterwards.
pub fn init(gpa: std.mem.Allocator, device: gpu.Device, network: *const Network) InitError!GpuNetwork {
    const head = try Head.init(network);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const rows = try weightRows(arena, network.layers);
    const weights: Weights = if (head.constant) constant: {
        assert(rows.len <= constant_rows);
        const constant = try gpa.create([constant_rows][4]f32);
        @memcpy(constant[0..rows.len], rows);
        @memset(constant[rows.len..], .{ 0, 0, 0, 0 });
        break :constant .{ .constant = constant };
    } else .{ .staged = try initBuffer(device, std.mem.sliceAsBytes(rows), @sizeOf([4]f32), "mlp rows") };
    errdefer deinitWeights(weights, gpa);
    const encoding = try initEncoding(arena, device, network.encoding);
    errdefer deinitEncoding(encoding);

    const encoding_type = try switch (network.encoding) {
        .raw => gpa.dupeSentinel(u8, "Raw", 0),
        .fourier => |fourier| std.fmt.allocPrintSentinel(gpa, "Fourier<{d}>", .{fourier.octaves}, 0),
        .hash_grid => |hash_grid| std.fmt.allocPrintSentinel(gpa, "HashGrid<{d}>", .{hash_grid.levels.len}, 0),
        .k_planes => |k_planes| std.fmt.allocPrintSentinel(gpa, "KPlanes<{d}>", .{k_planes.features}, 0),
    };
    errdefer gpa.free(encoding_type);
    const source = if (head.constant)
        try arena.dupe(u8, "Constant")
    else
        try std.fmt.allocPrint(arena, "GroupStaged<{d}>", .{head.staging_rows});
    const network_type = try std.fmt.allocPrintSentinel(gpa, "Mlp<{d}, {d}, {s}, {s}, {s}>", .{
        head.width, head.hidden, activationType(head.first), activationType(head.rest), source,
    }, 0);
    errdefer gpa.free(network_type);
    const description = try describe(gpa, network, head);
    return .{
        .generic_args = .{ encoding_type, network_type },
        .description = description,
        .uniforms = .{ .output_scale = network.output_scale, .mu_rotation = @intFromBool(network.mu_rotation) },
        .weights = weights,
        .encoding = encoding,
    };
}

pub fn deinit(gpu_network: *GpuNetwork, gpa: std.mem.Allocator) void {
    deinitEncoding(gpu_network.encoding);
    deinitWeights(gpu_network.weights, gpa);
    gpa.free(gpu_network.description);
    for (gpu_network.generic_args) |arg| gpa.free(arg);
    gpu_network.* = undefined;
}

/// The bindings of every global of shaders/neural/ this network reads, located in
/// `kernel`, which must be specialized with `generic_args`; written to `buffer`.
pub fn bindings(gpu_network: *const GpuNetwork, kernel: gpu.Kernel, buffer: *[binding_count_max]gpu.CommandList.Binding) gpu.Error![]const gpu.CommandList.Binding {
    var list: std.ArrayList(gpu.CommandList.Binding) = .initBuffer(buffer);
    list.appendAssumeCapacity(.{ .param = try kernel.param("network_uniforms"), .resource = .{ .data = std.mem.asBytes(&gpu_network.uniforms) } });
    list.appendAssumeCapacity(switch (gpu_network.weights) {
        .constant => |rows| .{ .param = try kernel.param("mlp_constants"), .resource = .{ .data = std.mem.sliceAsBytes(rows) } },
        .staged => |buffer_rows| .{ .param = try kernel.param("mlp_rows"), .resource = .{ .buffer = buffer_rows } },
    });
    switch (gpu_network.encoding) {
        .raw => {},
        .fourier => |*frequency| list.appendAssumeCapacity(.{ .param = try kernel.param("fourier_frequency"), .resource = .{ .data = std.mem.asBytes(frequency) } }),
        .hash_grid => |hash_grid| {
            list.appendAssumeCapacity(.{ .param = try kernel.param("hash_levels"), .resource = .{ .buffer = hash_grid.levels } });
            list.appendAssumeCapacity(.{ .param = try kernel.param("hash_table"), .resource = .{ .buffer = hash_grid.table } });
        },
        .k_planes => |*k_planes| {
            list.appendAssumeCapacity(.{ .param = try kernel.param("kplanes_vertices"), .resource = .{ .data = std.mem.asBytes(&k_planes.vertices) } });
            const names = [Network.k_planes_scales][*:0]const u8{ "kplanes_scale0", "kplanes_scale1", "kplanes_scale2" };
            for (names, k_planes.scales) |name, texture| {
                list.appendAssumeCapacity(.{ .param = try kernel.param(name), .resource = .{ .texture = texture } });
            }
        },
    }
    return list.items;
}

/// Uniforms, weights, and at most four encoding resources.
pub const binding_count_max = 6;

/// The dense layers as shaders/neural/mlp.slang's `Mlp` evaluates them.
const Head = struct {
    /// Neurons per hidden layer; 1 for a linear head, which has none but sizes the
    /// shader's unused hidden array.
    width: u32,
    hidden: u32,
    first: Network.Activation,
    rest: Network.Activation,
    constant: bool,
    /// Float4 rows of the largest layer, the groupshared capacity.
    staging_rows: u32,

    fn init(network: *const Network) error{NetworkShapeUnsupported}!Head {
        const layers = network.layers;
        const hidden: u32 = @intCast(layers.len - 1);
        const width = if (hidden == 0) 1 else layers[0].out_dim;
        var staging_rows: u32 = 0;
        var row_count: u32 = 0;
        for (layers, 0..) |layer, index| {
            const last = index == hidden;
            const shape_fits = if (last)
                layer.out_dim == output_count and layer.activation == .relu
            else
                layer.out_dim == width and (index < 2 or layer.activation == layers[1].activation);
            if (!shape_fits) return unsupported(network);
            const rows = layer.out_dim * divCeil(layer.in_dim) + divCeil(layer.out_dim);
            staging_rows = @max(staging_rows, rows);
            row_count += rows;
        }
        if (width > width_max or (width >= width_max and width % 4 != 0)) return unsupported(network);
        return .{
            .width = width,
            .hidden = hidden,
            .first = if (hidden == 0) .identity else layers[0].activation,
            .rest = if (hidden < 2) .identity else layers[1].activation,
            .constant = hidden <= 1 and width <= constant_width_max and row_count <= constant_rows,
            .staging_rows = staging_rows,
        };
    }

    fn unsupported(network: *const Network) error{NetworkShapeUnsupported} {
        log.err("the neural mode evaluates {d} encoded features -> hidden layers of one width up to {d}, all " ++
            "but the first with one activation -> {d} ReLU outputs; this network has {d} layers", .{
            network.encoding.featureCount(), width_max, output_count, network.layers.len,
        });
        return error.NetworkShapeUnsupported;
    }
};

fn divCeil(count: u32) u32 {
    return std.math.divCeil(u32, count, 4) catch unreachable;
}

fn activationType(activation: Network.Activation) []const u8 {
    return switch (activation) {
        .identity => "Identity",
        .gelu => "Gelu",
        .relu => "Relu",
    };
}

/// Per layer, out_dim rows of the [out][in] matrix zero-padded to whole float4s,
/// then the bias zero-padded to whole float4s.
fn weightRows(arena: std.mem.Allocator, layers: []const Network.Layer) error{OutOfMemory}![]const [4]f32 {
    var rows: std.ArrayList([4]f32) = .empty;
    for (layers) |layer| {
        const in_rows = divCeil(layer.in_dim);
        for (0..layer.out_dim) |o| {
            const weights = layer.weight[o * layer.in_dim ..][0..layer.in_dim];
            for (0..in_rows) |r| try rows.append(arena, padded(weights[4 * r ..]));
        }
        for (0..divCeil(layer.out_dim)) |r| try rows.append(arena, padded(layer.bias[4 * r ..]));
    }
    return rows.items;
}

fn padded(values: []const f32) [4]f32 {
    var row: [4]f32 = @splat(0);
    const count = @min(values.len, 4);
    @memcpy(row[0..count], values[0..count]);
    return row;
}

fn initEncoding(arena: std.mem.Allocator, device: gpu.Device, encoding: Network.Encoding) InitError!Encoding {
    switch (encoding) {
        .raw => return .raw,
        .fourier => |fourier| return .{ .fourier = fourier.frequency },
        .hash_grid => |hash_grid| {
            const levels = try initBuffer(device, std.mem.sliceAsBytes(hash_grid.levels), @sizeOf([4]u32), "hash levels");
            errdefer levels.deinit();
            const table = try initBuffer(device, std.mem.sliceAsBytes(try halves(arena, hash_grid.table)), 0, "hash table");
            return .{ .hash_grid = .{ .levels = levels, .table = table } };
        },
        .k_planes => |k_planes| {
            const max_size = device.info().max_texture_dimension_2d;
            const max_layers = device.info().max_texture_array_layers;
            const layers = 6 * k_planes.features / 4;
            var scales: [Network.k_planes_scales]gpu.Texture = undefined;
            var created: usize = 0;
            errdefer for (scales[0..created]) |texture| texture.deinit();
            var source = k_planes.planes;
            for (&scales, k_planes.vertices) |*texture, vertices| {
                const side = std.mem.max(u32, &vertices);
                if (side > max_size or layers > max_layers) {
                    log.err("K-Planes planes of {d} vertices in {d} layers exceed this GPU's {d} x {d}", .{ side, layers, max_size, max_layers });
                    return error.NetworkShapeUnsupported;
                }
                const texels = try arena.alloc([4]f32, @as(usize, side) * side * layers);
                @memset(texels, .{ 0, 0, 0, 0 });
                for (Network.k_planes_plane_axes, 0..) |axes, plane| {
                    for (0..vertices[axes[1]]) |v| {
                        for (0..vertices[axes[0]]) |u| {
                            for (0..k_planes.features) |f| {
                                const layer = plane * k_planes.features / 4 + f / 4;
                                texels[(layer * side + v) * side + u][f % 4] = source[0];
                                source = source[1..];
                            }
                        }
                    }
                }
                texture.* = try .init(device, .{
                    .kind = .@"2d_array",
                    .format = .rgba16_float,
                    .width = side,
                    .height = side,
                    .depth_or_layers = layers,
                    .usage = .{ .shader_resource = true, .copy_destination = true },
                    .label = "k-planes scale",
                });
                created += 1;
                const data = try halves(arena, @ptrCast(texels));
                try texture.upload(device, .{ .x = 0, .y = 0, .z = 0, .width = side, .height = side, .depth = layers }, std.mem.sliceAsBytes(data), @as(u64, side) * @sizeOf([4]f16));
            }
            assert(source.len == 0);
            return .{ .k_planes = .{ .vertices = k_planes.vertices, .scales = scales } };
        },
    }
}

/// `values` as half floats, which hold trained tables and planes to 1e-3 relative.
fn halves(arena: std.mem.Allocator, values: []const f32) error{ OutOfMemory, NetworkOutOfRange }![]const f16 {
    const result = try arena.alloc(f16, values.len);
    for (result, values) |*half, value| {
        if (!(@abs(value) <= std.math.floatMax(f16))) {
            log.err("encoding value {e} exceeds half precision", .{value});
            return error.NetworkOutOfRange;
        }
        half.* = @floatCast(value);
    }
    return result;
}

fn initBuffer(device: gpu.Device, bytes: []const u8, element_size: u32, label: [*:0]const u8) gpu.Error!gpu.Buffer {
    return .init(device, .{
        .size = bytes.len,
        .element_size = element_size,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .memory = .device,
        .label = label,
    }, bytes);
}

fn deinitWeights(weights: Weights, gpa: std.mem.Allocator) void {
    switch (weights) {
        .constant => |rows| gpa.destroy(rows),
        .staged => |buffer| buffer.deinit(),
    }
}

fn deinitEncoding(encoding: Encoding) void {
    switch (encoding) {
        .raw, .fourier => {},
        .hash_grid => |hash_grid| {
            hash_grid.table.deinit();
            hash_grid.levels.deinit();
        },
        .k_planes => |k_planes| for (k_planes.scales) |texture| texture.deinit(),
    }
}

fn describe(gpa: std.mem.Allocator, network: *const Network, head: Head) error{OutOfMemory}![:0]const u8 {
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    writeDescription(&text.writer, network, head) catch return error.OutOfMemory;
    return text.toOwnedSliceSentinel(0);
}

fn writeDescription(writer: *std.Io.Writer, network: *const Network, head: Head) std.Io.Writer.Error!void {
    switch (network.encoding) {
        .raw => try writer.writeAll("raw uvwz"),
        .fourier => |fourier| try writer.print("Fourier, {d} octaves", .{fourier.octaves}),
        .hash_grid => |hash_grid| try writer.print("hash grid, {d} levels of 2^{d}", .{
            hash_grid.levels.len, std.math.log2_int(u64, @as(u64, hash_grid.levels[0].mask) + 1),
        }),
        .k_planes => |k_planes| try writer.print("K-Planes, F = {d}", .{k_planes.features}),
    }
    if (head.hidden == 0) {
        try writer.writeAll(", linear head");
    } else {
        try writer.print(", {d} x {d} {t}", .{ head.width, head.hidden, if (head.hidden == 1) head.first else head.rest });
    }
    try writer.print(", {d} parameters{s}", .{ network.parameterCount(), if (network.mu_rotation) ", u_mu rotated" else "" });
}
