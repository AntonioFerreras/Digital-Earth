//! ProbeLayout.zig's cells' depths inside the cloud, measured on the GPU
//! (shaders/modes/cloud_bench_layout.slang) from the volume's density there: its
//! occupancy is marked once, then each layout the search tries reads its cells' depths
//! back. The host computes which texels each cell overlaps, so that the depths, and the
//! layouts, are those of ProbeLayout.Occupancy on the CPU.

const LayoutDepths = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const gpu = @import("../gpu.zig");
const CloudVolume = @import("../assets/CloudVolume.zig");
const CloudBench = @import("CloudBench.zig");
const ProbeLayout = @import("ProbeLayout.zig");

device: gpu.Device,
grid: ProbeLayout.Grid,
voxels: [3]u32,
occupancy: gpu.Texture,
measure_cells: gpu.Kernel,

/// shaders/modes/cloud_bench_layout.slang's.
const LayoutUniforms = extern struct {
    voxels: [4]u32,
    texels: [4]u32,
    cells: [4]u32,
};

/// Marks the occupancy of `volume`, whose density `density` holds on the GPU.
pub fn init(device: gpu.Device, volume: *const CloudVolume, density: gpu.Texture) gpu.Kernel.LayoutError!LayoutDepths {
    const grid: ProbeLayout.Grid = .init(volume);
    // The distances, which `mark_occupancy` overwrites with -1 where they hold no cloud.
    const occupancy = try CloudBench.uploadVolume(device, grid.size, volume.distance, .{ .shader_resource = true, .unordered_access = true }, "cloud bench occupancy");
    errdefer occupancy.deinit();

    const mark: gpu.Kernel = try .init(device, "modes/cloud_bench_layout", "mark_occupancy", &.{});
    defer mark.deinit();
    const uniforms: LayoutUniforms = .{
        .voxels = .{ volume.size[0], volume.size[1], volume.size[2], CloudVolume.distance_downsample },
        .texels = .{ grid.size[0], grid.size[1], grid.size[2], 0 },
        .cells = @splat(0),
    };
    try mark.checkLayout(LayoutUniforms, "layout");
    const bindings = [_]gpu.CommandList.Binding{
        .{ .param = try mark.param("layout"), .resource = .{ .data = std.mem.asBytes(&uniforms) } },
        .{ .param = try mark.param("layout_density"), .resource = .{ .texture = density } },
        .{ .param = try mark.param("layout_occupancy"), .resource = .{ .texture = occupancy } },
    };
    const commands: gpu.CommandList = try .begin(device);
    const dispatched = commands.dispatch(mark, &bindings, mark.groupCount(grid.size));
    try commands.submit(null);
    try dispatched;
    // `mark` goes once the GPU is done with it.
    try device.waitIdle();

    const measure_cells: gpu.Kernel = try .init(device, "modes/cloud_bench_layout", "measure_cells", &.{});
    errdefer measure_cells.deinit();
    try measure_cells.checkLayout(LayoutUniforms, "layout");
    return .{ .device = device, .grid = grid, .voxels = volume.size, .occupancy = occupancy, .measure_cells = measure_cells };
}

pub fn deinit(depths: *LayoutDepths) void {
    depths.measure_cells.deinit();
    depths.occupancy.deinit();
    depths.* = undefined;
}

pub fn source(depths: *const LayoutDepths) ProbeLayout.DepthSource {
    return .{ .context = depths, .measure = measure };
}

/// Blocks until the GPU has measured the cells, as the layout's build goes on from them.
fn measure(context: *const anyopaque, gpa: Allocator, io: Io, h0: f32, cells: [3]u32) ProbeLayout.BuildError!ProbeLayout.Depths {
    _ = io;
    const depths: *const LayoutDepths = @ptrCast(@alignCast(context));
    const device = depths.device;
    const ranges = try ProbeLayout.texelRanges(gpa, depths.grid, h0, cells);
    defer gpa.free(ranges);
    const range_buffer: gpu.Buffer = try .init(device, .{
        .size = std.mem.sliceAsBytes(ranges).len,
        .element_size = @sizeOf([2]u32),
        .usage = .{ .shader_resource = true },
        .memory = .device,
        .label = "cloud bench layout ranges",
    }, std.mem.sliceAsBytes(ranges));
    defer range_buffer.deinit();
    const count = @as(usize, cells[0]) * cells[1] * cells[2];
    const output: gpu.Buffer = try .init(device, .{
        .size = count * @sizeOf(f32),
        .element_size = @sizeOf(f32),
        .usage = .{ .unordered_access = true, .copy_source = true },
        .memory = .device,
        .label = "cloud bench layout depths",
    }, null);
    defer output.deinit();

    const kernel = depths.measure_cells;
    const uniforms: LayoutUniforms = .{
        .voxels = .{ depths.voxels[0], depths.voxels[1], depths.voxels[2], CloudVolume.distance_downsample },
        .texels = .{ depths.grid.size[0], depths.grid.size[1], depths.grid.size[2], 0 },
        .cells = .{ cells[0], cells[1], cells[2], 0 },
    };
    const bindings = [_]gpu.CommandList.Binding{
        .{ .param = try kernel.param("layout"), .resource = .{ .data = std.mem.asBytes(&uniforms) } },
        .{ .param = try kernel.param("layout_occupancy"), .resource = .{ .texture = depths.occupancy } },
        .{ .param = try kernel.param("layout_ranges"), .resource = .{ .buffer = range_buffer } },
        .{ .param = try kernel.param("layout_depths"), .resource = .{ .buffer = output } },
    };
    const commands: gpu.CommandList = try .begin(device);
    const dispatched = commands.dispatch(kernel, &bindings, kernel.groupCount(cells));
    try commands.submit(null);
    try dispatched;

    var result: ProbeLayout.Depths = try .init(gpa, cells);
    errdefer result.deinit(gpa);
    try output.read(device, 0, std.mem.sliceAsBytes(result.depth));
    for (result.occupied, result.depth) |*occupied, depth| occupied.* = depth >= 0;
    return result;
}
