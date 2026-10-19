//! The `bruneton` render mode (shaders/modes/bruneton.slang): the symmetric scene's
//! radiance looked up per camera ray in the dense uvwz table that
//! `Renderer.Shared.lut4d` names, read and uploaded once per activation in the folded
//! layout of shaders/atmosphere/lut4d.slang.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("../gpu.zig");
const Lut4d = @import("../assets/Lut4d.zig");
const uniforms = @import("uniforms.zig");
const ModeKernel = @import("ModeKernel.zig");
const Renderer = @import("Renderer.zig");

const BrunetonMode = @This();

kernel: ModeKernel,
lut_param: gpu.Kernel.Param,
shape_param: gpu.Kernel.Param,
lut: gpu.Texture,
/// `Lut4d.shape`, as the shader's `uint4`.
shape: [4]u32,

/// Half floats hold the sky's radiance (below 100) to 1e-3 relative, far below its
/// path-tracing noise, at half the memory and bandwidth of f32. A table holding the
/// sun disk exceeds their range and is refused.
const texture_format: gpu.Texture.Format = .rgba16_float;

pub const InitError = gpu.Kernel.LayoutError || UploadError || Lut4d.LoadError || error{Lut4dMissing};
pub const UploadError = gpu.Error || error{ OutOfMemory, Lut4dTooLarge, Lut4dOutOfRange };

const log = std.log.scoped(.bruneton);

pub fn init(gpa: std.mem.Allocator, shared: *const Renderer.Shared) InitError!BrunetonMode {
    const file = shared.lut4d orelse {
        log.err("bruneton mode needs a scattering table: pass --lut <file.edgn> or bake " ++ Lut4d.default_path ++
            " with `earth datagen --sampling grid --grid 32,32,128,32 --exclude-sun-disk`", .{});
        return error.Lut4dMissing;
    };
    var lut4d = Lut4d.load(gpa, file.io, file.dir, file.path) catch |err| {
        log.err("{s}: {t}", .{ file.path, err });
        return err;
    };
    defer lut4d.deinit(gpa);
    var kernel: ModeKernel = try .init(shared.device, "modes/bruneton", "bruneton_main", &.{});
    errdefer kernel.deinit();
    const lut_param = try kernel.kernel.param("scattering_lut");
    const shape_param = try kernel.kernel.param("scattering_lut_shape");
    const lut = try initTexture(gpa, shared.device, &lut4d, texture_format);
    return .{ .kernel = kernel, .lut_param = lut_param, .shape_param = shape_param, .lut = lut, .shape = lut4d.shape };
}

pub fn deinit(mode: *BrunetonMode, gpa: std.mem.Allocator) void {
    _ = gpa;
    mode.lut.deinit();
    mode.kernel.deinit();
    mode.* = undefined;
}

pub fn record(
    mode: *BrunetonMode,
    command_list: gpu.CommandList,
    shared: *const Renderer.Shared,
    frame: *const Renderer.Frame,
) gpu.Error!void {
    try mode.kernel.dispatch(command_list, shared, frame, &.{
        .{ .param = mode.lut_param, .resource = .{ .texture = mode.lut } },
        .{ .param = mode.shape_param, .resource = .{ .data = std.mem.asBytes(&mode.shape) } },
    });
}

/// The table holds the symmetric scene only.
pub fn invalidate(mode: *BrunetonMode, scene: *const uniforms.SceneUniforms) void {
    _ = mode;
    _ = scene;
}

/// `lut` as the folded 3D texture of shaders/atmosphere/lut4d.slang, alpha 1, in
/// `format` (`rgba16_float` or `rgba32_float`).
pub fn initTexture(gpa: std.mem.Allocator, device: gpu.Device, lut: *const Lut4d, format: gpu.Texture.Format) UploadError!gpu.Texture {
    const nu_size, const mu_s_size, const mu_size, const r_size = lut.shape;
    const width = nu_size * mu_s_size;
    const size_max = device.info().max_texture_dimension_3d;
    if (width > size_max or mu_size > size_max or r_size > size_max) {
        log.err("a {d}x{d}x{d}x{d} table folds into a {d}x{d}x{d} texture; this GPU allows {d} per axis", .{
            nu_size, mu_s_size, mu_size, r_size, width, mu_size, r_size, size_max,
        });
        return error.Lut4dTooLarge;
    }
    const texture: gpu.Texture = try .init(device, .{
        .kind = .@"3d",
        .format = format,
        .width = width,
        .height = mu_size,
        .depth_or_layers = r_size,
        .usage = .{ .shader_resource = true, .copy_destination = true },
        .label = "scattering lut",
    });
    errdefer texture.deinit();

    switch (format) {
        inline .rgba16_float, .rgba32_float => |texel_format| {
            const Channel = if (texel_format == .rgba16_float) f16 else f32;
            const texels = try gpa.alloc([4]Channel, lut.records.len);
            defer gpa.free(texels);
            var index: usize = 0;
            for (0..r_size) |r| {
                for (0..mu_size) |mu| {
                    for (0..nu_size) |nu| {
                        for (0..mu_s_size) |mu_s| {
                            const rgb = lut.at(.{ @intCast(nu), @intCast(mu_s), @intCast(mu), @intCast(r) });
                            for (rgb) |channel| {
                                if (@abs(channel) <= std.math.floatMax(Channel)) continue;
                                log.err("radiance {e} exceeds the {t} texture; bake the table with " ++
                                    "--exclude-sun-disk", .{ channel, format });
                                return error.Lut4dOutOfRange;
                            }
                            texels[index] = .{ @floatCast(rgb[0]), @floatCast(rgb[1]), @floatCast(rgb[2]), 1 };
                            index += 1;
                        }
                    }
                }
            }
            assert(index == texels.len);
            try texture.upload(device, .{
                .x = 0,
                .y = 0,
                .z = 0,
                .width = width,
                .height = mu_size,
                .depth = r_size,
            }, std.mem.sliceAsBytes(texels), @as(u64, width) * @sizeOf([4]Channel));
        },
        else => unreachable,
    }
    return texture;
}
