//! What a frame is rendered from: a pinhole camera in planet-centred metres and the sun.
//! The renderer restarts accumulation whenever any of it changes.

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;
const FrameUniforms = @import("uniforms.zig").FrameUniforms;

const View = @This();

position: Vec3,
/// Orthonormal and right-handed: right = forward x up.
forward: Vec3,
right: Vec3,
up: Vec3,
/// Full vertical angle of view.
vertical_fov_degrees: f64,
/// Horizontal stretch of the image plane on top of the image aspect ratio.
aspect_scale: f64,
sun: Sun,

pub const Vec3 = @Vector(3, f64);

pub const vertical_fov_degrees_min = 1;
pub const vertical_fov_degrees_max = 170;

pub const Sun = struct {
    /// Position along the sun's path, the original's `sun_angle`: 0 is +Z, 90 is -X.
    angle_degrees: f64 = 60,
    /// Tilt of the sun's path about the X axis, the original's `sun_path_rot`.
    path_rotation_degrees: f64 = -45,
};

/// Writes the camera and sun fields of `uniforms` for an image of `image_size` pixels;
/// the image-plane half height is tan(vertical_fov / 2).
pub fn writeUniforms(view: View, uniforms: *FrameUniforms, image_size: [2]u32) void {
    assert(image_size[0] > 0);
    assert(image_size[1] > 0);
    assert(view.vertical_fov_degrees >= vertical_fov_degrees_min);
    assert(view.vertical_fov_degrees <= vertical_fov_degrees_max);
    assert(@abs(@reduce(.Add, view.forward * view.right)) < 1e-6);
    const aspect = @as(f64, @floatFromInt(image_size[0])) / @as(f64, @floatFromInt(image_size[1]));
    const half_height = @tan(math.degreesToRadians(view.vertical_fov_degrees) / 2);
    uniforms.camera_position = vec4(view.position, 0);
    uniforms.camera_forward = vec4(view.forward, half_height);
    uniforms.camera_right = vec4(view.right, aspect * view.aspect_scale);
    uniforms.camera_up = vec4(view.up, 0);

    // The original's light direction.
    const angle = math.degreesToRadians(view.sun.angle_degrees);
    const rotation = math.degreesToRadians(view.sun.path_rotation_degrees);
    uniforms.sun_direction = vec4(.{ -@sin(angle), @cos(angle) * -@sin(rotation), @cos(angle) * @cos(rotation) }, 0);
}

fn vec4(vector: Vec3, w: f64) [4]f32 {
    return .{ @floatCast(vector[0]), @floatCast(vector[1]), @floatCast(vector[2]), @floatCast(w) };
}

const testing = std.testing;

test "uniforms carry the image-plane half height, aspect, and sun direction" {
    var view: View = .{
        .position = .{ -15e6, 0, 0 },
        .forward = .{ 1, 0, 0 },
        .right = .{ 0, 0, 1 },
        .up = .{ 0, 1, 0 },
        .vertical_fov_degrees = 90,
        .aspect_scale = 1.25,
        .sun = .{ .angle_degrees = 90, .path_rotation_degrees = 0 },
    };
    var uniforms = std.mem.zeroes(FrameUniforms);
    view.writeUniforms(&uniforms, .{ 200, 100 });
    try testing.expectApproxEqAbs(1, uniforms.camera_forward[3], 1e-6);
    try testing.expectApproxEqAbs(2.5, uniforms.camera_right[3], 1e-6);
    try testing.expectEqual(-15e6, uniforms.camera_position[0]);
    try testing.expectApproxEqAbs(-1, uniforms.sun_direction[0], 1e-6);
    try testing.expectApproxEqAbs(0, uniforms.sun_direction[2], 1e-6);

    view.vertical_fov_degrees = 27;
    view.writeUniforms(&uniforms, .{ 1920, 1080 });
    try testing.expectApproxEqAbs(@tan(math.degreesToRadians(13.5)), uniforms.camera_forward[3], 1e-6);
}
