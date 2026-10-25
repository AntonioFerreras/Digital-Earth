//! The viewer camera: planet-centred f64 state and the fly controls of the original
//! Taichi viewer, producing the renderer's `View`.

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;
const View = @import("../render/View.zig");

const Camera = @This();

/// Metres from the planet centre; never below `planet_radius_m`.
position: Vec3 = .{ -15e6, 0, 15e6 },
/// Unit view direction.
forward: Vec3 = .{ math.sqrt1_2, 0, -math.sqrt1_2 },
/// Unit reference up: the yaw axis, the Space/Ctrl axis, and the roll anchor.
up: Vec3 = .{ 0, 1, 0 },
/// Full vertical angle of view.
vertical_fov_degrees: f64 = 27,
/// Horizontal stretch of the image plane on top of the image aspect ratio.
aspect_scale: f64 = 1,

pub const Vec3 = View.Vec3;

pub const planet_radius_m = 6371e3;

/// Radians turned per window-size of cursor travel.
const drag_radians = 3;
/// Speed per metre of altitude; altitude counts up to half the planet radius.
const speed_per_altitude_hz = 0.05 * 30;
const boost_factor = 3;
const teleport_altitude_m = 100;

/// One frame of user intent, in the original viewer's terms.
pub const Motion = struct {
    /// Signed thrust along the view direction, the left vector, and `up`.
    forward: f64 = 0,
    left: f64 = 0,
    up: f64 = 0,
    boost: bool = false,
    /// A factor on the speed, which grows with altitude, for scenes far smaller than
    /// the altitude they are seen from.
    speed_scale: f64 = 1,
    /// Cursor travel as a fraction of the window size (x right, y down).
    drag: [2]f64 = .{ 0, 0 },
    up_mode: ?UpMode = null,
    /// Jump to `teleport_altitude_m` above the point below the camera.
    teleport: bool = false,
};

pub const UpMode = enum { radial, world_y };

/// Orthonormal camera frame: right = forward x up.
pub const Basis = struct { forward: Vec3, right: Vec3, up: Vec3 };

/// Applies `motion` over `dt_s` seconds and returns whether the view changed. A step that
/// would end below the planet radius is dropped.
pub fn update(camera: *Camera, motion: Motion, dt_s: f64) bool {
    assert(dt_s >= 0);
    assert(length(camera.position) >= planet_radius_m);
    var changed = false;
    if (motion.up_mode) |up_mode| {
        camera.up = switch (up_mode) {
            .radial => normalize(camera.position),
            .world_y => .{ 0, 1, 0 },
        };
        changed = true;
    }
    if (motion.teleport) {
        camera.up = normalize(camera.position);
        camera.position = camera.up * splat(planet_radius_m + teleport_altitude_m);
        changed = true;
    }
    if (motion.forward != 0 or motion.left != 0 or motion.up != 0) {
        const direction = camera.forward * splat(motion.forward) +
            camera.left() * splat(motion.left) +
            camera.up * splat(motion.up);
        const altitude_m = math.clamp(camera.altitude(), 0, planet_radius_m / 2);
        const boost: f64 = if (motion.boost) boost_factor else 1;
        const moved = camera.position + direction * splat(speed_per_altitude_hz * altitude_m * boost * motion.speed_scale * dt_s);
        if (length(moved) >= planet_radius_m) camera.position = moved;
        changed = true;
    }
    if (motion.drag[0] != 0 or motion.drag[1] != 0) {
        // Yaw about `up`, then pitch about the left vector of the original direction.
        const left_axis = camera.left();
        const yawed = rotate(camera.forward, camera.up, -motion.drag[0] * drag_radians);
        camera.forward = normalize(rotate(yawed, left_axis, motion.drag[1] * drag_radians));
        changed = true;
    }
    assert(length(camera.position) >= planet_radius_m);
    return changed;
}

/// Metres above sea level, the planet's sphere.
pub fn altitude(camera: Camera) f64 {
    return length(camera.position) - planet_radius_m;
}

pub fn basis(camera: Camera) Basis {
    const right = -camera.left();
    const up = normalize(cross(right, camera.forward));
    return .{ .forward = camera.forward, .right = right, .up = up };
}

pub fn view(camera: Camera, sun_direction: View.Vec3) View {
    const frame = camera.basis();
    return .{
        .position = camera.position,
        .forward = frame.forward,
        .right = frame.right,
        .up = frame.up,
        .vertical_fov_degrees = camera.vertical_fov_degrees,
        .aspect_scale = camera.aspect_scale,
        .sun_direction = sun_direction,
    };
}

/// The cosine of the zenith angle of `sun_direction` at the camera, against the
/// planet's vertical there.
pub fn cosZenith(camera: Camera, sun_direction: View.Vec3) f64 {
    return @reduce(.Add, sun_direction * normalize(camera.position));
}

/// The local vertical at the camera and two horizontal axes: azimuth zero towards the
/// world's +Z, as the cloud bench's sun takes it, and a quarter turn towards +X.
pub const Horizon = struct {
    up: Vec3,
    north: Vec3,
    east: Vec3,

    pub fn elevationAzimuth(frame: Horizon, towards: Vec3) [2]f64 {
        return .{
            math.asin(math.clamp(dot(towards, frame.up), -1, 1)),
            math.atan2(dot(towards, frame.east), dot(towards, frame.north)),
        };
    }

    pub fn direction(frame: Horizon, elevation: f64, azimuth: f64) Vec3 {
        const level = frame.north * splat(@cos(azimuth)) + frame.east * splat(@sin(azimuth));
        return level * splat(@cos(elevation)) + frame.up * splat(@sin(elevation));
    }
};

pub fn horizon(camera: Camera) Horizon {
    const up = normalize(camera.position);
    const reference: Vec3 = if (@abs(up[2]) < 0.9) .{ 0, 0, 1 } else .{ 1, 0, 0 };
    const north = normalize(reference - up * splat(dot(reference, up)));
    return .{ .up = up, .north = north, .east = cross(up, north) };
}

/// Unit vector to the left of the view, orthogonal to `forward`. Looking along `up`
/// leaves it undefined, so the world axis least aligned with the view stands in.
fn left(camera: Camera) Vec3 {
    if (@abs(dot(camera.up, camera.forward)) <= 0.999) {
        return normalize(cross(camera.up, camera.forward));
    }
    const alignment: [3]f64 = @abs(camera.forward);
    var axis: [3]f64 = .{ 0, 0, 0 };
    axis[std.mem.findMin(f64, &alignment)] = 1;
    return normalize(cross(axis, camera.forward));
}

/// Counter-clockwise rotation of `vector` about the unit `axis` (Rodrigues).
fn rotate(vector: Vec3, axis: Vec3, angle: f64) Vec3 {
    const cos = @cos(angle);
    return vector * splat(cos) + cross(axis, vector) * splat(@sin(angle)) +
        axis * splat(dot(axis, vector) * (1 - cos));
}

fn splat(value: f64) Vec3 {
    return @splat(value);
}

fn dot(a: Vec3, b: Vec3) f64 {
    return @reduce(.Add, a * b);
}

fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

fn length(vector: Vec3) f64 {
    return @sqrt(dot(vector, vector));
}

fn normalize(vector: Vec3) Vec3 {
    const vector_length = length(vector);
    assert(vector_length > 0);
    return vector / splat(vector_length);
}

const testing = std.testing;
const tolerance = 1e-9;

fn expectVec(expected: Vec3, actual: Vec3, abs_tolerance: f64) !void {
    inline for (0..3) |axis| try testing.expectApproxEqAbs(expected[axis], actual[axis], abs_tolerance);
}

fn expectOrthonormal(frame: Basis) !void {
    for ([_]Vec3{ frame.forward, frame.right, frame.up }) |axis| {
        try testing.expect(math.isFinite(@reduce(.Add, axis)));
        try testing.expectApproxEqAbs(1, length(axis), tolerance);
    }
    try testing.expectApproxEqAbs(0, dot(frame.forward, frame.right), tolerance);
    try testing.expectApproxEqAbs(0, dot(frame.forward, frame.up), tolerance);
    try testing.expectApproxEqAbs(0, dot(frame.right, frame.up), tolerance);
}

test "default camera looks at the planet centre with a right-handed frame" {
    const camera: Camera = .{};
    try testing.expectApproxEqAbs(1, dot(camera.forward, -normalize(camera.position)), tolerance);
    const frame = camera.basis();
    try expectOrthonormal(frame);
    try expectVec(frame.forward, cross(frame.up, frame.right), tolerance);
}

test "speed scales with altitude up to half the planet radius, boost triples it" {
    // Facing straight out, a forward step changes only the radius.
    const altitude_m = 1000;
    const start: Camera = .{ .position = .{ planet_radius_m + altitude_m, 0, 0 }, .forward = .{ 1, 0, 0 } };
    var camera = start;
    try testing.expect(camera.update(.{ .forward = 1 }, 0.5));
    try testing.expectApproxEqAbs(1.5 * altitude_m * 0.5, camera.position[0] - start.position[0], 1e-6);

    camera = start;
    _ = camera.update(.{ .forward = 1, .boost = true }, 0.5);
    try testing.expectApproxEqAbs(3 * 1.5 * altitude_m * 0.5, camera.position[0] - start.position[0], 1e-6);

    const far: Camera = .{ .position = .{ 3 * planet_radius_m, 0, 0 }, .forward = .{ 1, 0, 0 } };
    camera = far;
    _ = camera.update(.{ .forward = 1 }, 1);
    try testing.expectApproxEqAbs(1.5 * planet_radius_m / 2, camera.position[0] - far.position[0], 1e-3);
}

test "a step that would end below the planet radius is dropped" {
    const start: Camera = .{ .position = .{ planet_radius_m + 10, 0, 0 }, .forward = .{ -1, 0, 0 } };
    var camera = start;
    // 15 m/s at 10 m altitude: one second would end 5 m underground.
    try testing.expect(camera.update(.{ .forward = 1 }, 1));
    try expectVec(start.position, camera.position, 0);

    _ = camera.update(.{ .forward = 1 }, 0.5);
    try testing.expectApproxEqAbs(planet_radius_m + 2.5, camera.position[0], 1e-6);
}

test "dragging right turns right and dragging down looks down" {
    const start: Camera = .{ .position = .{ -15e6, 0, 0 }, .forward = .{ 1, 0, 0 } };
    const right_before = start.basis().right;

    var camera = start;
    try testing.expect(camera.update(.{ .drag = .{ 0.1, 0 } }, 0));
    try testing.expect(dot(camera.forward, right_before) > 0);
    try testing.expectApproxEqAbs(@cos(0.3), dot(camera.forward, start.forward), tolerance);

    camera = start;
    _ = camera.update(.{ .drag = .{ 0, 0.1 } }, 0);
    try testing.expect(camera.forward[1] < 0);
    try testing.expectApproxEqAbs(@cos(0.3), dot(camera.forward, start.forward), tolerance);
}

test "up modes and teleport" {
    var camera: Camera = .{};
    try testing.expect(camera.update(.{ .up_mode = .radial }, 0));
    try expectVec(normalize(camera.position), camera.up, tolerance);
    _ = camera.update(.{ .up_mode = .world_y }, 0);
    try expectVec(.{ 0, 1, 0 }, camera.up, 0);

    const direction = normalize(camera.position);
    _ = camera.update(.{ .teleport = true }, 0);
    try testing.expectApproxEqAbs(planet_radius_m + 100, length(camera.position), 1e-6);
    try expectVec(direction, camera.up, tolerance);
    try testing.expect(!camera.update(.{}, 1));
}

test "basis stays orthonormal when looking along the up reference" {
    for ([_]Vec3{ .{ 0, 1, 0 }, .{ 1, 0, 0 }, .{ 0, 0, -1 } }) |axis| {
        const camera: Camera = .{ .position = axis * splat(2 * planet_radius_m), .forward = -axis, .up = axis };
        try expectOrthonormal(camera.basis());
    }
}

test "no input leaves the camera unchanged" {
    const start: Camera = .{};
    var camera = start;
    try testing.expect(!camera.update(.{}, 0.016));
    try testing.expectEqualDeep(start, camera);
}
