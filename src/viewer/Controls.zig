//! Key bindings of the viewer, by physical key (scancode) so WASD keeps its shape on
//! every keyboard layout.

const std = @import("std");
const c = @import("c");
const Camera = @import("Camera.zig");
const Window = @import("../platform/Window.zig");

const Controls = @This();

move_forward: Scancode = c.SDL_SCANCODE_W,
move_back: Scancode = c.SDL_SCANCODE_S,
move_left: Scancode = c.SDL_SCANCODE_A,
move_right: Scancode = c.SDL_SCANCODE_D,
move_up: Scancode = c.SDL_SCANCODE_SPACE,
move_down: Scancode = c.SDL_SCANCODE_LCTRL,
boost: Scancode = c.SDL_SCANCODE_LSHIFT,
up_radial: Scancode = c.SDL_SCANCODE_Q,
up_world_y: Scancode = c.SDL_SCANCODE_E,
teleport: Scancode = c.SDL_SCANCODE_R,
save_config: Scancode = c.SDL_SCANCODE_I,
load_config: Scancode = c.SDL_SCANCODE_O,
screenshot: Scancode = c.SDL_SCANCODE_P,
toggle_gui: Scancode = c.SDL_SCANCODE_G,

pub const Scancode = u16;

/// Everything one frame of input asks for.
pub const Intent = struct {
    motion: Camera.Motion,
    save_config: bool,
    load_config: bool,
    screenshot: bool,
    toggle_gui: bool,
};

/// Movement keys act while held; every other binding acts once per press.
pub fn map(controls: Controls, input: Window.Input) Intent {
    const held = input.held;
    const pressed = input.pressed;
    return .{
        .motion = .{
            .forward = axis(held, controls.move_forward, controls.move_back),
            .left = axis(held, controls.move_left, controls.move_right),
            .up = axis(held, controls.move_up, controls.move_down),
            .boost = held.isSet(controls.boost),
            .drag = input.drag,
            // E wins over Q, matching the order the original viewer applied them in.
            .up_mode = if (pressed.isSet(controls.up_world_y))
                .world_y
            else if (pressed.isSet(controls.up_radial))
                .radial
            else
                null,
            .teleport = pressed.isSet(controls.teleport),
        },
        .save_config = pressed.isSet(controls.save_config),
        .load_config = pressed.isSet(controls.load_config),
        .screenshot = pressed.isSet(controls.screenshot),
        .toggle_gui = pressed.isSet(controls.toggle_gui),
    };
}

fn axis(held: Window.KeySet, positive: Scancode, negative: Scancode) f64 {
    const positive_value: f64 = @floatFromInt(@intFromBool(held.isSet(positive)));
    const negative_value: f64 = @floatFromInt(@intFromBool(held.isSet(negative)));
    return positive_value - negative_value;
}

test "held keys thrust, opposite keys cancel, presses fire once" {
    const controls: Controls = .{};
    var input: Window.Input = .{ .drag = .{ 0.25, -0.5 } };
    input.held.set(controls.move_forward);
    input.held.set(controls.move_left);
    input.held.set(controls.move_right);
    input.held.set(controls.boost);
    input.pressed.set(controls.up_radial);
    input.pressed.set(controls.toggle_gui);

    const intent = controls.map(input);
    try std.testing.expectEqual(1, intent.motion.forward);
    try std.testing.expectEqual(0, intent.motion.left);
    try std.testing.expectEqual(0, intent.motion.up);
    try std.testing.expect(intent.motion.boost);
    try std.testing.expectEqual(input.drag, intent.motion.drag);
    try std.testing.expectEqual(Camera.UpMode.radial, intent.motion.up_mode.?);
    try std.testing.expect(!intent.motion.teleport);
    try std.testing.expect(intent.toggle_gui);
    try std.testing.expect(!intent.save_config);

    input.pressed.set(controls.up_world_y);
    try std.testing.expectEqual(Camera.UpMode.world_y, controls.map(input).motion.up_mode.?);
}
