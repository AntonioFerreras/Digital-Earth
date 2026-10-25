//! The SDL3 application window. It owns SDL itself (`SDL_Init` to `SDL_Quit`), so there
//! is one per process, and it must outlive the surface and the UI built on it.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const c = @import("c");

const Window = @This();

sdl_window: *c.SDL_Window,

const log = std.log.scoped(.platform);

pub const Error = error{ SdlInit, WindowCreate, NativeHandleUnavailable };

/// Keyboard state indexed by `SDL_Scancode`.
pub const KeySet = std.bit_set.Static(c.SDL_SCANCODE_COUNT);

/// What happened since the previous `pump`.
pub const Input = struct {
    quit: bool = false,
    /// The drawable size changed; the surface needs reconfiguring.
    resized: bool = false,
    held: KeySet = .empty,
    /// Pressed since the previous pump, auto-repeat excluded.
    pressed: KeySet = .empty,
    /// Cursor travel while the left button is held, as a fraction of the window size
    /// (x right, y down).
    drag: [2]f64 = .{ 0, 0 },
    /// Cursor travel while the right button is held, in pixels (x right, y down).
    right_drag: [2]f64 = .{ 0, 0 },
    right_held: bool = false,
};

/// Upper bound on events handled per pump; the rest wait for the next frame.
const events_per_pump_max = 1024;

pub fn init(title: [*:0]const u8, size: [2]u32) Error!Window {
    assert(size[0] > 0);
    assert(size[1] > 0);
    // slang-rhi has no Wayland surface, so Linux goes through X11 (XWayland).
    if (builtin.os.tag == .linux) _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "x11");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) {
        log.err("SDL_Init failed: {s}", .{c.SDL_GetError()});
        return error.SdlInit;
    }
    errdefer c.SDL_Quit();
    const flags = c.SDL_WINDOW_RESIZABLE | c.SDL_WINDOW_HIGH_PIXEL_DENSITY;
    const sdl_window = c.SDL_CreateWindow(title, @intCast(size[0]), @intCast(size[1]), flags) orelse {
        log.err("SDL_CreateWindow failed: {s}", .{c.SDL_GetError()});
        return error.WindowCreate;
    };
    return .{ .sdl_window = sdl_window };
}

pub fn deinit(window: Window) void {
    c.SDL_DestroyWindow(window.sdl_window);
    c.SDL_Quit();
}

/// The drawable size in pixels, which differs from the window size on high-DPI displays.
/// Zero while minimized.
pub fn pixelSize(window: Window) [2]u32 {
    var width: c_int = 0;
    var height: c_int = 0;
    _ = c.SDL_GetWindowSizeInPixels(window.sdl_window, &width, &height);
    return .{ @intCast(width), @intCast(height) };
}

pub fn nativeHandle(window: Window) Error!c.en_window_handle {
    const properties = c.SDL_GetWindowProperties(window.sdl_window);
    const handle: c.en_window_handle = switch (builtin.os.tag) {
        .linux => .{
            .kind = c.EN_WINDOW_XLIB,
            .display = c.SDL_GetPointerProperty(properties, c.SDL_PROP_WINDOW_X11_DISPLAY_POINTER, null),
            .window = @intCast(c.SDL_GetNumberProperty(properties, c.SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)),
        },
        .windows => .{
            .kind = c.EN_WINDOW_WIN32,
            .window = @intFromPtr(c.SDL_GetPointerProperty(properties, c.SDL_PROP_WINDOW_WIN32_HWND_POINTER, null)),
        },
        .macos => .{
            .kind = c.EN_WINDOW_COCOA,
            .window = @intFromPtr(c.SDL_GetPointerProperty(properties, c.SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, null)),
        },
        else => @compileError("unsupported windowing platform"),
    };
    if (handle.window == 0) {
        log.err("no native window handle from SDL video driver '{s}'", .{c.SDL_GetCurrentVideoDriver()});
        return error.NativeHandleUnavailable;
    }
    return handle;
}

/// Drains pending events, forwarding each to ImGui, and returns the input the
/// application may act on: keyboard and mouse input ImGui claims is withheld.
/// Requires the ImGui context from `en_ui_init`.
pub fn pump(window: Window) Input {
    var input: Input = .{};
    var window_width: c_int = 0;
    var window_height: c_int = 0;
    _ = c.SDL_GetWindowSize(window.sdl_window, &window_width, &window_height);
    const imgui_io = c.ImGui_GetIO();
    var event: c.SDL_Event = undefined;
    var event_count: u32 = 0;
    while (event_count < events_per_pump_max and c.SDL_PollEvent(&event)) : (event_count += 1) {
        c.en_ui_process_event(&event);
        switch (event.type) {
            c.SDL_EVENT_QUIT, c.SDL_EVENT_WINDOW_CLOSE_REQUESTED => input.quit = true,
            c.SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED => input.resized = true,
            c.SDL_EVENT_KEY_DOWN => if (!event.key.repeat and !imgui_io.*.WantCaptureKeyboard) {
                input.pressed.set(event.key.scancode);
            },
            // The cursor hides while the left button looks around, and comes back where it was.
            c.SDL_EVENT_MOUSE_BUTTON_DOWN => if (event.button.button == c.SDL_BUTTON_LEFT and !imgui_io.*.WantCaptureMouse) {
                _ = c.SDL_SetWindowRelativeMouseMode(window.sdl_window, true);
            },
            c.SDL_EVENT_MOUSE_BUTTON_UP => if (event.button.button == c.SDL_BUTTON_LEFT) {
                _ = c.SDL_SetWindowRelativeMouseMode(window.sdl_window, false);
            },
            c.SDL_EVENT_MOUSE_MOTION => if (!imgui_io.*.WantCaptureMouse) {
                if ((event.motion.state & c.SDL_BUTTON_LMASK) != 0 and window_width > 0 and window_height > 0) {
                    input.drag[0] += event.motion.xrel / @as(f64, @floatFromInt(window_width));
                    input.drag[1] += event.motion.yrel / @as(f64, @floatFromInt(window_height));
                }
                if ((event.motion.state & c.SDL_BUTTON_RMASK) != 0) {
                    input.right_drag[0] += event.motion.xrel;
                    input.right_drag[1] += event.motion.yrel;
                }
            },
            else => {},
        }
    }
    if (!imgui_io.*.WantCaptureKeyboard) {
        var key_count: c_int = 0;
        const keys = c.SDL_GetKeyboardState(&key_count)[0..@intCast(key_count)];
        assert(keys.len <= KeySet.bit_length);
        for (keys, 0..) |down, scancode| input.held.setValue(scancode, down);
    }
    input.right_held = (c.SDL_GetMouseState(null, null) & c.SDL_BUTTON_RMASK) != 0;
    return input;
}
