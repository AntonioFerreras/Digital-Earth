//! The SDL3 application window. It owns SDL itself (`SDL_Init` to `SDL_Quit`), so there
//! is one per process, and it must outlive the surface and the UI built on it.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Io = std.Io;
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
    /// Cursor travel while the right button is held, as a fraction of the window size
    /// (x right, y down).
    drag: [2]f64 = .{ 0, 0 },
    /// The file chosen in the dialog from `showOpenDialog`; valid until the next one opens.
    opened_path: ?[]const u8 = null,
};

/// Upper bound on events handled per pump; the rest wait for the next frame.
const events_per_pump_max = 1024;

/// SDL may call the dialog back on another thread, even after the window is destroyed,
/// so the slot outlives every window.
var open_dialog: DialogSlot = .{};

/// Hands the path chosen in one dialog from SDL's callback, on any thread, to `pump`.
const DialogSlot = struct {
    state: std.atomic.Value(State) = .init(.idle),
    /// Written by the callback only while `open`, read by `pump` only once `ready`.
    path_buffer: [Io.Dir.max_path_bytes]u8 = undefined,
    path_len: usize = 0,

    const State = enum(u8) { idle, open, ready };

    /// Main thread only, which makes it the only writer of `open`.
    fn begin(slot: *DialogSlot) bool {
        return slot.state.cmpxchgStrong(.idle, .open, .monotonic, .monotonic) == null;
    }

    /// The returned path stays put until the next `begin`, which only the caller makes.
    fn take(slot: *DialogSlot) ?[]const u8 {
        if (slot.state.load(.acquire) != .ready) return null;
        slot.state.store(.idle, .monotonic);
        return slot.path_buffer[0..slot.path_len];
    }

    fn onChosen(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, filter: c_int) callconv(.c) void {
        _ = filter;
        const slot: *DialogSlot = @ptrCast(@alignCast(userdata));
        assert(slot.state.load(.monotonic) == .open);
        if (filelist == null) {
            log.err("file dialog failed: {s}", .{c.SDL_GetError()});
            slot.state.store(.idle, .release);
            return;
        }
        // An empty list or path means the user cancelled.
        const path = if (filelist[0] == null) "" else std.mem.span(filelist[0]);
        const outcome: State = if (path.len == 0) .idle else if (path.len > slot.path_buffer.len) too_long: {
            log.err("file dialog: path longer than {d} bytes", .{slot.path_buffer.len});
            break :too_long .idle;
        } else chosen: {
            @memcpy(slot.path_buffer[0..path.len], path);
            slot.path_len = path.len;
            break :chosen .ready;
        };
        slot.state.store(outcome, .release);
    }
};

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

/// Opens the native file-open dialog at `location` (the system's choice when null)
/// unless one is already open; `pump` reports the choice. `filters` and `location` must
/// stay valid until then.
pub fn showOpenDialog(window: Window, filters: []const c.SDL_DialogFileFilter, location: ?[*:0]const u8) void {
    if (!open_dialog.begin()) return;
    c.SDL_ShowOpenFileDialog(DialogSlot.onChosen, &open_dialog, window.sdl_window, filters.ptr, @intCast(filters.len), location, false);
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
            c.SDL_EVENT_MOUSE_MOTION => if ((event.motion.state & c.SDL_BUTTON_RMASK) != 0 and
                !imgui_io.*.WantCaptureMouse and window_width > 0 and window_height > 0)
            {
                input.drag[0] += event.motion.xrel / @as(f64, @floatFromInt(window_width));
                input.drag[1] += event.motion.yrel / @as(f64, @floatFromInt(window_height));
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
    input.opened_path = open_dialog.take();
    return input;
}

const testing = std.testing;

test "the dialog slot hands over a chosen path once, and nothing on cancel or error" {
    var slot: DialogSlot = .{};
    try testing.expect(slot.begin());
    try testing.expect(!slot.begin());
    try testing.expectEqual(null, slot.take());
    const chosen = [_][*c]const u8{ "/presets/florida.zon", null };
    DialogSlot.onChosen(&slot, &chosen, 0);
    try testing.expect(!slot.begin());
    try testing.expectEqualStrings("/presets/florida.zon", slot.take().?);
    try testing.expectEqual(null, slot.take());

    const cancelled = [_][*c]const u8{null};
    try testing.expect(slot.begin());
    DialogSlot.onChosen(&slot, &cancelled, -1);
    try testing.expectEqual(null, slot.take());

    try testing.expect(slot.begin());
    DialogSlot.onChosen(&slot, null, -1);
    try testing.expectEqual(null, slot.take());
    try testing.expect(slot.begin());
}
