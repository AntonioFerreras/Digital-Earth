//! Typed wrappers over the GPU half of libearth_native (native/include/earth_native.h).
//! Every wrapper owns exactly one native handle, maps `en_status` to `Error`, and logs
//! `en_last_error()` at the failing call so callers only propagate the error.

const std = @import("std");
const c = @import("c");

pub const Device = @import("gpu/Device.zig");
pub const Buffer = @import("gpu/Buffer.zig");
pub const Texture = @import("gpu/Texture.zig");
pub const Sampler = @import("gpu/Sampler.zig");
pub const Kernel = @import("gpu/Kernel.zig");
pub const CommandList = @import("gpu/CommandList.zig");
pub const Fence = @import("gpu/Fence.zig");
pub const Surface = @import("gpu/Surface.zig");

const log = std.log.scoped(.gpu);

/// One error per `en_status` failure code.
pub const Error = error{
    InvalidArgument,
    Unsupported,
    OutOfMemory,
    ShaderCompile,
    NotFound,
    DeviceLost,
    SurfaceOutOfDate,
    Internal,
};

/// Maps `status` to `Error`, logging the native message once. An out-of-date surface is
/// an expected event that the caller handles by reconfiguring, so it is not logged.
pub fn check(status: c.en_status, comptime operation: []const u8) Error!void {
    const err: Error = switch (status) {
        c.EN_OK => return,
        c.EN_ERR_SURFACE_OUT_OF_DATE => return error.SurfaceOutOfDate,
        c.EN_ERR_INVALID_ARGUMENT => error.InvalidArgument,
        c.EN_ERR_UNSUPPORTED => error.Unsupported,
        c.EN_ERR_OUT_OF_MEMORY => error.OutOfMemory,
        c.EN_ERR_SHADER_COMPILE => error.ShaderCompile,
        c.EN_ERR_NOT_FOUND => error.NotFound,
        c.EN_ERR_DEVICE_LOST => error.DeviceLost,
        else => error.Internal,
    };
    log.err(operation ++ " failed ({t}): {s}", .{ err, c.en_last_error() });
    return err;
}

/// Compile-time proof that a Zig mirror of a C enum or flag set matches the header:
/// enum field `foo` must equal `prefix ++ "FOO"`, and bool field `foo` of a packed flag
/// struct must sit at the bit of `prefix ++ "FOO"`.
pub fn assertMirrorsC(comptime T: type, comptime prefix: []const u8) void {
    comptime {
        switch (@typeInfo(T)) {
            .@"enum" => |info| for (info.field_names, info.field_values) |name, value| {
                if (value != @field(c, prefix ++ upper(name))) {
                    @compileError(@typeName(T) ++ "." ++ name ++ " differs from C");
                }
            },
            .@"struct" => |info| for (info.field_names, info.field_types) |name, FieldType| {
                if (FieldType != bool) continue;
                var flags: T = .{};
                @field(flags, name) = true;
                const bit: info.backing_integer.? = @bitCast(flags);
                if (bit != @field(c, prefix ++ upper(name))) {
                    @compileError(@typeName(T) ++ "." ++ name ++ " differs from C");
                }
            },
            else => @compileError("expected an enum or a packed flag struct"),
        }
    }
}

fn upper(comptime name: []const u8) []const u8 {
    comptime {
        var result: [name.len]u8 = undefined;
        _ = std.ascii.upperString(&result, name);
        const final = result;
        return &final;
    }
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("gpu/test_gpu.zig");
}
