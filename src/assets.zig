//! Runtime assets: spectral and camera-response LUTs, trained networks, the `bruneton`
//! mode's scattering table, NASA Earth maps with their GPU tiling, stb image
//! codecs, and the datagen file format.

const std = @import("std");

pub const EarthMaps = @import("assets/EarthMaps.zig");
pub const DataRoot = @import("assets/DataRoot.zig");
pub const Image = @import("assets/Image.zig");
pub const Luts = @import("assets/Luts.zig");
pub const Network = @import("assets/Network.zig");
pub const Lut4d = @import("assets/Lut4d.zig");
pub const TilePlan = @import("assets/TilePlan.zig");
pub const datagen_format = @import("assets/datagen_format.zig");

test {
    std.testing.refAllDecls(@This());
}
