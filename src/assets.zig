//! Runtime assets: spectral and camera-response LUTs, trained networks, the `bruneton`
//! mode's scattering table, the cloud bench's volume, NASA Earth maps with their GPU tiling, stb image
//! codecs, the datagen file format, and the tile download.

const std = @import("std");

pub const EarthMaps = @import("assets/EarthMaps.zig");
pub const DataRoot = @import("assets/DataRoot.zig");
pub const Image = @import("assets/Image.zig");
pub const Luts = @import("assets/Luts.zig");
pub const MapBounds = @import("assets/MapBounds.zig");
pub const Network = @import("assets/Network.zig");
pub const Lut4d = @import("assets/Lut4d.zig");
pub const CloudVolume = @import("assets/CloudVolume.zig");
pub const TilePlan = @import("assets/TilePlan.zig");
pub const TileSet = @import("assets/TileSet.zig");
pub const WindMap = @import("assets/WindMap.zig");
pub const datagen_format = @import("assets/datagen_format.zig");
pub const tile_download = @import("assets/tile_download.zig");

test {
    std.testing.refAllDecls(@This());
}
