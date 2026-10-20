//! Render orchestration: the renderer and its modes, the display pass, training-data
//! generation, scene presets, and the host mirrors of the shader uniforms.

const std = @import("std");

pub const Renderer = @import("render/Renderer.zig");
pub const ModeKernel = @import("render/ModeKernel.zig");
pub const GpuLuts = @import("render/GpuLuts.zig");
pub const Post = @import("render/Post.zig");
pub const View = @import("render/View.zig");
pub const DebugMode = @import("render/DebugMode.zig");
pub const PathTracer = @import("render/PathTracer.zig");
pub const RayMarcher = @import("render/RayMarcher.zig");
pub const NeuralMode = @import("render/NeuralMode.zig");
pub const GpuNetwork = @import("render/GpuNetwork.zig");
pub const BrunetonMode = @import("render/BrunetonMode.zig");
pub const Datagen = @import("render/Datagen.zig");
pub const GpuEarthMaps = @import("render/GpuEarthMaps.zig");
pub const TileResidency = @import("render/TileResidency.zig");
pub const scene = @import("render/scene.zig");
pub const uniforms = @import("render/uniforms.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("render/test_goldens.zig");
    _ = @import("render/test_raymarch.zig");
    _ = @import("render/test_renderer.zig");
    _ = @import("render/test_earth_maps.zig");
    _ = @import("render/test_datagen.zig");
    _ = @import("render/test_lut4d.zig");
    _ = @import("render/test_brdf.zig");
}
