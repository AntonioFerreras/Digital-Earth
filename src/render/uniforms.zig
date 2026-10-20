//! Host mirrors of shaders/shared/uniforms.slang. Field order, names, and
//! types must match the Slang declarations; gpu.Kernel checks offsets.

pub const FrameUniforms = extern struct {
    camera_position: [4]f32,
    camera_forward: [4]f32,
    camera_right: [4]f32,
    camera_up: [4]f32,
    sun_direction: [4]f32,
    frame: [4]u32,
};

pub const SceneFlags = packed struct(u32) {
    terrain: bool = false,
    clouds: bool = false,
    night_lights: bool = false,
    stars: bool = false,
    earth_maps: bool = false,
    canonicalize: bool = false,
    single_scattering: bool = false,
    _: u25 = 0,
};

pub const SceneUniforms = extern struct {
    config: [4]u32,
    media: [4]f32,
    surface: [4]f32,
};

/// Field names are those of `assets.EarthMaps.Role`.
pub const EarthMapUniforms = extern struct {
    albedo: [4]u32,
    topography: [4]u32,
    ocean_mask: [4]u32,
    clouds: [4]u32,
    bathymetry: [4]u32,
    night_lights: [4]u32,
    stars: [4]u32,
};

pub const DatagenFlags = packed struct(u32) {
    exclude_sun_disk: bool = false,
    _: u31 = 0,
};

pub const DatagenUniforms = extern struct {
    batch: [4]u32,
    dispatch: [4]u32,
};

pub const PostUniforms = extern struct {
    tone: [4]f32,
    state: [4]u32,
};
