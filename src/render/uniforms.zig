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
    /// Surface albedo and water from the cube-face tiles instead of the Earth maps.
    tiles: bool = false,
    water_glint: bool = false,
    water_body: bool = false,
    clear_deep_water: bool = false,
    calm_lakes: bool = false,
    wind_map: bool = false,
    roughness_texture: bool = false,
    clouds_only: bool = false,
    march_cloud_shadows: bool = false,
    _: u16 = 0,
};

pub const SceneUniforms = extern struct {
    config: [4]u32,
    media: [4]f32,
    surface: [4]f32,
    water: [4]f32,
    glint: [4]f32,
    cloud_scattering: [4]f32,
    cloud_profile: [4]f32,
};

pub const CloudBenchFlags = packed struct(u32) {
    sky: bool = false,
    ratio_tracking: bool = false,
    sun_order3: bool = false,
    probe_gather: bool = false,
    forward_peak: bool = false,
    p1_closure: bool = false,
    adaptive_probes: bool = false,
    photons: bool = false,
    leaf_lengths: bool = false,
    _: u23 = 0,
};

pub const CloudBenchUniforms = extern struct {
    origin: [4]f32,
    grid_min: [4]f32,
    grid_size: [4]u32,
    medium: [4]f32,
    march: [4]f32,
    options: [4]u32,
    samples: [4]u32,
    probe_origin: [4]f32,
    probe_spacing: [4]f32,
    probe_cells: [4]u32,
    probe_counts: [4]u32,
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
    /// The bound map's (width, height, 0, 0).
    bounds: [4]u32,
};

/// Per `TileSet.Layer`: the last level, and the first indirection entry of its face 0.
pub const TileUniforms = extern struct {
    height: [4]u32,
    surface: [4]u32,
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
    options: [4]u32,
};
