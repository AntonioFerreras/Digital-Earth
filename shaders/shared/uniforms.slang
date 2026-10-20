// Host-shared parameter blocks, mirrored field for field by src/render/uniforms.zig.
// Only 4-byte scalars and 4-vectors so std140, scalar, and C layouts coincide;
// gpu.Kernel verifies every offset against reflection at creation.

module uniforms;

public struct FrameUniforms {
    // xyz: camera position, metres, planet-centred. w: unused.
    public float4 camera_position;
    // xyz: unit view direction. w: image-plane half height at unit distance.
    public float4 camera_forward;
    // xyz: unit right vector. w: half width / half height (aspect * aspect scale).
    public float4 camera_right;
    // xyz: unit up vector orthogonal to forward and right. w: unused.
    public float4 camera_up;
    // xyz: unit direction towards the sun (its centre). w: unused.
    public float4 sun_direction;
    // x: frame index since the last reset, y: image width, z: image height,
    // w: seed, added to x in the random-number key.
    public uint4 frame;
};

public static const uint SCENE_TERRAIN = 1u << 0;
public static const uint SCENE_CLOUDS = 1u << 1;
public static const uint SCENE_NIGHT_LIGHTS = 1u << 2;
public static const uint SCENE_STARS = 1u << 3;
public static const uint SCENE_EARTH_MAPS = 1u << 4;
// Round-trip every camera ray through the uvwz parameterization (symmetric scene).
public static const uint SCENE_CANONICALIZE = 1u << 5;
// Light scatters once only (with `config.y` = 1): the ray marcher drops its
// multiple-scattering table. For accuracy studies of the modes against each other.
public static const uint SCENE_SINGLE_SCATTERING = 1u << 6;

public struct SceneUniforms {
    // x: SCENE_* flags, y: max bounces, z: first bounce with Russian roulette, w: unused.
    public uint4 config;
    // x: Mie turbidity, y: cloud extinction, z: cloud density, w: Rayleigh sea-level density.
    public float4 media;
    // x: constant sRGB albedo when SCENE_EARTH_MAPS is off, y: land height scale (m),
    // z: night-light radiance scale, w: star radiance scale.
    public float4 surface;
};

// Tile layout of each Earth map (src/assets/TilePlan.zig), one field per
// assets.EarthMaps.Role: x map width, y map height, z tile width, w apron.
public struct EarthMapUniforms {
    public uint4 albedo;
    public uint4 topography;
    public uint4 ocean_mask;
    public uint4 clouds;
    public uint4 bathymetry;
    public uint4 night_lights;
    public uint4 stars;
    // The bound map's (width, height, 0, 0).
    public uint4 bounds;
};

// Leave the sun disk out of the escaping rays' radiance (modes/datagen.slang).
public static const uint DATAGEN_EXCLUDE_SUN_DISK = 1u << 0;

public struct DatagenUniforms {
    // x: global index of the batch's first point, y: paths per point,
    // z: paths per chunk, w: chunks per point.
    public uint4 batch;
    // x: first `accum` row of this dispatch, y: groups in the batch (points x chunks),
    // z: DATAGEN_* flags, w: unused.
    public uint4 dispatch;
};

public struct PostUniforms {
    // x: exposure (stops), y: gamma, z: OpenDRT saturation, w: OpenDRT contrast.
    public float4 tone;
    // x: samples accumulated, y: CRF index, z: CRF count, w: OpenDRT tonescale preset.
    public uint4 state;
};
