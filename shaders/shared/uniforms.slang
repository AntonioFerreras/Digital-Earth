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
// With SCENE_EARTH_MAPS: albedo and water from the cube-face tiles (earth/tiles.slang).
public static const uint SCENE_TILES = 1u << 7;
// Temporary switches from the previous water to the physical model's parts: the
// Cox-Munk glint, the light scattered up out of the water, clear water in place of
// the albedo map's deep-ocean fill, and lakes calmer than the sea.
public static const uint SCENE_WATER_GLINT = 1u << 8;
public static const uint SCENE_WATER_BODY = 1u << 9;
public static const uint SCENE_CLEAR_DEEP_WATER = 1u << 10;
public static const uint SCENE_CALM_LAKES = 1u << 11;
// Temporary switches: the sea's wind from the wind map where it has one, and the
// procedural roughness texture over it.
public static const uint SCENE_WIND_MAP = 1u << 12;
public static const uint SCENE_ROUGHNESS_TEXTURE = 1u << 13;
// The clouds alone: no air and a black surface, to study the clouds' light.
public static const uint SCENE_CLOUDS_ONLY = 1u << 14;
// Temporary switch: the ray marcher marches every cloud shadow instead of reading
// distant ones from its shadow map.
public static const uint SCENE_MARCH_CLOUD_SHADOWS = 1u << 15;

public struct SceneUniforms {
    // x: SCENE_* flags, y: max bounces, z: first bounce with Russian roulette, w: unused.
    public uint4 config;
    // x: Mie turbidity, y: cloud extinction, z: cloud density, w: Rayleigh sea-level density.
    public float4 media;
    // x: constant sRGB albedo when SCENE_EARTH_MAPS is off, y: land height scale (m),
    // z: night-light radiance scale, w: star radiance scale.
    public float4 surface;
    // x: the sea's wind speed in m/s at 10 m where the wind map is off or holds none;
    // temporary, y: the roughness texture's strength and z its size, both factors on
    // their defaults; w: how far the wind map's speed replaces x, 0 to 1.
    public float4 water;
    // Temporary, x: the weight of the sea's slope tail, 0 to 1; y: the roughness
    // texture's streak length in wavelengths; z: its warp in coarsest wavelengths per
    // unit of noise; w: the factor on the slope variance the wind adds.
    public float4 glint;
    // The clouds' multiple scattering, after Nubis Cubed (clouds/multiple_scattering.slang):
    // x its strength; its attenuation rates per optical depth towards the sun, y for views
    // away from the sun and, for views towards it, z at a cloud's surface and w at
    // `cloud_profile.z` inside it.
    public float4 cloud_scattering;
    // x: how far the dimensional profile dims the multiply scattered light towards a
    // cloud's edge and the sky's light towards its core, 0 to 1; y and z: the depths
    // inside a cloud, as optical depths at full density, over which the profile rises
    // from 0 to 1 and the attenuation towards the sun relaxes; w: the secondary
    // scattering phase's HG g.
    public float4 cloud_profile;
};

// The cloud bench's sky shines; without it only the sun lights the cloud.
public static const uint CLOUD_BENCH_SKY = 1u << 0;
// The probes' walks estimate the sun's transmittance by ratio tracking, not a march
// (modes/cloud_bench_probes.slang).
public static const uint CLOUD_BENCH_RATIO_TRACKING = 1u << 1;
// The probes fit the sun's log fluence through order 3 instead of 2.
public static const uint CLOUD_BENCH_SUN_ORDER3 = 1u << 2;
// The approximation follows one scatter of its view at full resolution and reads the
// probes only past it (modes/cloud_bench.slang).
public static const uint CLOUD_BENCH_PROBE_GATHER = 1u << 3;
// The approximation takes the forward peak of its view's scatter as straight on, its
// light carried along the view at full resolution (modes/cloud_bench.slang).
public static const uint CLOUD_BENCH_FORWARD_PEAK = 1u << 4;
// The approximation scatters the probes' light by P1 instead of a von Mises-Fisher
// closure (modes/cloud_bench.slang).
public static const uint CLOUD_BENCH_P1_CLOSURE = 1u << 5;
// The probes lie in src/render/ProbeLayout.zig's adaptive layout, not its dense grid;
// read by src/render/CloudBench.zig alone.
public static const uint CLOUD_BENCH_ADAPTIVE_PROBES = 1u << 6;
// The probes bake from light traced forward into the cloud instead of walks from
// them, whose sky transfer then holds only the sky's light before it scatters
// (modes/cloud_bench_probes.slang).
public static const uint CLOUD_BENCH_PHOTONS = 1u << 7;
// The photons add exact path lengths per leaf instead of points along the path
// (modes/cloud_bench_probes.slang).
public static const uint CLOUD_BENCH_LEAF_LENGTHS = 1u << 8;


// The cloud bench (modes/cloud_bench.slang): one density grid in a frame of its own,
// "cloud units", axis-aligned with the planet-centred frame.
public struct CloudBenchUniforms {
    // xyz: where the cloud frame's origin lies, planet-centred metres; w: metres per
    // cloud unit.
    public float4 origin;
    // xyz: the grid's minimum corner, cloud units; w: the voxel edge.
    public float4 grid_min;
    // xyz: voxels per axis; w: voxels per majorant cell edge.
    public uint4 grid_size;
    // x: extinction per metre at unit density; yzw: unused.
    public float4 medium;
    // x: the approximation's view steps, as a multiple of those at MARCH_STEP_VOXELS;
    // y, z: the walk levels of its march along the view and towards the sun; w: unused.
    public float4 march;
    // x: the view (CLOUD_BENCH_VIEW_*), y: bounces per path, z: CLOUD_BENCH_* flags,
    // w: the probes' walks at which their bake stops.
    public uint4 options;
    // x: the reference's samples before this frame, y: the approximation's, z: the
    // probes' walks; w: the photons' rounds in a dispatch of their trace.
    public uint4 samples;
    // src/render/ProbeLayout.zig's. xyz: lattice point zero, cloud units; w: the
    // adaptive layout's ratio of spacing to depth inside the cloud.
    public float4 probe_origin;
    // xyz: the finest cells' edges, cloud units; w: unused.
    public float4 probe_spacing;
    // xyz: the finest cells per axis; w: the adaptive layout's probe budget.
    public uint4 probe_cells;
    // x: probes, y: leaves, z: the bake's rounds per frame; w: unused.
    public uint4 probe_counts;
};

public static const uint CLOUD_BENCH_VIEW_REFERENCE = 0;
public static const uint CLOUD_BENCH_VIEW_APPROXIMATION = 1;
// The reference left of the image's centre, the approximation right of it.
public static const uint CLOUD_BENCH_VIEW_SPLIT = 2;
// The reference tinted by the approximation's error: red where it is brighter, blue
// where darker, fully at a factor of two.
public static const uint CLOUD_BENCH_VIEW_ERROR = 3;
// The approximation's cloud opacity along the view, grey; python/clouds/score.py
// takes it to find the cloud.
public static const uint CLOUD_BENCH_VIEW_COVERAGE = 4;

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

// The cube-face tile quadtrees (earth/tiles.slang): per layer, x the last level and
// y the first indirection entry of its face 0; height layer first, then surface.
public struct TileUniforms {
    public uint4 height;
    public uint4 surface;
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
    // x: 1 to tone map with OpenDRT, 0 to clip the exposed linear colour; yzw: unused.
    public uint4 options;
};
