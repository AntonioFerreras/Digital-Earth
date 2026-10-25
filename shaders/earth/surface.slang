// The planet surface as both render modes see it: intersection, sun
// visibility, and the material at a hit. Every mode reaches the surface only
// through this module, so the heightfield and the Earth maps change in one place.
//
// With SCENE_TERRAIN the surface is the sphere displaced outwards by the
// topography map times `scene.surface.y`; otherwise it is the bare sphere. With
// SCENE_EARTH_MAPS the material comes from the maps; otherwise it is the
// constant grey `scene.surface.x`.

module surface;

import common.constants;
import common.geometry;
import common.sampling;
import earth.brdf;
import earth.map_bounds;
import earth.maps;
import earth.material;
import common.ray_cone;
import earth.tile_terrain;
import earth.tiles;
import shared.earth_bindings;
import shared.uniforms;
import spectral.planck;
import spectral.uplift;
import common.noise;

public static const float SURFACE_MISS = TILE_SURFACE_MISS;

public struct SurfaceSample
{
    // Unit shading normal.
    public float3 normal;
    // Graded albedo, encoded sRGB in [0, 1] as the Taichi renderer used it; of land
    // alone when the water body has its own colour.
    public float3 albedo_srgb;
    // The albedo map's linear colour, for the water body's.
    public float3 water_rgb;
    // How much of the water body takes the clear-ocean spectrum instead.
    public float clear_water;
    public Water water;
    // Night-light intensity in [0, 1].
    public float emissive;
};

// Water whose surface lies this many metres above sea level is a lake.
static const float LAKE_SURFACE_MIN = 1.0;
// Lakes are sheltered and short of fetch, so calmer than the open sea.
static const float LAKE_WIND_SPEED = 3.0;

static const uint SPHERE_TRACE_STEPS = 250;
// Skips and field steps together: a chord of the terrain shell spans at most
// 2 sqrt(2 PLANET_RADIUS h) for its height h, some 630 skips of REACH_MIN.
static const uint MARCH_ITERATIONS_MAX = 1024;
// A march stops once the field is within this fraction of the distance marched
// inside the atmosphere. Measured from the camera instead, the tolerance grows to
// kilometres from orbit and stops hits in bands above the sea surface.
static const float SPHERE_TRACE_RELATIVE_EPSILON = 1e-4;

// Not a true distance bound on steep slopes; the march and the normals use it
// exactly as the original did, so overshoots land where they did there.
float land_sdf(float3 position, SceneUniforms scene)
{
    float height = sample_tiled_exact(earth_topography, earth_maps.topography, sphere_uv(normalize(position)));
    return length(position) - PLANET_RADIUS - scene.surface.y * height;
}

// Backward differences along the world axes, one step being half an equatorial
// texel of the topography map.
float3 land_normal(float3 position, SceneUniforms scene)
{
    float e = PI * PLANET_RADIUS / float(earth_maps.topography.x);
    float d = land_sdf(position, scene);
    return normalize(d - float3(land_sdf(position - float3(e, 0.0, 0.0), scene),
                                land_sdf(position - float3(0.0, e, 0.0), scene),
                                land_sdf(position - float3(0.0, 0.0, e), scene)));
}

// Distance along the unit `dir` to the first surface hit no farther than `t_max`,
// or SURFACE_MISS; `cone` sets the detail of the tiles the march reads.
public float intersect_surface(float3 origin, float3 dir, SceneUniforms scene, RayCone cone, float t_max = FLOAT_MAX)
{
    if ((scene.config.x & (SCENE_TERRAIN | SCENE_TILES)) == (SCENE_TERRAIN | SCENE_TILES))
        return tile_intersect(origin, dir, cone, t_max);
    float2 ground = ray_sphere(origin, dir, PLANET_RADIUS);
    bool ground_hit = ground.x > 0.0 && ground.x <= t_max;
    if ((scene.config.x & SCENE_TERRAIN) == 0)
        return ground_hit ? ground.x : SURFACE_MISS;
    // The terrain lies between the planet sphere and the sphere lifted by the land
    // height scale; at or above sea level, it is hit no later than the planet sphere.
    float2 shell = ray_sphere(origin, dir, PLANET_RADIUS + scene.surface.y);
    if (shell.y <= 0.0)
        return SURFACE_MISS;
    float end = ground_hit ? ground.x : min(shell.y, t_max);
    float atmosphere_entry = max(ray_sphere(origin, dir, ATMOSPHERE_RADIUS).x, 0.0);
    float t = max(shell.x, 0.0);
    // Skips advance at least REACH_MIN each along a chord of the shell, which
    // MARCH_ITERATIONS_MAX covers, so only the field's steps count against
    // SPHERE_TRACE_STEPS.
    uint step_count = 0;
    for (uint iteration = 0; iteration < MARCH_ITERATIONS_MAX && step_count < SPHERE_TRACE_STEPS; iteration++)
    {
        float3 position = origin + t * dir;
        MapBound bound = MapBound(sphere_uv(normalize(position)));
        float skip = min(t + bound.reach, end);
        // Under a bound of 0 the terrain is the planet sphere, which `end` already holds.
        if (bound.topography == 0.0 ||
            radius_range(origin, dir, t, skip).x > PLANET_RADIUS + scene.surface.y * bound.topography)
        {
            t = skip;
            if (t >= end)
                break;
            continue;
        }
        float d = land_sdf(position, scene);
        t += d;
        step_count += 1;
        if (t > end || abs(d) < (t - atmosphere_entry) * SPHERE_TRACE_RELATIVE_EPSILON)
            break;
    }
    if (t < end)
        return t;
    return ground_hit ? ground.x : SURFACE_MISS;
}

// Whether the sun direction `dir` is unoccluded by the surface from `origin`.
public bool surface_visible(float3 origin, float3 dir, SceneUniforms scene, RayCone cone)
{
    if ((scene.config.x & SCENE_TERRAIN) != 0)
        return intersect_surface(origin, dir, scene, cone) < 0.0;
    return ray_sphere(origin, dir, PLANET_RADIUS).x < 0.0;
}

// The surface lifted along its normal far enough to start shadow and bounce rays
// without self-intersection.
public float3 offset_surface_point(float3 position, SceneUniforms scene)
{
    return position * (1.0 + 1e-4 * scene.surface.y / 12000.0);
}

// Above any 10 m wind the wind map holds; python/tiles/wind.py encodes speeds over it.
static const float WIND_SPEED_MAX = 32.0;

// The 10 m wind at unit `up`: xy its eastward and northward components and z its
// speed in m/s, w 1 where the wind map holds it and 0 where there is none.
float4 wind_at(float3 up)
{
    float4 cell = earth_wind.SampleLevel(earth_wind_sampler, sphere_uv(up), 0.0);
    return float4((cell.yz * 2.0 - 1.0) * WIND_SPEED_MAX, cell.x * WIND_SPEED_MAX, cell.w);
}

// The unit tangent at unit `up` the `wind` of wind_at blows along, or 0 where the
// map has none or it is too weak to set a direction.
float3 wind_direction(float3 up, float4 wind)
{
    if (!(wind.w > 0.0 && wind.z > 0.5))
        return 0.0;
    float3 east = float3(up.z, 0.0, -up.x);
    east *= rsqrt(max(dot(east, east), 1e-12));
    return normalize(wind.x * east + wind.y * cross(up, east));
}

// The roughness texture's coarsest wavelength in metres, its octave count, and the
// wavelength ratio and amplitude ratio between octaves: calm and rough patches from
// the scale of weather down to that of slicks, with no one scale standing out.
static const float ROUGHNESS_WAVELENGTH = 80e3;
static const uint ROUGHNESS_OCTAVES = 5;
static const float ROUGHNESS_LACUNARITY = 2.3;
static const float ROUGHNESS_GAIN = 0.6;
// Taps averaged along the wind in each octave but the coarsest, spread over the
// streak length and weighted by a sine-squared bell, which draw patches out into
// streaks; the coarsest octave stays round, as wind patches are.
static const uint STREAK_TAPS = 8;
// The standard deviation of one gradient_noise sample.
static const float GRADIENT_NOISE_DEVIATION = 0.191;

// A factor on the sea's slope variance for the slicks, wind shadows, and gust
// patches the wind field cannot resolve: log-normal fBm with mean 1, streaked
// `along` the wind where it has a direction over `streak_length` wavelengths, its
// coordinates bent by `warp` coarsest wavelengths per unit of noise so that patches
// take irregular shapes instead of the lattice's,
// strongest in light wind, where such patches survive, and fading with octaves finer
// than the `footprint` metres a sample spans, which average to the mean.
float roughness_texture(float3 position, float3 along, float wind_speed, float footprint, float strength, float size, float streak_length, float warp)
{
    float coarsest = ROUGHNESS_WAVELENGTH * size;
    float3 q = position / coarsest;
    float3 bend = float3(gradient_noise(q + 31.0), gradient_noise(q + 57.0), gradient_noise(q + 83.0));
    float3 warped = position + warp * coarsest * bend;
    bool streaked = any(along != 0.0) && streak_length > 0.0;
    // Averaging along a streak keeps 1 / sqrt(1 + 0.75 L^2) of the variance for a
    // streak L wavelengths long, fitted to within 10% over 0.5 to 6.
    float streak_variance = 1.0 / sqrt(1.0 + 0.75 * streak_length * streak_length);
    float noise = 0.0;
    float variance = 0.0;
    float wavelength = coarsest;
    float amplitude = 1.0;
    for (uint octave = 0; octave < ROUGHNESS_OCTAVES; octave++)
    {
        uint taps = streaked && octave > 0 ? STREAK_TAPS : 1;
        variance += amplitude * amplitude * (taps > 1 ? streak_variance : 1.0);
        float resolved = saturate(wavelength / (4.0 * footprint) - 0.5);
        if (resolved > 0.0)
        {
            float streak = 0.0;
            for (uint tap = 0; tap < taps; tap++)
            {
                float offset = taps > 1 ? (float(tap) / float(taps - 1) - 0.5) * streak_length : 0.0;
                float bell = sin(PI * (float(tap) + 0.5) / float(taps));
                float weight = taps > 1 ? bell * bell * 2.0 / float(taps) : 1.0;
                streak += weight * gradient_noise((warped + offset * wavelength * along) / wavelength + float(octave) * 17.0);
            }
            noise += amplitude * resolved * streak;
        }
        wavelength /= ROUGHNESS_LACUNARITY;
        amplitude *= ROUGHNESS_GAIN;
    }
    noise /= GRADIENT_NOISE_DEVIATION * sqrt(variance);
    // The log factor's standard deviation.
    float deviation = strength * lerp(0.8, 0.12, smoothstep(3.0, 10.0, wind_speed));
    return exp(deviation * noise - 0.5 * deviation * deviation);
}

// The material at `position`, read at the detail of `footprint`, the ray's cone there.
public SurfaceSample sample_surface(float3 position, RayCone footprint, SceneUniforms scene)
{
    float3 up = normalize(position);
    SurfaceSample s;
    bool terrain = (scene.config.x & SCENE_TERRAIN) != 0;
    if (terrain && (scene.config.x & SCENE_TILES) != 0)
        s.normal = tile_normal(position, footprint);
    else
        s.normal = terrain ? land_normal(position, scene) : up;
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
    {
        float2 uv = sphere_uv(up);
        bool tiles = (scene.config.x & SCENE_TILES) != 0;
        uint level = tile_level(footprint.width);
        s.water.fraction = tiles ? tile_water(up, level, footprint.streams) : sample_tiled(earth_ocean_mask, earth_maps.ocean_mask, uv);
        float3 albedo = tiles ? tile_albedo(up, level, footprint.streams) : sample_tiled(earth_albedo, earth_maps.albedo, uv).rgb;
        s.water.body = (scene.config.x & SCENE_WATER_BODY) != 0;
        s.albedo_srgb = grade_albedo(albedo, s.water.body ? 0.0 : s.water.fraction);
        s.water_rgb = srgb_eotf(albedo);
        s.clear_water = (scene.config.x & SCENE_CLEAR_DEEP_WATER) != 0 ? deep_ocean_fill(albedo) : 0.0;
        s.water.bathymetry = sample_tiled(earth_bathymetry, earth_maps.bathymetry, uv);
        s.water.glint = (scene.config.x & SCENE_WATER_GLINT) != 0;
        float4 wind = (scene.config.x & SCENE_WIND_MAP) != 0 ? wind_at(up) : 0.0;
        // From the data, not the hit: a march stops anywhere within its tolerance above
        // the sea, which far away exceeds a lake's height.
        float surface_height = tiles ? tile_height(up, level, false) : scene.surface.y * sample_tiled_exact(earth_topography, earth_maps.topography, uv);
        bool lake = (scene.config.x & SCENE_CALM_LAKES) != 0 && surface_height > LAKE_SURFACE_MIN;
        s.water.wind_speed = lake ? LAKE_WIND_SPEED : lerp(scene.water.x, wind.z, wind.w * scene.water.w);
        s.water.along = lake ? 0.0 : wind_direction(up, wind);
        s.water.tail = scene.glint.x;
        s.water.wind_roughness = scene.glint.w;
        s.water.slope_scale = (scene.config.x & SCENE_ROUGHNESS_TEXTURE) != 0
                                  ? roughness_texture(position, s.water.along, s.water.wind_speed, footprint.width, scene.water.y, scene.water.z, scene.glint.y, scene.glint.z)
                                  : 1.0;
        s.emissive = sample_tiled(earth_night_lights, earth_maps.night_lights, uv);
    }
    else
    {
        s.albedo_srgb = scene.surface.xxx;
        s.water_rgb = 0.0;
        s.clear_water = 0.0;
        s.water = { 0.0, 0.0, 1.0, float3(0.0), 0.0, 1.0, 1.0, false, false };
        s.emissive = 0.0;
    }
    return s;
}

// Spectral reflectances of `s` at `wavelength` nm, uplifted through `basis`.
public Reflectance surface_reflectance(SurfaceSample s, StructuredBuffer<float> basis, float wavelength)
{
    Reflectance albedo;
    albedo.land = srgb_to_spectrum(basis, s.albedo_srgb, wavelength);
    albedo.water = s.water.body ? lerp(srgb_to_spectrum(basis, s.water_rgb, wavelength), clear_ocean_reflectance(wavelength), s.clear_water) : 0.0;
    return albedo;
}

// Reflectance of `s` with spectral `albedo` for unit `v` and `l` pointing away
// from the surface. Without the Earth maps the ground is Lambertian at half its
// albedo, the shading the symmetric scene (and the MLP trained on it) used.
public SurfaceResponse surface_brdf(SurfaceSample s, Reflectance albedo, float3 v, float3 l, SceneUniforms scene)
{
    SurfaceResponse response;
    if ((scene.config.x & SCENE_CLOUDS_ONLY) != 0)
    {
        response.brdf = 0.0;
        response.n_dot_l = saturate(dot(s.normal, l));
        return response;
    }
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
        return earth_brdf(albedo, s.water, v, s.normal, l);
    response.brdf = albedo.land / PI * 0.5;
    response.n_dot_l = saturate(dot(s.normal, l));
    return response;
}

// A bounce direction from unit `v` off `s`, sampled in proportion to its
// reflectance where the BRDF has a sharp lobe.
public BounceSample sample_bounce(SurfaceSample s, float3 v, float3 u, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
        return sample_earth_bounce(u, s.water, v, s.normal);
    BounceSample bounce;
    bounce.dir = sample_hemisphere_cosine(u.xy, s.normal);
    bounce.pdf = saturate(dot(s.normal, bounce.dir)) / PI;
    return bounce;
}

// Spectral radiance the night lights of `s` emit.
public float surface_emission(SurfaceSample s, float wavelength, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_NIGHT_LIGHTS) == 0)
        return 0.0;
    return s.emissive * planck(NIGHT_LIGHT_TEMPERATURE, wavelength) * scene.surface.z;
}
