// The terrain of the cube-face tiles: ray intersection, sun visibility, and normals over
// the finest resident height tiles. A tile's edges lie on planes through the planet
// centre, so a ray leaves a tile where it crosses one of four planes, and the march
// jumps there whenever the ray stays above the tile's highest sample until then.
// Heights below sea level are sea floor under water and do not stop rays.

module tile_terrain;

import common.constants;
import common.geometry;
import common.ray_cone;
import earth.tiles;

// Above every summit, so the march starts on the shell that holds all terrain.
static const float TERRAIN_TOP = 9000.0;
// Field steps, as the equirectangular march allows; jumps across tiles do not count.
static const uint FIELD_STEPS_MAX = 250;
// Jumps and field steps together: a chord of the shell crosses at most a few hundred
// tiles at the levels a march samples.
static const uint ITERATIONS_MAX = 1024;
// A march stops once the field is within this fraction of the distance marched inside
// the atmosphere.
static const float RELATIVE_EPSILON = 1e-4;
public static const float TILE_SURFACE_MISS = -1.0;

// The surface height at `hit`: the terrain, or the sea where it lies below.
float surface_height(TileHit hit)
{
    return max(tile_height_at(hit), 0.0);
}

// Where the ray `origin + t dir` leaves `hit`'s tile beyond `t`, or FLOAT_MAX.
float tile_exit(float3 origin, float3 dir, TileHit hit, float t)
{
    FaceFrame frame = FaceFrame(hit.face);
    float size = 1.0 / float(1u << hit.level);
    float2 low = float2(hit.tile) * size;
    float2 high = low + size;
    // Each normal points out of the tile: inside points have a negative dot product.
    float3 normals[4] = { frame.edge_plane(frame.s_axis, high.x), -frame.edge_plane(frame.s_axis, low.x),
                          frame.edge_plane(frame.t_axis, high.y), -frame.edge_plane(frame.t_axis, low.y) };
    float exit = FLOAT_MAX;
    for (int k = 0; k < 4; k++)
    {
        float rate = dot(dir, normals[k]);
        if (rate > 0.0)
        {
            float crossing = -dot(origin, normals[k]) / rate;
            if (crossing > t)
                exit = min(exit, crossing);
        }
    }
    return exit;
}

// Distance along the unit `dir` to the first terrain hit no farther than `t_max`, or
// TILE_SURFACE_MISS; `cone` sets the level each sample asks for.
public float tile_intersect(float3 origin, float3 dir, RayCone cone, float t_max)
{
    float2 ground = ray_sphere(origin, dir, PLANET_RADIUS);
    bool ground_hit = ground.x > 0.0 && ground.x <= t_max;
    float2 shell = ray_sphere(origin, dir, PLANET_RADIUS + TERRAIN_TOP);
    if (shell.y <= 0.0)
        return TILE_SURFACE_MISS;
    float end = ground_hit ? ground.x : min(shell.y, t_max);
    float atmosphere_entry = max(ray_sphere(origin, dir, ATMOSPHERE_RADIUS).x, 0.0);
    float t = max(shell.x, 0.0);
    uint field_steps = 0;
    for (uint iteration = 0; iteration < ITERATIONS_MAX && field_steps < FIELD_STEPS_MAX; iteration++)
    {
        float3 position = origin + t * dir;
        TileHit hit = TileHit(earth_tiles.height, FacePoint(normalize(position)), tile_level(cone.width_at(t)), cone.streams);
        float top = PLANET_RADIUS + max(float(earth_tile_height_max[hit.index]), 0.0);
        float exit = min(tile_exit(origin, dir, hit, t), end);
        if (radius_range(origin, dir, t, exit).x > top)
        {
            // At least a metre, in case rounding puts the crossing at `t`.
            t = max(exit, t + 1.0);
            if (t >= end)
                break;
            continue;
        }
        float d = length(position) - PLANET_RADIUS - surface_height(hit);
        t += d;
        field_steps += 1;
        if (t > end || abs(d) < (t - atmosphere_entry) * RELATIVE_EPSILON)
            break;
    }
    if (t < end)
        return t;
    return ground_hit ? ground.x : TILE_SURFACE_MISS;
}

// The terrain normal at `position` from central differences of the surface one texel
// apart, at the level `footprint`'s width asks for.
public float3 tile_normal(float3 position, RayCone footprint)
{
    float3 up = normalize(position);
    TileHit hit = TileHit(earth_tiles.height, FacePoint(up), tile_level(footprint.width), footprint.streams);
    float texel = FACE_SPAN / (float(1u << hit.level) * float(TILE_INTERVALS));
    float3 east = normalize(cross(float3(0.0, 1.0, 0.0), up));
    if (!all(isfinite(east)))
        east = float3(1.0, 0.0, 0.0);
    float3 north = cross(up, east);
    float3 r = position;
    float h_east = surface_height(TileHit(earth_tiles.height, FacePoint(normalize(r + texel * east)), hit.level, false));
    float h_west = surface_height(TileHit(earth_tiles.height, FacePoint(normalize(r - texel * east)), hit.level, false));
    float h_north = surface_height(TileHit(earth_tiles.height, FacePoint(normalize(r + texel * north)), hit.level, false));
    float h_south = surface_height(TileHit(earth_tiles.height, FacePoint(normalize(r - texel * north)), hit.level, false));
    return normalize(up * (2.0 * texel) - east * (h_east - h_west) - north * (h_north - h_south));
}
