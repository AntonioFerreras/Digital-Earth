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
// The coarsest level a jump tries, a few times the tallest relief across.
static const uint JUMP_LEVEL_MIN = 4;
// A march stops once the field is within this fraction of the distance marched inside
// the atmosphere.
static const float RELATIVE_EPSILON = 1e-4;
// The shortest field step, in texels of the resident tile.
static const float STEP_MIN_TEXELS = 0.25;
// Secant steps that pin down a crossing once two field samples bracket it.
static const uint SECANT_STEPS = 3;
public static const float TILE_SURFACE_MISS = -1.0;

// The surface height at `hit`: the terrain, or the sea where it lies below.
float surface_height(TileHit hit)
{
    return max(tile_height_at(hit), 0.0);
}

// Where the ray `origin + t dir` crosses, no earlier than `t`, the plane through the
// planet centre at face coordinate `low` or `high` along `axis` that it moves toward
// from `t`; FLOAT_MAX if it never reaches it. A point that rounding put just past that
// plane has crossed it at `t`.
float axis_exit(float3 origin, float3 dir, float t, float3 axis, float3 normal, float low, float high)
{
    float3 position = origin + t * dir;
    float dir_axis = dot(dir, axis);
    float dir_normal = dot(dir, normal);
    // The sign of the rate of the gnomonic coordinate dot(x, axis) / dot(x, normal).
    float side = dir_axis * dot(position, normal) - dot(position, axis) * dir_normal > 0.0 ? 1.0 : -1.0;
    float gnomonic = tan(0.5 * PI * ((side > 0.0 ? high : low) - 0.5));
    // The plane's normal, axis - gnomonic normal, turned to point out of the tile.
    float rate = side * (dir_axis - gnomonic * dir_normal);
    if (rate <= 0.0)
        return FLOAT_MAX;
    return max(-side * (dot(origin, axis) - gnomonic * dot(origin, normal)) / rate, t);
}

// Where the ray `origin + t dir` leaves tile `tile` of `level` on `face`, no earlier
// than `t`, or FLOAT_MAX: through the edge it moves toward along each axis.
float tile_exit(float3 origin, float3 dir, uint face, uint level, uint2 tile, float t)
{
    FaceFrame frame = FaceFrame(face);
    float size = 1.0 / float(1u << level);
    float2 low = float2(tile) * size;
    float2 high = low + size;
    return min(axis_exit(origin, dir, t, frame.s_axis, frame.normal, low.x, high.x),
               axis_exit(origin, dir, t, frame.t_axis, frame.normal, low.y, high.y));
}

// The radial gap from the surface to `t` along the ray, through a lookup asking for
// `level`.
float surface_gap(float3 origin, float3 dir, float t, uint level)
{
    float3 position = origin + t * dir;
    TileHit hit = TileHit(earth_tiles.height, FacePoint(normalize(position)), level, false);
    return length(position) - PLANET_RADIUS - surface_height(hit);
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
    // The last point known above the surface, which brackets a crossing with the first
    // found below it; negative while the ray has none.
    float t_above = -1.0;
    float gap_above = 0.0;
    // Like a maximum-mipmap traversal over the tiles and, below the resident tile, its
    // blocks: the level a jump tries coarsens after each jump and refines while the ray
    // dips below the area's maximum, down to the finest block, where the field steps.
    uint jump_level = JUMP_LEVEL_MIN;
    uint field_steps = 0;
    for (uint iteration = 0; iteration < ITERATIONS_MAX && field_steps < FIELD_STEPS_MAX; iteration++)
    {
        float3 position = origin + t * dir;
        FacePoint p = FacePoint(normalize(position));
        TileHit hit = TileHit(earth_tiles.height, p, tile_level(cone.width_at(t)), cone.streams);
        uint finest = hit.level + BLOCK_LEVELS;
        uint level = min(jump_level, finest);
        float exit;
        float top;
        bool clear;
        for (;; level++)
        {
            uint side = 1u << level;
            uint2 area = min(uint2(p.uv * float(side)), side - 1);
            exit = min(tile_exit(origin, dir, p.face, level, area, t), end);
            top = PLANET_RADIUS + max(float(hit.area_max(level, area)), 0.0);
            // An exit at `t` is rounding of a crossing nearly parallel to the edge,
            // which the field steps past rather than jumps of a metre.
            clear = exit > t && radius_range(origin, dir, t, exit).x > top;
            if (clear || level == finest)
                break;
        }
        if (clear)
        {
            t = exit;
            jump_level = max(level, JUMP_LEVEL_MIN + 1) - 1;
            // Above the area's top, so above the surface by at least the difference.
            t_above = t;
            gap_above = max(length(origin + t * dir) - top, 0.0);
            if (t >= end)
                break;
            continue;
        }
        // Steps of the gap above the surface, but of at least a fraction of a texel, so
        // that grazing rays cross a block in a bounded count.
        float step_min = STEP_MIN_TEXELS * FACE_SPAN / (float(1u << hit.level) * float(TILE_INTERVALS));
        float gap = length(position) - PLANET_RADIUS - surface_height(hit);
        while (true)
        {
            if (gap >= 0.0)
            {
                if (gap < (t - atmosphere_entry) * RELATIVE_EPSILON)
                    return t;
                t_above = t;
                gap_above = gap;
            }
            else if (t_above >= 0.0)
            {
                // Secant steps within the bracket.
                for (uint k = 0; k < SECANT_STEPS; k++)
                {
                    float t_mid = t_above + (t - t_above) * gap_above / (gap_above - gap);
                    float gap_mid = surface_gap(origin, dir, t_mid, hit.level);
                    if (gap_mid < 0.0)
                    {
                        t = t_mid;
                        gap = gap_mid;
                    }
                    else
                    {
                        t_above = t_mid;
                        gap_above = gap_mid;
                    }
                }
                return t_above + (t - t_above) * gap_above / (gap_above - gap);
            }
            else
            {
                // Under the surface with no point above it: the ray starts inside.
                return t;
            }
            if (t >= end)
                return ground_hit ? ground.x : TILE_SURFACE_MISS;
            if (field_steps >= FIELD_STEPS_MAX)
                break;
            // The last step ends at `end`, so the stretch before it is sampled too.
            t = min(t + max(gap, step_min), end);
            field_steps += 1;
            // Past the block, the next one is looked up afresh.
            if (t >= exit)
                break;
            position = origin + t * dir;
            hit.move_to(FacePoint(normalize(position)));
            gap = length(position) - PLANET_RADIUS - surface_height(hit);
        }
        jump_level = finest - 1;
    }
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
