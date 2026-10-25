// The cloud shell along a ray: the coverage map's extinction (earth/clouds.slang)
// and the ray's passes through the shell, walked in stretches of one majorant each.
//
// Importing this module declares the globals of shared/earth_bindings.slang.

module cloud_shell;

import common.constants;
import common.geometry;
import earth.clouds;
import earth.map_bounds;
import earth.maps;
import shared.earth_bindings;
import shared.uniforms;

// The scene's clouds, which are grey.
public struct CloudMedium
{
    // SCENE_CLOUDS.
    public bool enabled;
    // Extinction at unit density and the density maximum.
    public float extinction;
    public float density_max;

    public __init(SceneUniforms scene)
    {
        enabled = (scene.config.x & SCENE_CLOUDS) != 0;
        extinction = scene.media.y;
        density_max = scene.media.z;
    }

    public bool active()
    {
        return enabled && extinction * density_max > 0.0;
    }

    // At `pos`, whose map coordinates are `uv`.
    public float extinction_at(float3 pos, float2 uv)
    {
        float coverage = sample_tiled(earth_clouds, earth_maps.clouds, uv);
        return extinction * cloud_density(length(pos), coverage, density_max);
    }

    // The column at map coordinates `uv`, for the lighting approximations: its
    // thickness and the distance up to its top from radius `r`, in metres.
    public float2 column_extent(float r, float2 uv)
    {
        float2 column = CLOUD_BASE_RADIUS +
                        CLOUD_THICKNESS * cloud_column(sample_tiled(earth_clouds, earth_maps.clouds, uv));
        return float2(column.y - column.x, column.y - r);
    }
};

// One pass of a ray through the cloud shell, walked in stretches that each have one
// majorant from the map bound at their start. Map coordinates are exact at the
// stretch ends and linear in between: over a stretch of a few kilometres a straight
// ray strays from that line by about 1e-6 rad, a small fraction of a map texel.
public struct CloudPass
{
    float3 pos;
    float3 dir;
    public float end;
    // The current stretch [start, stop) and its majorant.
    public float start;
    public float stop;
    public float majorant;
    float2 uv_start;
    // Map coordinates per metre along the ray, the u seam unwrapped.
    float2 uv_rate;
    float2 uv_stop;

    // The pass from `t` to `end` > `t`, at its first stretch.
    public __init(float3 pos, float3 dir, float t, float end, CloudMedium clouds)
    {
        this.pos = pos;
        this.dir = dir;
        this.end = end;
        stop = t;
        uv_stop = sphere_uv(normalize(pos + t * dir));
        advance(clouds);
    }

    // Moves to the stretch that starts where the current one stops.
    [mutating]
    public void advance(CloudMedium clouds)
    {
        start = stop;
        uv_start = uv_stop;
        MapBound bound = MapBound(uv_start);
        stop = min(start + bound.reach, end);
        uv_stop = sphere_uv(normalize(pos + stop * dir));
        float2 delta = uv_stop - uv_start;
        delta.x -= round(delta.x);
        uv_rate = delta / (stop - start);
        float2 h = (radius_range(pos, dir, start, stop) - CLOUD_BASE_RADIUS) / CLOUD_THICKNESS;
        float2 column = cloud_column(bound.clouds);
        bool overlaps = max(h.x, column.x) < min(h.y, column.y);
        majorant = overlaps ? clouds.extinction * cloud_density_bound(bound.clouds, clouds.density_max) : 0.0;
    }

    // The cloud extinction at `t` in the current stretch.
    public float extinction(float t, CloudMedium clouds)
    {
        return clouds.extinction_at(pos + t * dir, uv_start + (t - start) * uv_rate);
    }
};

// The ray's two passes through the cloud shell, (start, end) each, cut at the
// surface hit `t_surface` (negative for none) so that terrain inside the shell
// occludes the clouds behind it.
public float4 cloud_segments(float3 pos, float3 dir, float t_surface)
{
    float2 top = ray_sphere(pos, dir, CLOUD_TOP_RADIUS);
    float2 base = ray_sphere(pos, dir, CLOUD_BASE_RADIUS);
    float4 segments = float4(max(top.x, 0.0), top.y, 0.0, 0.0);
    if (base.y > 0.0)
    {
        // The inner sphere splits the pass: before it (if ahead) and after it.
        segments = base.x > 0.0 ? float4(segments.x, base.x, base.y, top.y) : float4(base.y, top.y, 0.0, 0.0);
    }
    float end = t_surface >= 0.0 ? t_surface : top.y;
    return float4(segments.x, min(segments.y, end), segments.z, min(segments.w, end));
}
