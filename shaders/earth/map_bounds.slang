// Upper bounds of the topography and the cloud coverage over a neighbourhood of a
// point, from the bound map of src/assets/MapBounds.zig, so that marches skip
// stretches where the full maps cannot produce a hit.
//
// A query point's bound holds over its cell's 3 x 3 neighbourhood. That contains
// every point within the angle min(cell height, cell width at the equator) cos(lat)
// of the query point, which a straight ray outside the planet sphere cannot leave
// in less than PLANET_RADIUS times that angle.

module map_bounds;

import common.constants;
import shared.earth_bindings;

// Below this reach, near the poles, the bounds give way to the maps' maxima.
public static const float REACH_MIN = 1000.0;
static const float POLAR_REACH = 20000.0;

public struct MapBound
{
    // Bounds in map units: the topography's height fraction and the cloud coverage.
    public float topography;
    public float clouds;
    // Distance in metres from the query point along any ray staying outside the
    // planet sphere over which both bounds hold.
    public float reach;

    // `uv` are the query point's map coordinates (earth/maps.slang `sphere_uv`).
    public __init(float2 uv)
    {
        float2 size = float2(earth_maps.bounds.xy);
        // sin(pi v) is the cosine of the latitude; 0.99 is a margin for the
        // small-angle form of the reach.
        reach = 0.99 * PLANET_RADIUS * min(PI / size.y, 2.0 * PI / size.x) * sin(PI * saturate(uv.y));
        if (reach < REACH_MIN)
        {
            topography = 1.0;
            clouds = 1.0;
            reach = POLAR_REACH;
            return;
        }
        uint2 cell = min(uint2(uv * size), earth_maps.bounds.xy - 1);
        float4 bound = earth_bounds.Load(int3(cell, 0));
        topography = bound.r;
        clouds = bound.g;
    }
};
