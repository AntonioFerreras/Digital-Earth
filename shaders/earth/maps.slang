module maps;

import common.constants;

// Equirectangular coordinates of the unit `n`: +Y is north, u = 0.5 is the
// Greenwich meridian on -X, u = 0.75 is 90 E on +Z, and the seam is +X. v = 0
// is the south pole, matching the south-first rows of the map cache.
public float2 sphere_uv(float3 n)
{
    return float2((atan2(n.z, -n.x) / PI + 1.0) / 2.0, asin(n.y) / PI + 0.5);
}
