module clouds;

import common.constants;

// From this bounce on, clouds use the thinner extinction with an isotropic phase.
public static const uint CLOUD_DEEP_BOUNCE = 10;
public static const float CLOUD_EXTINCTION_DEEP = 0.02;
public static const float CLOUD_ALBEDO = 0.99;

// Density at radius `r` under a column of map `coverage` in [0, 1], at most
// `max_density` (the delta-tracking majorant). Covered columns span shell
// heights (0.2 - 0.2 c, 0.2 + 0.8 c), so thick clouds grow mostly upwards; thin
// cover is floored at 0.4 so it stays visible.
public float cloud_density(float r, float coverage, float max_density)
{
    if (r <= CLOUD_BASE_RADIUS || r >= CLOUD_TOP_RADIUS)
        return 0.0;
    float h = (r - CLOUD_BASE_RADIUS) / CLOUD_THICKNESS;
    const float split = 0.2;
    bool inside = h - split < coverage * (1.0 - split) && split - h < coverage * split;
    return inside ? max(coverage, 0.4) * max_density : 0.0;
}
