module clouds;

import common.constants;

// From this bounce on, clouds use the thinner extinction with an isotropic phase.
public static const uint CLOUD_DEEP_BOUNCE = 10;
public static const float CLOUD_EXTINCTION_DEEP = 0.02;
public static const float CLOUD_ALBEDO = 0.99;

// Thin cover is floored at this density fraction so it stays visible.
static const float CLOUD_DENSITY_FLOOR = 0.4;

// The open interval of shell heights (r - CLOUD_BASE_RADIUS) / CLOUD_THICKNESS that
// a column of map `coverage` in [0, 1] fills: (0.2 - 0.2 c, 0.2 + 0.8 c), so thick
// clouds grow mostly upwards. It grows with the coverage.
public float2 cloud_column(float coverage)
{
    const float split = 0.2;
    return float2(split - coverage * split, split + coverage * (1.0 - split));
}

// Density at radius `r` under a column of map `coverage`, at most `max_density`.
public float cloud_density(float r, float coverage, float max_density)
{
    // In f32 the column's top can round past the shell's.
    if (r <= CLOUD_BASE_RADIUS || r >= CLOUD_TOP_RADIUS)
        return 0.0;
    float h = (r - CLOUD_BASE_RADIUS) / CLOUD_THICKNESS;
    float2 column = cloud_column(coverage);
    return h > column.x && h < column.y ? cloud_density_bound(coverage, max_density) : 0.0;
}

// The density inside a column of `coverage`, which bounds the density under every
// smaller coverage.
public float cloud_density_bound(float coverage, float max_density)
{
    return coverage > 0.0 ? max(coverage, CLOUD_DENSITY_FLOOR) * max_density : 0.0;
}
