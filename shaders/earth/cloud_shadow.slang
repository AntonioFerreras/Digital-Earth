// The clouds' shadow map: the optical depth of the cloud shell along every sun ray,
// baked per sun direction so that a shadow is one lookup.
//
// Sun rays are parallel lines, labelled by their offset from the planet centre in
// the plane facing the sun: a radius rho and an angle phi around the sun axis. Rows
// are uniform in theta = asin(rho / PLANET_RADIUS), the angle from the subsolar
// point, so neighbouring rows meet the ground the same distance apart at noon and
// at a grazing sunset, where uniform rho would stretch them by 1 / sin(elevation).
// CLOUD_SHADOW_TWILIGHT_ROWS more rows, uniform in rho, take the lines that pass
// above the planet through the shell.
//
// Along a line, depth is the length inside the shell from the sunward end, which
// skips the clear stretch between a twilight line's two crossings. Layer k holds the
// transmittance at depth (k + 1) / CLOUD_SHADOW_SLICES of the line's whole length
// in the shell: filtering transmittance averages the light across a shadow's edge,
// where filtering optical depth would darken it.

module cloud_shadow;

import common.constants;
import earth.cloud_shell;
import earth.maps;
import shared.earth_bindings;

public static const uint CLOUD_SHADOW_WIDTH = 4096;
public static const uint CLOUD_SHADOW_DAY_ROWS = 2048;
public static const uint CLOUD_SHADOW_TWILIGHT_ROWS = 128;
public static const uint CLOUD_SHADOW_HEIGHT = CLOUD_SHADOW_DAY_ROWS + CLOUD_SHADOW_TWILIGHT_ROWS;
public static const uint CLOUD_SHADOW_SLICES = 8;

// sqrt(a^2 - b^2) for a >= b >= 0, without the cancellation of a * a - b * b.
float leg(float a, float b)
{
    return sqrt(max((a - b) * (a + b), 0.0));
}

// One sun ray, the line rho (cos phi e1 + sin phi e2) + z sun.
public struct SunLine
{
    public float rho;
    // Where the line enters the shell, z_top, and leaves its first crossing, z_first.
    public float z_top;
    float z_first;
    // Whether it crosses the shell twice, above the planet, the second time over
    // [-z_top, -z_first].
    bool twice;
    // The line's whole length in the shell.
    public float length;

    public __init(float rho)
    {
        this.rho = rho;
        z_top = leg(CLOUD_TOP_RADIUS, rho);
        // Lines inside the base sphere go below it; below the planet's they stop at the ground.
        z_first = rho < CLOUD_BASE_RADIUS ? leg(CLOUD_BASE_RADIUS, rho) : -z_top;
        twice = rho >= PLANET_RADIUS && rho < CLOUD_BASE_RADIUS;
        length = (z_top - z_first) * (twice ? 2.0 : 1.0);
    }

    // The shell length from the sunward end down to `z`.
    public float depth(float z)
    {
        float first = z_top - clamp(z, z_first, z_top);
        float second = twice ? -z_first - clamp(z, -z_top, -z_first) : 0.0;
        return first + second;
    }

    // The `z` at shell length `depth` from the sunward end, at most `length`.
    public float z(float depth)
    {
        float first = z_top - z_first;
        return depth <= first ? z_top - depth : -z_first - (depth - first);
    }
};

// The planet's frame of the sun's lines.
public struct SunFrame
{
    public float3 sun;
    public float3 e1;
    public float3 e2;

    public __init(float3 sun)
    {
        this.sun = sun;
        float3 reference = abs(sun.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        e1 = normalize(cross(reference, sun));
        e2 = cross(sun, e1);
    }
};

// The row coordinate (texel units) of the lines at `rho` < CLOUD_TOP_RADIUS.
public float cloud_shadow_row(float rho)
{
    if (rho < PLANET_RADIUS)
        return atan2_fast(rho, leg(PLANET_RADIUS, rho)) * (2.0 / PI) * float(CLOUD_SHADOW_DAY_ROWS);
    return float(CLOUD_SHADOW_DAY_ROWS) +
           (rho - PLANET_RADIUS) / (CLOUD_TOP_RADIUS - PLANET_RADIUS) * float(CLOUD_SHADOW_TWILIGHT_ROWS);
}

// The inverse of `cloud_shadow_row`.
public float cloud_shadow_rho(float row)
{
    if (row < float(CLOUD_SHADOW_DAY_ROWS))
        return PLANET_RADIUS * sin(row / float(CLOUD_SHADOW_DAY_ROWS) * (0.5 * PI));
    return PLANET_RADIUS +
           (row - float(CLOUD_SHADOW_DAY_ROWS)) / float(CLOUD_SHADOW_TWILIGHT_ROWS) * (CLOUD_TOP_RADIUS - PLANET_RADIUS);
}

// The clouds' transmittance from `pos` towards the sun of `frame` in `map`, baked
// for that sun, and that of the whole sun line through `pos`. Bilinear across lines
// and linear in depth between slices.
public float2 cloud_shadow(Texture2DArray<float> map, SamplerState sampler, SunFrame frame, float3 pos)
{
    float z = dot(pos, frame.sun);
    float3 offset = pos - z * frame.sun;
    float rho = length(offset);
    if (rho >= CLOUD_TOP_RADIUS)
        return float2(1.0);
    SunLine line = SunLine(rho);
    if (!(line.length > 0.0))
        return float2(1.0);
    // Sunward of the shell, as most air above the clouds is.
    float depth = line.depth(z);
    if (!(depth > 0.0))
        return float2(1.0);
    float phi = atan2_fast(dot(offset, frame.e2), dot(offset, frame.e1));
    float2 uv = float2(phi / (2.0 * PI) + 0.5, cloud_shadow_row(rho) / float(CLOUD_SHADOW_HEIGHT));
    float slice = depth / line.length * float(CLOUD_SHADOW_SLICES) - 1.0;
    float below = floor(slice);
    float before = below >= 0.0 ? map.SampleLevel(sampler, float3(uv, below), 0.0) : 1.0;
    float after = map.SampleLevel(sampler, float3(uv, min(below + 1.0, float(CLOUD_SHADOW_SLICES - 1))), 0.0);
    float whole = map.SampleLevel(sampler, float3(uv, float(CLOUD_SHADOW_SLICES - 1)), 0.0);
    return float2(lerp(before, after, slice - below), whole);
}

// The transmittances of the line at texel `texel` for `frame`'s sun, one per slice,
// from `steps_per_slice` midpoint samples of each slice.
public void bake_cloud_shadow_line(uint2 texel, SunFrame frame, CloudMedium clouds, uint steps_per_slice,
                                   out float transmittance[CLOUD_SHADOW_SLICES])
{
    float phi = ((float(texel.x) + 0.5) / float(CLOUD_SHADOW_WIDTH) - 0.5) * (2.0 * PI);
    float rho = cloud_shadow_rho(float(texel.y) + 0.5);
    float3 offset = rho * (cos(phi) * frame.e1 + sin(phi) * frame.e2);
    SunLine line = SunLine(rho);
    float dt = line.length / float(CLOUD_SHADOW_SLICES * steps_per_slice);
    float optical_depth = 0.0;
    for (uint k = 0; k < CLOUD_SHADOW_SLICES; k++)
    {
        for (uint i = 0; i < steps_per_slice; i++)
        {
            float3 position = offset + line.z((float(k * steps_per_slice + i) + 0.5) * dt) * frame.sun;
            optical_depth += clouds.extinction_at(position, sphere_uv_fast(position)) * dt;
        }
        transmittance[k] = exp(-optical_depth);
    }
}
