// The ray marcher's lookup tables: column densities to the top of the
// atmosphere, and the spectral multiple-scattering radiance of Hillaire 2020, "A
// Scalable and Production Ready Sky and Atmosphere Rendering Technique". Both are
// baked at texel centres.
//
// Column densities: a Texture2D<float4>, xyz the logarithms of the (Rayleigh, Mie,
// ozone) densities integrated along the ray plus COLUMN_DENSITY_FLOOR, read with
// bilinear filtering in f32. The media's extinction separates into these three, so
// one lookup serves every wavelength: the transmittance is exp(-dot(extinction, column)).
//   (u, v) = bruneton.transmittance_uv(mu, r), only rays that reach the top of the
//   atmosphere; lookups report rays that hit the planet as not visible.
// Multiple scattering: a Texture3D<float4> read with linear filtering, one channel
// per member of a spectral/cie.slang wavelength bundle, so one lookup serves a bundle.
//   w:       slice i holds the bundle of u = i / (slices - 1), at texel centres, so
//            w = (0.5 + u (slices - 1)) / slices interpolates between bundles that
//            stratify the CDF from end to end.
//   (u, v):  u = 0.5 + 0.5 sign(mu_s) sqrt|mu_s|, v = sqrt(h / ATMOSPHERE_HEIGHT),
//            for the sun cosine mu_s and the altitude h; the square roots spend
//            texels on the horizon and the lower atmosphere.

module scattering_luts;

import common.constants;
import common.geometry;
import atmosphere.bruneton;
import atmosphere.media;
import atmosphere.phase;

public static const uint2 COLUMN_DENSITY_LUT_SIZE = uint2(256, 256);
// Added before the logarithm, which then stays finite where a column vanishes. Unit-density
// extinction stays below 1e-4 / m and ozone cross sections below 1e-22 m^2, so
// columns this thin change no transmittance by more than 1e-4.
public static const float3 COLUMN_DENSITY_FLOOR = float3(1.0, 1.0, 1e18);
// Four channels of 128 slices sample the CDF about as finely as 1 nm steps.
public static const uint3 MULTIPLE_SCATTERING_LUT_SIZE = uint3(32, 32, 128);

// Midpoint-rule steps of the column-density bake.
static const uint COLUMN_DENSITY_STEP_COUNT = 256;
// Hillaire's sample counts: directions over the sphere, and steps per direction.
static const uint MULTIPLE_SCATTERING_DIRECTION_COUNT = 64;
static const uint MULTIPLE_SCATTERING_STEP_COUNT = 20;
// Lambertian ground albedo in the multiple-scattering bake; the original's value.
static const float MULTIPLE_SCATTERING_GROUND_ALBEDO = 0.04;

public float3 lut_texel_centre(uint3 texel, uint3 size)
{
    return (float3(texel) + 0.5) / float3(size);
}

// The bundle's uniform number at multiple-scattering slice `i`.
public float multiple_scattering_slice_u(uint i)
{
    return float(i) / float(MULTIPLE_SCATTERING_LUT_SIZE.z - 1);
}

// The bake integrand: column densities from radius `r` along the zenith cosine
// `mu` to the top of the atmosphere, ignoring the planet, as the table stores them.
public float3 log_column_density_to_top(SpectralMedium medium, float r, float mu)
{
    float3 position = float3(0.0, r, 0.0);
    float3 dir = float3(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    float dt = distance_to_atmosphere_top(r, mu) / float(COLUMN_DENSITY_STEP_COUNT);
    float3 density = 0.0;
    for (uint i = 0; i < COLUMN_DENSITY_STEP_COUNT; i++)
        density += medium.density(position + (float(i) + 0.5) * dt * dir);
    return log(density * dt + COLUMN_DENSITY_FLOOR);
}

// The column densities from `position` along the unit `dir` to the top of the
// atmosphere.
public struct ColumnDensity
{
    // False when the planet is in the way and the transmittance is 0.
    public bool visible;
    public float3 density;

    // Bilinear between texel centres, clamped at the edges like a linear-clamp
    // sampler, but in f32 where hardware filtering has 8-bit weights and some
    // devices cannot filter f32 at all. Near the horizon a column grows
    // exponentially across texels, so its logarithm is what interpolates well.
    public __init(Texture2D<float4> lut, float3 position, float3 dir)
    {
        float radius = length(position);
        float mu = dot(position, dir) / radius;
        // Below sea level the Bruneton mapping is undefined; the sea-level ray stands in.
        float r = clamp(radius, PLANET_RADIUS, ATMOSPHERE_RADIUS);
        visible = !ray_intersects_ground(r, mu);
        float2 texel = transmittance_uv(mu, r, float2(COLUMN_DENSITY_LUT_SIZE)) * float2(COLUMN_DENSITY_LUT_SIZE) - 0.5;
        float2 below = floor(texel);
        float2 weight = texel - below;
        int2 last = int2(COLUMN_DENSITY_LUT_SIZE) - 1;
        int2 lo = clamp(int2(below), 0, last);
        int2 hi = clamp(int2(below) + 1, 0, last);
        float3 bottom = lerp(lut.Load(int3(lo.x, lo.y, 0)).xyz, lut.Load(int3(hi.x, lo.y, 0)).xyz, weight.x);
        float3 top = lerp(lut.Load(int3(lo.x, hi.y, 0)).xyz, lut.Load(int3(hi.x, hi.y, 0)).xyz, weight.x);
        density = exp(lerp(bottom, top, weight.y)) - COLUMN_DENSITY_FLOOR;
    }

    // Transmittance at the wavelength with unit-density `extinction`.
    public float transmittance(float3 extinction)
    {
        return visible ? exp(-dot(extinction, density)) : 0.0;
    }
};

// (mu_s, h) of the multiple-scattering texel centred at `uv`.
public float2 multiple_scattering_mu_s_h(float2 uv)
{
    float s = 2.0 * uv.x - 1.0;
    return float2(sign(s) * s * s, ATMOSPHERE_HEIGHT * uv.y * uv.y);
}

// Where the multiple-scattering table holds the isotropic radiance of all
// scattering orders above the first, per unit sun irradiance, at `position` for
// the sun direction `sun`, one value per bundle member.
public struct MultipleScatteringLookup
{
    float2 uv;

    public __init(float3 position, float3 sun)
    {
        float r = length(position);
        float mu_s = dot(position, sun) / r;
        float h = clamp(r - PLANET_RADIUS, 0.0, ATMOSPHERE_HEIGHT);
        uv = float2(0.5 + 0.5 * sign(mu_s) * sqrt(abs(mu_s)), sqrt(h / ATMOSPHERE_HEIGHT));
    }

    // `bundle_u` is WavelengthBundle.u.
    public float4 sample(Texture3D<float4> lut, SamplerState linear_clamp, float bundle_u)
    {
        float slice_count = float(MULTIPLE_SCATTERING_LUT_SIZE.z);
        float w = (0.5 + bundle_u * (slice_count - 1.0)) / slice_count;
        return lut.SampleLevel(linear_clamp, float3(uv, w), 0.0);
    }
};

// Hillaire's two sphere averages at one point, both per unit sun irradiance
// and with isotropic phase: `second_order` is the radiance scattered once
// before reaching the point (sun and ground lit), `transfer` (f_ms) the
// fraction of isotropic radiance the surroundings scatter back.
public struct MultipleScattering
{
    public float second_order;
    public float transfer;

    // The geometric series of all orders >= 2: L_2nd / (1 - f_ms).
    public float radiance()
    {
        return second_order / (1.0 - transfer);
    }
};

public MultipleScattering multiple_scattering(Texture2D<float4> column_density_lut, SpectralMedium medium, float mu_s,
                                              float h)
{
    float3 position = float3(0.0, PLANET_RADIUS + h, 0.0);
    float3 sun = float3(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    // Fibonacci sphere: deterministic, evenly spread, equal solid angles.
    const float golden_angle = PI * (3.0 - sqrt(5.0));

    MultipleScattering result = { 0.0, 0.0 };
    for (uint d = 0; d < MULTIPLE_SCATTERING_DIRECTION_COUNT; d++)
    {
        float z = 1.0 - (2.0 * float(d) + 1.0) / float(MULTIPLE_SCATTERING_DIRECTION_COUNT);
        float ring = sqrt(1.0 - z * z);
        float phi = golden_angle * float(d);
        float3 dir = float3(ring * cos(phi), z, ring * sin(phi));

        float ground = ray_sphere(position, dir, PLANET_RADIUS).x;
        bool hits_ground = ground > 0.0;
        float t_end = hits_ground ? ground : ray_sphere(position, dir, ATMOSPHERE_RADIUS).y;
        float dt = t_end / float(MULTIPLE_SCATTERING_STEP_COUNT);

        float transmittance = 1.0;
        for (uint i = 0; i < MULTIPLE_SCATTERING_STEP_COUNT; i++)
        {
            float3 sample = position + (float(i) + 0.5) * dt * dir;
            float3 density = medium.density(sample);
            float step_depth = dot(medium.extinction, density) * dt;
            // Integral of the transmittance over the step, with the density held constant.
            float visible = transmittance * step_integral(step_depth) * dt;
            float scattering = dot(medium.scattering, density.xy);
            float sun_transmittance =
                ColumnDensity(column_density_lut, sample, sun).transmittance(medium.extinction);
            result.second_order += visible * scattering * ISOTROPIC_PHASE * sun_transmittance;
            result.transfer += visible * scattering;
            transmittance *= exp(-step_depth);
        }
        if (hits_ground)
        {
            float3 ground_position = position + t_end * dir;
            float cos_sun = saturate(dot(normalize(ground_position), sun));
            float sun_transmittance =
                ColumnDensity(column_density_lut, ground_position, sun).transmittance(medium.extinction);
            result.second_order +=
                transmittance * MULTIPLE_SCATTERING_GROUND_ALBEDO / PI * cos_sun * sun_transmittance;
        }
    }
    // Both integrals are sphere averages against the isotropic phase, 1 / 4 pi.
    result.second_order /= float(MULTIPLE_SCATTERING_DIRECTION_COUNT);
    result.transfer /= float(MULTIPLE_SCATTERING_DIRECTION_COUNT);
    return result;
}
