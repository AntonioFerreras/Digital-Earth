// The ray marcher: up to BOUNCE_COUNT_MAX segments, each marched through the
// atmosphere with single scattering from the sun and the multiple-scattering
// LUT, joined by BRDF-sampled bounces off the surface. A deterministic
// approximation of the path tracer that converges in a few samples. Each sample
// carries a bundle of wavelengths (spectral/cie.slang) along one geometric path.

import atmosphere.media;
import atmosphere.phase;
import atmosphere.scattering_luts;
import common.constants;
import common.geometry;
import common.sampling;
import earth.brdf;
import earth.surface;
import shared.bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

Texture2D<float4> column_density_lut;
// One channel per bundle member.
Texture3D<float4> multiple_scattering_lut;

static const uint BOUNCE_COUNT_MAX = 3;
// Steps per segment grow linearly with its length up to SEGMENT_FULL_LENGTH, as in
// IntegrateScatteredLuminance of Hillaire 2020 (sebh/UnrealEngineSkyAtmosphere);
// optical-depth placement lets a long segment take far fewer than uniform steps would.
static const uint SEGMENT_STEP_COUNT_MIN = 12;
static const uint SEGMENT_STEP_COUNT_MAX = 32;
static const float SEGMENT_FULL_LENGTH = 1000e3;
// Equal cells of the optical-depth estimate that places the steps.
static const uint DEPTH_CELL_COUNT = 16;
// Share of the steps spread uniformly over the segment, so that thin air still gets
// steps where it carries the light (twilight above a shadowed ground).
static const float UNIFORM_STEP_SHARE = 0.25;

// In-scattered radiance per unit sun irradiance and transmittance of one segment.
struct Segment
{
    Spectrum in_scattering;
    Spectrum transmittance;
};

// Step boundaries uniform in the cumulative share of each cell's optical depth (at its
// centre), mixed with UNIFORM_STEP_SHARE. Steps crowd into dense air wherever it lies:
// Hillaire's t^2 spacing serves the near end only, not a limb ray's tangent point or
// the ground end of a ray from orbit.
struct StepPlacement
{
    float cell_share[DEPTH_CELL_COUNT];
    float t_start;
    float cell_length;
    // The cell of the last `t` and the share of the cells before it.
    uint cell;
    float share_before;

    __init(SpectralMediumBundle media, float3 origin, float3 dir, float t_start, float t_end)
    {
        this.t_start = t_start;
        cell_length = (t_end - t_start) / float(DEPTH_CELL_COUNT);
        cell = 0;
        share_before = 0.0;
        float depth = 0.0;
        for (uint i = 0; i < DEPTH_CELL_COUNT; i++)
        {
            float3 position = origin + (t_start + (float(i) + 0.5) * cell_length) * dir;
            Spectrum extinction = mul(media.extinction, media.first.density(position));
            cell_share[i] = dot(extinction, Spectrum(1.0)) * cell_length;
            depth += cell_share[i];
        }
        float uniform_share = 1.0 / float(DEPTH_CELL_COUNT);
        for (uint i = 0; i < DEPTH_CELL_COUNT; i++)
            cell_share[i] = depth > 0.0 ? lerp(cell_share[i] / depth, uniform_share, UNIFORM_STEP_SHARE) : uniform_share;
    }

    // The distance along the ray at cumulative share `u`, which must not decrease
    // from one call to the next.
    [mutating]
    float t(float u)
    {
        while (cell + 1 < DEPTH_CELL_COUNT && share_before + cell_share[cell] <= u)
        {
            share_before += cell_share[cell];
            cell++;
        }
        return t_start + (float(cell) + saturate((u - share_before) / cell_share[cell])) * cell_length;
    }
};

// Samples sit at a jittered offset within each step (the original took the
// step's start), so accumulation averages the step quadrature away.
Segment march_segment(SpectralMediumBundle media, float bundle_u, float3 origin, float3 dir, float t_start, float t_end,
                      float3 sun, float jitter, bool multiple_scattering)
{
    float cos_theta = dot(dir, sun);
    float2 phase = float2(rayleigh_phase(cos_theta), klein_nishina_phase(cos_theta, MIE_KLEIN_NISHINA_E));
    float length_fraction = saturate((t_end - t_start) / SEGMENT_FULL_LENGTH);
    uint step_count = uint(round(lerp(float(SEGMENT_STEP_COUNT_MIN), float(SEGMENT_STEP_COUNT_MAX), length_fraction)));
    StepPlacement placement = StepPlacement(media, origin, dir, t_start, t_end);

    Segment segment = { Spectrum(0.0), Spectrum(1.0) };
    float step_start = t_start;
    for (uint i = 0; i < step_count; i++)
    {
        // The last step ends exactly at the segment's end, whatever the rounding.
        float step_end = i + 1 < step_count ? placement.t(float(i + 1) / float(step_count)) : t_end;
        float dt = step_end - step_start;
        float3 position = origin + lerp(step_start, step_end, jitter) * dir;
        step_start = step_end;
        float3 density = media.first.density(position);
        Spectrum step_depth = mul(media.extinction, density) * dt;
        ColumnDensity sun_column = ColumnDensity(column_density_lut, position, sun);
        Spectrum in_scattering = mul(media.scattering, density.xy * phase);
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
            in_scattering[k] *= sun_column.transmittance(media.extinction[k]);
        if (multiple_scattering)
        {
            in_scattering += mul(media.scattering, density.xy) *
                             MultipleScatteringLookup(position, sun).sample(multiple_scattering_lut, linear_clamp,
                                                                            bundle_u);
        }
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        {
            float visible = segment.transmittance[k] * step_integral(step_depth[k]) * dt;
            segment.in_scattering[k] += in_scattering[k] * visible;
        }
        segment.transmittance *= exp(-step_depth);
    }
    return segment;
}

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void raymarch_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 dir = s.camera_ray_direction();
    WavelengthBundle w = sample_wavelength_bundle(cie, s.rng.next4().x);

    SpectralMediumBundle media = SpectralMediumBundle(ozone_cross_section, ozone_profile, w.wavelength, scene.media);
    float cos_sun_radius = cos(SUN_ANGULAR_RADIUS);
    Spectrum sun_irradiance;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        sun_irradiance[k] = solar_irradiance(w.wavelength[k]);

    float3 origin = frame.camera_position.xyz;
    Spectrum radiance = 0.0;
    Spectrum throughput = 1.0;
    bool multiple_scattering = (scene.config.x & SCENE_SINGLE_SCATTERING) == 0;
    for (uint bounce = 0; bounce < min(BOUNCE_COUNT_MAX, scene.config.y); bounce++)
    {
        float2 atmosphere = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
        if (atmosphere.y < 0.0)
            break;
        float t_surface = intersect_surface(origin, dir, scene);
        float t_end = t_surface == SURFACE_MISS ? atmosphere.y : t_surface;

        float4 u = s.rng.next4();
        float3 sun = sample_cone(u.xy, cos_sun_radius, frame.sun_direction.xyz);
        Segment segment = march_segment(media, w.u, origin, dir, max(atmosphere.x, 0.0), t_end, sun, u.z,
                                        multiple_scattering);
        radiance += throughput * segment.in_scattering * sun_irradiance;
        throughput *= segment.transmittance;
        if (t_surface == SURFACE_MISS)
            break;

        float3 position = origin + t_surface * dir;
        SurfaceSample surface = sample_surface(position, scene);
        float3 lifted = offset_surface_point(position, scene);
        bool sun_visible = surface_visible(lifted, sun, scene);
        ColumnDensity sun_column = ColumnDensity(column_density_lut, lifted, sun);
        BounceSample bounce = sample_bounce(surface, -dir, s.rng.next4().xyz, scene);
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        {
            float albedo = srgb_to_spectrum(srgb_to_spectrum_basis, surface.albedo_srgb, w.wavelength[k]);
            radiance[k] += throughput[k] * surface_emission(surface, w.wavelength[k], scene);
            float sun_visibility = sun_visible ? sun_column.transmittance(media.extinction[k]) : 0.0;
            SurfaceResponse direct = surface_brdf(surface, albedo, -dir, sun, scene);
            radiance[k] += throughput[k] * sun_visibility * sun_irradiance[k] * direct.brdf * direct.n_dot_l;
            SurfaceResponse bounced = surface_brdf(surface, albedo, -dir, bounce.dir, scene);
            throughput[k] *= bounced.brdf * bounced.n_dot_l / bounce.pdf;
        }
        if (!(bounce.pdf > 0.0))
            break;
        origin = lifted;
        dir = bounce.dir;
    }

    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
    {
        if (!(radiance[k] >= 0.0) || isinf(radiance[k]))
            radiance[k] = 0.0;
    }
    s.accumulate(xyz_to_rec709(mul(w.xyz_weight, radiance)));
}
