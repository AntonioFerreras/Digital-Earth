// One segment of the ray marcher (modes/raymarch.slang) marched through the atmosphere
// and the cloud shell: single scattering from the sun, the atmosphere's
// multiple-scattering LUT, and the clouds' multiple-scattering model.
//
// Importing this module declares its two tables next to the globals of
// shared/bindings.slang and shared/earth_bindings.slang.

module segment_march;

import atmosphere.media;
import atmosphere.phase;
import atmosphere.scattering_luts;
import clouds.multiple_scattering;
import common.constants;
import common.geometry;
import earth.cloud_shadow;
import earth.cloud_shell;
import earth.clouds;
import earth.maps;
import earth.surface;
import shared.bindings;
import shared.uniforms;
import spectral.cie;

// One channel per bundle member.
public Texture3D<float4> multiple_scattering_lut;
// earth/cloud_shadow.slang, baked by modes/cloud_shadow_bake.slang for the sun of `frame`.
public Texture2DArray<float> cloud_shadow_map;

// Steps per segment grow linearly with its length up to SEGMENT_FULL_LENGTH, as in
// IntegrateScatteredLuminance of Hillaire 2020 (sebh/UnrealEngineSkyAtmosphere);
// optical-depth placement lets a long segment take far fewer than uniform steps would.
static const uint SEGMENT_STEP_COUNT_MIN = 12;
static const uint SEGMENT_STEP_COUNT_MAX = 32;
static const float SEGMENT_FULL_LENGTH = 1000e3;
// Cloud passes take a step per CLOUD_STEP_DEPTH of each stretch's majorant optical
// depth plus one per CLOUD_AIR_STEP_LENGTH, so that the air in the shell keeps steps
// where no cloud is.
static const float CLOUD_STEP_DEPTH = 1.0;
static const float CLOUD_AIR_STEP_LENGTH = 20e3;
static const uint CLOUD_STRETCH_STEP_COUNT_MAX = 32;
static const uint CLOUD_SHADOW_STEP_COUNT = 8;
static const float CLOUD_SHADOW_STEP_RATIO = 1.7;
// Cloud farther along a shadow ray than this is left out.
static const float CLOUD_SHADOW_REACH = 150e3;
// Shafts of light between clouds are finer than the shadow map's lines, so outside
// clouds the shadow is marched within CLOUD_SHADOW_MARCH_NEAR of the camera and looked
// up beyond CLOUD_SHADOW_MARCH_FAR, where they shrink below a pixel. In between, each
// sample picks one, the map with a chance growing linearly with distance.
static const float CLOUD_SHADOW_MARCH_NEAR = 50e3;
static const float CLOUD_SHADOW_MARCH_FAR = 100e3;
// Shadow optical depth past which both the sun's beam and the multiply scattered
// light are negligible.
static const float CLOUD_SHADOW_DEPTH_MAX = 100.0;
// The clouds' multiply scattered light is clouds/multiple_scattering.slang's; sky
// light reaches the part of a column near its top, dimmed by the cloud above.

// Equal cells of the optical-depth estimate that places the steps.
static const uint DEPTH_CELL_COUNT = 16;
// Share of the steps spread uniformly over the segment, so that thin air still gets
// steps where it carries the light (twilight above a shadowed ground).
static const float UNIFORM_STEP_SHARE = 0.25;

// In-scattered radiance per unit sun irradiance and transmittance of one segment.
public struct Segment
{
    public Spectrum in_scattering;
    public Spectrum transmittance;
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

// The clouds' optical depth from `pos` towards the unit `dir`, at most about
// CLOUD_SHADOW_DEPTH_MAX: CLOUD_SHADOW_STEP_COUNT steps over the shell's first pass,
// growing geometrically from its start up to CLOUD_SHADOW_REACH, so that the near
// cloud, which shades most, gets the short steps.
float cloud_optical_depth(float3 pos, float3 dir, CloudMedium clouds, float jitter)
{
    float2 pass = cloud_segments(pos, dir, SURFACE_MISS).xy;
    float length = min(pass.y - pass.x, CLOUD_SHADOW_REACH);
    if (!(length > 0.0))
        return 0.0;
    // Step i spans first * ratio^i.
    float first = length * (CLOUD_SHADOW_STEP_RATIO - 1.0) / (pow(CLOUD_SHADOW_STEP_RATIO, float(CLOUD_SHADOW_STEP_COUNT)) - 1.0);
    float t = pass.x;
    float dt = first;
    float depth = 0.0;
    for (uint i = 0; i < CLOUD_SHADOW_STEP_COUNT && depth < CLOUD_SHADOW_DEPTH_MAX; i++)
    {
        float3 position = pos + (t + jitter * dt) * dir;
        depth += clouds.extinction_at(position, sphere_uv_fast(position)) * dt;
        t += dt;
        dt *= CLOUD_SHADOW_STEP_RATIO;
    }
    return depth;
}

// The clouds' transmittance from `pos`, outside clouds, towards the sun `dir` near
// that of `sun_frame`; `u.x` jitters the march and `u.y` picks the method.
public float clear_air_cloud_transmittance(float3 pos, float3 dir, SunFrame sun_frame, CloudMedium clouds, float2 u)
{
    float camera_distance = distance(pos, frame.camera_position.xyz);
    bool marched = (scene.config.x & SCENE_MARCH_CLOUD_SHADOWS) != 0;
    if (marched || camera_distance < lerp(CLOUD_SHADOW_MARCH_NEAR, CLOUD_SHADOW_MARCH_FAR, u.y))
        return exp(-cloud_optical_depth(pos, dir, clouds, u.x));
    return cloud_shadow(cloud_shadow_map, linear_repeat_u, sun_frame, pos).x;
}

// One segment's march, step by step in order along the ray. Each step takes its
// media at a jittered point within it (the original took the step's start), so
// accumulation averages the step quadrature away, and integrates them with the
// step's own extinction (`step_integral`).
public struct SegmentMarch
{
    SpectralMediumBundle media;
    CloudMedium clouds;
    // The frame of `cloud_shadow_map`'s sun.
    SunFrame sun_frame;
    float bundle_u;
    float3 origin;
    float3 dir;
    float3 sun;
    float2 air_phase;
    float droplet_phase;
    float cos_sun;
    float jitter;
    // Jitters the shadow march and picks the clear-air method.
    float2 shadow_u;
    bool multiple_scattering;
    // Whether sky light reaches the clouds.
    bool sky_light;
    public Segment segment;

    // `reduce_peak` as in the path tracer, for segments after a bounce.
    public __init(SpectralMediumBundle media, CloudMedium clouds, SunFrame sun_frame, float bundle_u, float3 origin,
           float3 dir, float3 sun, float3 jitter, bool reduce_peak, bool multiple_scattering, bool sky_light)
    {
        this.media = media;
        this.clouds = clouds;
        this.sun_frame = sun_frame;
        this.bundle_u = bundle_u;
        this.origin = origin;
        this.dir = dir;
        this.sun = sun;
        float cos_theta = dot(dir, sun);
        air_phase = float2(rayleigh_phase(cos_theta), klein_nishina_phase(cos_theta, MIE_KLEIN_NISHINA_E));
        droplet_phase = cloud_phase(cos_theta, reduce_peak);
        cos_sun = cos_theta;
        this.jitter = jitter.x;
        shadow_u = jitter.yz;
        this.multiple_scattering = multiple_scattering;
        this.sky_light = sky_light;
        segment = { Spectrum(0.0), Spectrum(1.0) };
    }

    bool opaque()
    {
        return !any(segment.transmittance >= 1e-4);
    }

    // The step of length `dt` sampled at `t`, where the cloud extinction is `cloud_extinction`.
    [mutating]
    void step(float t, float dt, float cloud_extinction)
    {
        float3 position = origin + t * dir;
        float3 density = media.first.density(position);
        Spectrum step_depth = (mul(media.extinction, density) + cloud_extinction) * dt;
        ColumnDensity sun_column = ColumnDensity(column_density_lut, position, sun);
        // The clouds' transmittance towards the sun, and the droplets' in-scattering of
        // the sun's light and the share of the sky's they receive. A cloud's own
        // shading needs the sharp near shadow that the map's slices blur.
        float sun_cloud = 1.0;
        float droplet_light = 0.0;
        float sky_share = 0.0;
        if (cloud_extinction > 0.0)
        {
            float depth = sun_column.visible ? cloud_optical_depth(position, sun, clouds, shadow_u.x) : 0.0;
            sun_cloud = exp(-depth);
            droplet_light = droplet_phase * sun_cloud;
            if (multiple_scattering)
            {
                float2 column_m = clouds.column_extent(length(position), sphere_uv_fast(position));
                // A point lies as deep inside the shell as the nearer of its column's top
                // and base.
                float inside_depth = min(column_m.y, column_m.x - column_m.y) * cloud_extinction;
                droplet_light += cloud_multiple_scattering(depth, cos_sun, inside_depth, scene);
                sky_share = sky_light ? exp(-column_m.y * cloud_extinction) : 0.0;
            }
        }
        else if (clouds.active() && sun_column.visible)
        {
            sun_cloud = clear_air_cloud_transmittance(position, sun, sun_frame, clouds, shadow_u);
        }
        Spectrum in_scattering = mul(media.scattering, density.xy * air_phase) * sun_cloud +
                                 cloud_extinction * CLOUD_ALBEDO * droplet_light;
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
            in_scattering[k] *= sun_column.transmittance(media.extinction[k]);
        if (multiple_scattering)
        {
            // The atmosphere's multiply scattered light stands in for the sky's.
            in_scattering += (mul(media.scattering, density.xy) + cloud_extinction * CLOUD_ALBEDO * sky_share) *
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

    // Air of [start, end), which lies outside the cloud shell's passes.
    [mutating]
    void air_step(float start, float end)
    {
        step(lerp(start, end, jitter), end - start, 0.0);
    }

    // A pass through the cloud shell from `start` to `end`.
    [mutating]
    void march_cloud_pass(float start, float end)
    {
        CloudPass pass = CloudPass(origin, dir, start, end, clouds);
        while (true)
        {
            float length = pass.stop - pass.start;
            float steps = clamp(ceil(length * pass.majorant / CLOUD_STEP_DEPTH + length / CLOUD_AIR_STEP_LENGTH), 1.0,
                                float(CLOUD_STRETCH_STEP_COUNT_MAX));
            float dt = length / steps;
            for (float i = 0.0; i < steps; i++)
            {
                float t = pass.start + (i + jitter) * dt;
                step(t, dt, pass.majorant > 0.0 ? pass.extinction(t, clouds) : 0.0);
            }
            if (pass.stop >= pass.end || opaque())
                break;
            pass.advance(clouds);
        }
    }

    // [t_start, t_end) in order. The air steps are placed over the whole segment, and
    // a pass through the cloud shell takes the place of the air steps it overlaps.
    [mutating]
    public void march(float t_start, float t_end)
    {
        float4 passes = clouds.active() ? cloud_segments(origin, dir, t_end) : float4(0.0);
        float length_fraction = saturate((t_end - t_start) / SEGMENT_FULL_LENGTH);
        uint step_count = uint(round(lerp(float(SEGMENT_STEP_COUNT_MIN), float(SEGMENT_STEP_COUNT_MAX), length_fraction)));
        StepPlacement placement = StepPlacement(media, origin, dir, t_start, t_end);
        // The end of what is marched so far, and the next pass.
        float t = t_start;
        uint next = 0;
        for (uint i = 0; i < step_count; i++)
        {
            // The last step ends exactly at the segment's end, whatever the rounding.
            float step_end = i + 1 < step_count ? placement.t(float(i + 1) / float(step_count)) : t_end;
            while (next < 2)
            {
                float2 pass = next == 0 ? passes.xy : passes.zw;
                if (pass.x < pass.y && pass.x >= step_end)
                    break;
                next++;
                if (!(pass.x < pass.y))
                    continue;
                if (t < pass.x)
                    air_step(t, pass.x);
                march_cloud_pass(pass.x, pass.y);
                t = pass.y;
                if (opaque())
                    return;
            }
            if (t < step_end)
                air_step(t, step_end);
            t = max(t, step_end);
        }
    }
};
