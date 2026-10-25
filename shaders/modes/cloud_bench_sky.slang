// The cloud bench's light for the sun of `frame`, taken at the cloud's centre: the
// sky's radiance by direction, marched through the atmosphere by the ray marcher's
// segment march (earth/segment_march.slang) as Hillaire's sky-view table, and the
// sun's irradiance through the atmosphere. Below the horizon the march ends at a
// black planet. Both are linear Rec.709, averaged over wavelength bundles that
// stratify the CIE sampling.

import atmosphere.media;
import atmosphere.scattering_luts;
import atmosphere.sunlight;
import clouds.bench_volume;
import common.constants;
import common.geometry;
import common.spherical_harmonics;
import earth.cloud_shadow;
import earth.cloud_shell;
import earth.segment_march;
import shared.bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;

[format("rgba16f")]
RWTexture2D<float4> bench_sky_target;
// Texel (0, 0): the sun's irradiance; texels (1 + j, 0): the sky's radiance projected
// onto spherical harmonic j, for the approximation's sky light.
[format("rgba32f")]
RWTexture2D<float4> bench_sun_target;

static const uint SKY_BUNDLE_COUNT = 16;
static const uint SKY_PROJECTION_DIRECTION_COUNT = 4096;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void bake_sky(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= BENCH_SKY_SIZE))
        return;
    float3 dir = sky_direction((float2(thread.xy) + 0.5) / float2(BENCH_SKY_SIZE));
    float3 pos = BenchVolume().world_centre();
    float3 sun = frame.sun_direction.xyz;
    CloudMedium clouds = CloudMedium(scene);
    clouds.enabled = false;
    SunFrame sun_frame = SunFrame(sun);
    float2 atmosphere = ray_sphere(pos, dir, ATMOSPHERE_RADIUS);
    float2 planet = ray_sphere(pos, dir, PLANET_RADIUS);
    float t_end = planet.x > 0.0 ? planet.x : atmosphere.y;
    float3 rgb = 0.0;
    for (uint k = 0; k < SKY_BUNDLE_COUNT; k++)
    {
        WavelengthBundle w = sample_wavelength_bundle(cie, (float(k) + 0.5) / float(SKY_BUNDLE_COUNT));
        SpectralMediumBundle media = SpectralMediumBundle(ozone_cross_section, ozone_profile, w.wavelength, scene.media);
        // The sky is the light source, so it keeps its own multiple scattering under
        // SCENE_SINGLE_SCATTERING.
        SegmentMarch march = SegmentMarch(media, clouds, sun_frame, w.u, pos, dir, sun, float3(0.5), false, true, true);
        march.march(max(atmosphere.x, 0.0), t_end);
        Spectrum radiance = march.segment.in_scattering;
        for (int j = 0; j < BUNDLE_WAVELENGTH_COUNT; j++)
            radiance[j] *= solar_irradiance(w.wavelength[j]);
        rgb += xyz_to_rec709(mul(w.xyz_weight, radiance));
    }
    bench_sky_target[thread.xy] = float4(rgb / float(SKY_BUNDLE_COUNT), 0.0);
}

// One thread, after `bake_sky`.
[shader("compute")]
[numthreads(1, 1, 1)]
void bake_sun(uint3 thread: SV_DispatchThreadID)
{
    bench_sun_target[uint2(0, 0)] = float4(sun_irradiance(BenchVolume().world_centre(), frame.sun_direction.xyz), 0.0);

    // A Fibonacci sphere of directions, each reading its nearest texel.
    float3 sky[SH_COUNT];
    for (uint j = 0; j < SH_COUNT; j++)
        sky[j] = 0.0;
    for (uint i = 0; i < SKY_PROJECTION_DIRECTION_COUNT; i++)
    {
        float z = 1.0 - (2.0 * float(i) + 1.0) / float(SKY_PROJECTION_DIRECTION_COUNT);
        float phi = float(i) * PI * (3.0 - sqrt(5.0));
        float ring = sqrt(1.0 - z * z);
        float3 dir = float3(ring * cos(phi), z, ring * sin(phi));
        uint2 texel = min(uint2(sky_uv(dir) * float2(BENCH_SKY_SIZE)), BENCH_SKY_SIZE - 1);
        float3 radiance = bench_sky_target[texel].rgb;
        float basis[SH_COUNT] = sh_basis(dir);
        for (uint j = 0; j < SH_COUNT; j++)
            sky[j] += radiance * basis[j];
    }
    for (uint j = 0; j < SH_COUNT; j++)
        bench_sun_target[uint2(1 + j, 0)] = float4(sky[j] * (4.0 * PI / float(SKY_PROJECTION_DIRECTION_COUNT)), 0.0);
}
