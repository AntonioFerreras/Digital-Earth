// The ray marcher: up to BOUNCE_COUNT_MAX segments, each marched through the
// atmosphere and the cloud shell with single scattering from the sun, the
// atmosphere's multiple-scattering LUT, and the clouds' multiple-scattering model,
// joined by BRDF-sampled bounces off the surface (earth/segment_march.slang). A
// deterministic approximation of the path tracer that converges in a few samples.
// Each sample carries a bundle of wavelengths (spectral/cie.slang) along one geometric
// path.

import atmosphere.media;
import atmosphere.scattering_luts;
import common.constants;
import common.geometry;
import common.ray_cone;
import common.sampling;
import earth.brdf;
import earth.cloud_shadow;
import earth.cloud_shell;
import earth.segment_march;
import earth.surface;
import shared.bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

static const uint BOUNCE_COUNT_MAX = 3;

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
    if ((scene.config.x & SCENE_CLOUDS_ONLY) != 0)
        media.remove();
    CloudMedium clouds = CloudMedium(scene);
    SunFrame sun_frame = SunFrame(frame.sun_direction.xyz);
    float cos_sun_radius = cos(SUN_ANGULAR_RADIUS);
    Spectrum sun_irradiance;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        sun_irradiance[k] = solar_irradiance(w.wavelength[k]);

    float3 origin = frame.camera_position.xyz;
    Spectrum radiance = 0.0;
    Spectrum throughput = 1.0;
    bool multiple_scattering = (scene.config.x & SCENE_SINGLE_SCATTERING) == 0;
    bool sky_light = (scene.config.x & SCENE_CLOUDS_ONLY) == 0;
    RayCone cone = { 0.0, s.pixel_spread(), true };
    for (uint bounce = 0; bounce < min(BOUNCE_COUNT_MAX, scene.config.y); bounce++)
    {
        float2 atmosphere = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
        if (atmosphere.y < 0.0)
            break;
        float t_surface = intersect_surface(origin, dir, scene, cone);
        float t_end = t_surface == SURFACE_MISS ? atmosphere.y : t_surface;

        float4 u = s.rng.next4();
        float4 v = s.rng.next4();
        float3 sun = sample_cone(u.xy, cos_sun_radius, frame.sun_direction.xyz);
        SegmentMarch march = SegmentMarch(media, clouds, sun_frame, w.u, origin, dir, sun, float3(u.zw, v.x), bounce > 0,
                                          multiple_scattering, sky_light);
        march.march(max(atmosphere.x, 0.0), t_end);
        radiance += throughput * march.segment.in_scattering * sun_irradiance;
        throughput *= march.segment.transmittance;
        if (t_surface == SURFACE_MISS)
            break;

        float3 position = origin + t_surface * dir;
        RayCone footprint = { cone.width_at(t_surface), cone.spread, cone.streams };
        SurfaceSample surface = sample_surface(position, footprint, scene);
        float3 lifted = offset_surface_point(position, scene);
        RayCone sun_cone = { cone.width_at(t_surface), cone.spread, false };
        bool sun_visible = surface_visible(lifted, sun, scene, sun_cone);
        ColumnDensity sun_column = ColumnDensity(column_density_lut, lifted, sun);
        float sun_cloud = sun_visible && sun_column.visible && clouds.active()
                              ? clear_air_cloud_transmittance(lifted, sun, sun_frame, clouds, v.yz)
                              : 1.0;
        BounceSample bounce = sample_bounce(surface, -dir, s.rng.next4().xyz, scene);
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        {
            Reflectance albedo = surface_reflectance(surface, srgb_to_spectrum_basis, w.wavelength[k]);
            radiance[k] += throughput[k] * surface_emission(surface, w.wavelength[k], scene);
            float sun_visibility = sun_visible ? sun_column.transmittance(media.extinction[k]) * sun_cloud : 0.0;
            SurfaceResponse direct = surface_brdf(surface, albedo, -dir, sun, scene);
            radiance[k] += throughput[k] * sun_visibility * sun_irradiance[k] * direct.brdf * direct.n_dot_l;
            SurfaceResponse bounced = surface_brdf(surface, albedo, -dir, bounce.dir, scene);
            throughput[k] *= bounced.brdf * bounced.n_dot_l / bounce.pdf;
        }
        if (!(bounce.pdf > 0.0))
            break;
        origin = lifted;
        cone.advance(t_surface, SURFACE_BOUNCE_SPREAD);
        dir = bounce.dir;
    }

    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
    {
        if (!(radiance[k] >= 0.0) || isinf(radiance[k]))
            radiance[k] = 0.0;
    }
    s.accumulate(xyz_to_rec709(mul(w.xyz_weight, radiance)));
}
