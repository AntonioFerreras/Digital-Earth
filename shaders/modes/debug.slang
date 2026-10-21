// A sun-lit Lambertian planet in empty space: no atmosphere, no maps, one
// wavelength per sample. It exercises the camera, the spectral pipeline,
// accumulation, and post with nothing to tune, so a wrong image points at the
// renderer core rather than at the physics. With the tiles on, it shows their
// terrain instead, coloured by the level of the height tile each pixel reads, lit
// from the camera, outlined per tile, and striped while a finer tile is on its way.

import common.constants;
import common.geometry;
import common.ray_cone;
import earth.tile_terrain;
import earth.tiles;
import shared.bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

// Encoded sRGB per level, coarse to fine.
static const float3 LEVEL_COLOURS[10] = {
    float3(0.35, 0.35, 0.35), float3(0.45, 0.25, 0.60), float3(0.25, 0.30, 0.80),
    float3(0.15, 0.55, 0.85), float3(0.10, 0.70, 0.60), float3(0.25, 0.75, 0.25),
    float3(0.70, 0.80, 0.15), float3(0.95, 0.75, 0.10), float3(0.95, 0.45, 0.10),
    float3(0.90, 0.15, 0.15),
};

// The encoded sRGB colour of the tile terrain hit `t` along `dir`.
float3 tile_level_colour(float3 origin, float3 dir, float t, RayCone cone, uint2 pixel)
{
    float3 position = origin + t * dir;
    RayCone footprint = { cone.width_at(t), cone.spread, false };
    uint wanted = min(tile_level(footprint.width), earth_tiles.height.x);
    TileHit hit = TileHit(earth_tiles.height, FacePoint(normalize(position)), wanted, false);
    float3 colour = LEVEL_COLOURS[min(hit.level, 9u)];
    float lit = 0.25 + 0.75 * max(dot(tile_normal(position, footprint), -dir), 0.0);
    // A few pixels wide at any distance.
    float tile_width = FACE_SPAN / float(1u << hit.level);
    float2 edge = min(hit.local, 1.0 - hit.local) * tile_width;
    if (min(edge.x, edge.y) < 1.5 * footprint.width)
        lit *= 0.3;
    if (hit.level < wanted && ((pixel.x + pixel.y) / 6u) % 2u == 0u)
        colour = float3(1.0);
    return colour * lit;
}

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void debug_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 dir = s.camera_ray_direction();
    float3 sun = frame.sun_direction.xyz;
    float3 origin = frame.camera_position.xyz;

    if ((scene.config.x & SCENE_TILES) != 0)
    {
        WavelengthBundle w = sample_wavelength_bundle(cie, s.rng.next4().x);
        Spectrum radiance = Spectrum(0.0);
        RayCone cone = { 0.0, s.pixel_spread(), true };
        float t = tile_intersect(origin, dir, cone, FLOAT_MAX);
        // Lit by a zenith sun, so the levels read alike day and night.
        if (t > 0.0)
        {
            float3 colour = tile_level_colour(origin, dir, t, cone, thread.xy);
            [ForceUnroll]
            for (uint k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
                radiance[k] = srgb_to_spectrum(srgb_to_spectrum_basis, colour, w.wavelength[k]) / PI * solar_irradiance(w.wavelength[k]);
        }
        s.accumulate(xyz_to_rec709(mul(w.xyz_weight, radiance)));
        return;
    }

    WavelengthSample w = sample_wavelength(cie, s.rng.next4().x);
    float radiance = 0.0;
    float2 hit = ray_sphere(origin, dir, PLANET_RADIUS);
    if (hit.x > 0.0)
    {
        float3 n = normalize(origin + hit.x * dir);
        float albedo = srgb_to_spectrum(srgb_to_spectrum_basis, float3(scene.surface.x), w.wavelength);
        radiance = albedo / PI * solar_irradiance(w.wavelength) * max(dot(n, sun), 0.0);
    }
    else if (dot(dir, sun) >= cos(SUN_ANGULAR_RADIUS))
    {
        radiance = planck(SUN_TEMPERATURE, w.wavelength);
    }
    s.accumulate(xyz_to_rec709(radiance * w.cmf * w.rcp_pdf));
}
