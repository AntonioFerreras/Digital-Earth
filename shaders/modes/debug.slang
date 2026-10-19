// A sun-lit Lambertian planet in empty space: no atmosphere, no maps, one
// wavelength per sample. It exercises the camera, the spectral pipeline,
// accumulation, and post with nothing to tune, so a wrong image points at the
// renderer core rather than at the physics.

import common.constants;
import common.geometry;
import shared.bindings;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void debug_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 dir = s.camera_ray_direction();
    WavelengthSample w = sample_wavelength(cie, s.rng.next4().x);

    float3 sun = frame.sun_direction.xyz;
    float3 origin = frame.camera_position.xyz;
    float2 hit = ray_sphere(origin, dir, PLANET_RADIUS);

    float radiance = 0.0;
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
