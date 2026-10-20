// The path-tracer mode: one path per pixel per frame through
// modes/path_radiance.slang, carrying a wavelength bundle, for either scene preset.

import atmosphere.bruneton;
import common.ray_cone;
import common.rng;
import modes.path_radiance;
import shared.bindings;
import shared.uniforms;
import spectral.cie;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void pathtrace_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 dir = s.camera_ray_direction();
    WavelengthBundle w = sample_wavelength_bundle(cie, s.rng.next4().x);
    float3 sun = frame.sun_direction.xyz;
    CanonicalRay ray = { frame.camera_position.xyz, dir, sun };
    // Rays that miss the atmosphere stay in the world frame: the round trip would
    // clamp their r onto the atmosphere and light empty space.
    if ((scene.config.x & SCENE_CANONICALIZE) != 0 && enter_atmosphere(ray.pos, dir))
    {
        // Through uvwz and back, so that the image depends on (r, mu, mu_s, nu)
        // at the MLP's texel resolution only: exactly what the MLP learns.
        ScatteringParams p = scattering_params_from_uvwz(scattering_uvwz(scattering_params_from_ray(ray.pos, dir, sun)));
        ray = canonical_ray(p.r, p.mu, p.mu_s, p.nu);
    }
    RayCone cone = { 0.0, s.pixel_spread(), true };
    PathSample path = path_radiance(ray.pos, ray.dir, ray.sun, w.wavelength, cone, s.rng, scene);
    Spectrum radiance = path.radiance + (path.escaped ? sky_radiance(dir, sun, w.wavelength, scene) : Spectrum(0.0));
    s.accumulate(xyz_to_rec709(mul(w.xyz_weight, radiance)));
}
