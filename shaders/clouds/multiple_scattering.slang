// The ray marchers' model of the clouds' multiple scattering, after Nubis Cubed
// (Schneider, "Nubis Cubed", Advances in Real-Time Rendering, SIGGRAPH 2023, slide
// 138). The dimensional profile, 0 at a cloud's edge and 1 deep inside, is the
// probability that light has scattered in from its first path, and the light it
// carries is absorbed with the optical depth towards the sun at one Beer-Lambert rate.
// The rate relaxes for views towards the sun and deep inside the cloud, which lights a
// sunlit cloud's core from within. A Henyey-Greenstein lobe makes the result
// directional, the second term of Nubis's direct scattering, transmittance * primary
// phase + multiple scattering * secondary phase.
//
// Nubis's constants apply to its summed density samples and voxel distances, whose
// scale it does not publish; here every rate and depth is per optical depth at full
// density instead, so that the model means the same at any cloud scale and extinction.
// Nubis has no gain either: light piles up inside a thick cloud of albedo 1 to several
// times what one scatter of the phase carries, so a strength scales the term. The
// constants are the scene's `cloud_scattering` and `cloud_profile`.

module multiple_scattering;

import atmosphere.phase;
import shared.uniforms;

// Nubis's view-to-sun cosine at and past which the rate is fully relaxed.
static const float RELAXED_SUN_COSINE = 0.9;

// The dimensional profile at `inside_depth`, how deep a point lies inside its cloud.
public float cloud_profile(float inside_depth, SceneUniforms scene)
{
    return saturate(inside_depth / scene.cloud_profile.y);
}

// The multiply scattered sunlight leaving a cloud point along `-dir`, per unit sun
// irradiance and unit scattering coefficient: `depth` is the cloud's optical depth
// towards the sun, `cos_sun` = dot(dir, sun), and `inside_depth` how deep the point
// lies inside its cloud, the distance to the cloud's edge as an optical depth at full
// density.
public float cloud_multiple_scattering(float depth, float cos_sun, float inside_depth, SceneUniforms scene)
{
    // Nubis's Remap and ValueRemap are taken as clamped, so that the rate moves
    // between its ends and no further.
    float towards_sun = lerp(scene.cloud_scattering.z, scene.cloud_scattering.w, saturate(inside_depth / scene.cloud_profile.z));
    float rate = lerp(scene.cloud_scattering.y, towards_sun, saturate(cos_sun / RELAXED_SUN_COSINE));
    float edge = lerp(1.0 - scene.cloud_profile.x, 1.0, cloud_profile(inside_depth, scene));
    return scene.cloud_scattering.x * edge * exp(-depth * rate) * hg_phase(cos_sun, scene.cloud_profile.w);
}
