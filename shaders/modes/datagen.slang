// MLP training data: the symmetric scene's linear Rec.709 radiance at uvwz
// points, each the mean of many spectral paths through modes/path_radiance.slang.
//
// One thread group sums one chunk of one point's paths; `accum` receives the
// group's sum at texel SV_GroupID (rgb, and in w the number of paths that saw
// the sun disk). The host adds a point's chunks in order and divides by the
// path count, so every value is a fixed sequence of f32 and f64 additions:
// no atomics, and the output depends on the seed alone.

import atmosphere.bruneton;
import common.ray_cone;
import common.rng;
import modes.path_radiance;
import shared.bindings;
import shared.uniforms;
import spectral.cie;

static const uint GROUP_THREADS = 256;

ConstantBuffer<DatagenUniforms> datagen;
// The batch's points, uvwz each.
StructuredBuffer<float4> points;

groupshared float4 partial_sums[GROUP_THREADS];

[shader("compute")]
[numthreads(GROUP_THREADS, 1, 1)]
void datagen_main(uint3 group: SV_GroupID, uint thread: SV_GroupIndex)
{
    uint2 texel = uint2(group.x, group.y + datagen.dispatch.x);
    uint group_index = texel.y * frame.frame.y + texel.x;
    uint chunk_count = datagen.batch.w;
    // Uniform per group, so the barriers below stay in uniform control flow.
    if (group_index >= datagen.dispatch.y)
        return;
    uint point = group_index / chunk_count;
    uint chunk = group_index - point * chunk_count;
    uint path_count = datagen.batch.y;
    uint path_end = min((chunk + 1) * datagen.batch.z, path_count);

    ScatteringParams p = scattering_params_from_uvwz(points[point]);
    CanonicalRay ray = canonical_ray(p.r, p.mu, p.mu_s, p.nu);
    bool include_sun_disk = (datagen.dispatch.z & DATAGEN_EXCLUDE_SUN_DISK) == 0;

    float4 sum = float4(0.0);
    for (uint path = chunk * datagen.batch.z + thread; path < path_end; path += GROUP_THREADS)
    {
        // Keyed by (point, path, seed): independent of chunking and batching.
        Rng rng = Rng(uint2(datagen.batch.x + point, path), frame.frame.x);
        WavelengthBundle w = sample_wavelength_bundle(cie, rng.next4().x);
        // The symmetric scene reads no textures, so the footprint does not matter.
        RayCone cone = { 0.0, 0.0, false };
        PathSample s = path_radiance(ray.pos, ray.dir, ray.sun, w.wavelength, cone, rng, scene);
        // Escaping rays see the sun disk in the canonical frame, as the original's
        // batch_path_trace did; nothing else of the sky is lit in this scene.
        Spectrum sky = s.escaped ? sky_radiance(ray.dir, ray.sun, w.wavelength, scene) : Spectrum(0.0);
        Spectrum radiance = s.radiance + (include_sun_disk ? sky : Spectrum(0.0));
        sum += float4(xyz_to_rec709(mul(w.xyz_weight, radiance)), any(sky > 0.0) ? 1.0 : 0.0);
    }

    // A fixed pairwise tree, so the group sum is the same on every run.
    partial_sums[thread] = sum;
    for (uint stride = GROUP_THREADS / 2; stride > 0; stride /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (thread < stride)
            partial_sums[thread] += partial_sums[thread + stride];
    }
    if (thread == 0)
        accum[texel] = partial_sums[0];
}
