// The cloud bench (src/render/CloudBench.zig): one cloud volume lit by the sun and
// the sky (modes/cloud_bench_sky.slang), rendered two ways into accumulations of
// their own, which `resolve_main` shows. The reference path-traces it exactly; the
// approximation marches the sun's single scattering exactly, and takes the high order
// scattering and the sky's single scattering from the transfer probes
// (modes/cloud_bench_probes.slang). The droplets are grey, so paths carry one
// throughput.

import atmosphere.media;
import atmosphere.phase;
import clouds.bench_volume;
import common.constants;
import common.rng;
import common.sampling;
import common.spherical_harmonics;
import shared.bindings;
import shared.uniforms;

[format("rgba32f")]
RWTexture2D<float4> bench_reference;
// rgb: radiance; a: the cloud's opacity along the view.
[format("rgba32f")]
RWTexture2D<float4> bench_approximation;
// modes/cloud_bench_sky.slang's: the sky by direction, for the light, and through the
// camera, for the background.
Texture2D<float4> bench_sky;
Texture2D<float4> bench_sky_view;
Texture2D<float4> bench_sun;
// src/render/ProbeLayout.zig's leaf per finest cell, x fastest: its slot in the low
// LEAF_SLOT_BITS, its level above them, or NO_LEAF.
StructuredBuffer<uint> bench_probe_cells;
// modes/cloud_bench_probes.slang's atlas of the probes' light at each leaf's corners.
Texture3D<float4> bench_probe_light;
// src/render/BlueNoise.zig's tile.
Texture2D<float4> blue_noise;

static const uint LEAF_SLOT_BITS = 27;
static const uint NO_LEAF = 0xffffffff;

// Of the approximation's march, in voxels: half a voxel, as the trilinear filter
// varies within one.
static const float MARCH_STEP_VOXELS = 0.5;
static const float MARCH_TRANSMITTANCE_MIN = 1e-4;
// Half-width of the probe reads' tent offset, in finest spacings; a full spacing is a
// tricubic B-spline of the lattice, narrower keeps more of its detail.
static const float PROBE_JITTER_SPACINGS = 0.5;

// The approximation's jitters, in [0, 1): with CLOUD_BENCH_BLUE_NOISE the blue noise
// tile at the pixel, shifted per accumulated sample by the R4 sequence (Roberts 2018)
// and per draw along the ray by another, so that every draw of a sample stays blue
// across the screen while a pixel's samples and a ray's draws stratify; the shifts
// wrap in 32-bit fixed point, which no sample count degrades. Without it, the hash.
struct Jitter
{
    uint4 fixed;
    Rng rng;
    bool blue;

    __init(uint2 pixel, uint sample, Rng rng)
    {
        this.rng = rng;
        blue = (bench.options.z & CLOUD_BENCH_BLUE_NOISE) != 0;
        uint2 size;
        blue_noise.GetDimensions(size.x, size.y);
        float4 texel = blue_noise.Load(int3(pixel % size, 0));
        fixed = uint4(texel * 4294967295.0) + sample * JITTER_SAMPLE_SHIFT;
    }

    [mutating]
    float4 next4()
    {
        if (!blue)
            return rng.next4();
        float4 value = float4(fixed >> 8u) * (1.0 / 16777216.0);
        fixed += JITTER_DRAW_SHIFT;
        return value;
    }
};
// 2^32 times the R4 sequence's steps, and the golden ratio's and R2's for the draws.
static const uint4 JITTER_SAMPLE_SHIFT = uint4(3518318545u, 2882110344u, 2360940464u, 1933992832u);
static const uint4 JITTER_DRAW_SHIFT = uint4(2654435769u, 3242174889u, 2447445413u, 3037000500u);

// A tent over (-1, 1) from a uniform draw, by its inverse distribution, which keeps a
// blue draw blue where the sum of two would not.
float3 tent(float3 u)
{
    return select(u < 0.5, sqrt(2.0 * u) - 1.0, 1.0 - sqrt(2.0 - 2.0 * u));
}
// Optical depth towards the sun past which its beam, exp(-depth), adds nothing visible.
static const float BEAM_DEPTH_MAX = 12.0;

// The light: the sun's irradiance and the sky's radiance, each unless it is off.
struct Light
{
    float3 sun;
    bool sky;

    __init()
    {
        sun = (bench.options.z & CLOUD_BENCH_SUN) != 0 ? bench_sun.Load(int3(0, 0, 0)).rgb : float3(0.0);
        sky = (bench.options.z & CLOUD_BENCH_SKY) != 0;
    }

    float3 sky_radiance(float3 dir)
    {
        return sky ? bench_sky.SampleLevel(linear_repeat_u, sky_uv(dir), 0.0).rgb : float3(0.0);
    }

    // The sky behind `pixel`, from the view's texels, a stride of pixels apart, by the
    // bicubic B-spline's four filtered fetches: bilinear creases over such a stride.
    float3 sky_background(uint2 pixel)
    {
        if (!sky)
            return float3(0.0);
        uint2 texels;
        bench_sky_view.GetDimensions(texels.x, texels.y);
        float2 size = float2(texels);
        CubicTaps<2> taps = CubicTaps<2>((float2(pixel) + 0.5) / float2(frame.frame.yz) * size - 0.5, size);
        float3 radiance = 0.0;
        for (uint lobe = 0; lobe < 4; lobe++)
        {
            bool2 up = bool2((lobe & 1) != 0, (lobe & 2) != 0);
            float2 weight = select(up, taps.g1, taps.g0);
            radiance += weight.x * weight.y * bench_sky_view.SampleLevel(linear_clamp, select(up, taps.upper, taps.lower), 0.0).rgb;
        }
        return radiance;
    }
};

// Of `light`'s (phi, F), the radiance scattered towards `towards` per unit scattering
// coefficient by a phase of first moment `g`, by the M1 closure: of the non-negative
// radiance distributions with that fluence and flux, a von Mises-Fisher lobe about F.
// Scattering keeps it one and multiplies its mean cosine |F| / phi by g, as convolution
// on the sphere multiplies first moments; Banerjee et al. 2005 invert the mean cosine
// to the concentration. Where the light is barely directional it is P1's radiance.
float closure_radiance(float4 light, float3 towards, float g)
{
    if (!(light.x > 0.0))
        return 0.0;
    float flux = length(light.yzw);
    // Noise in the probes may leave |F| above phi.
    float mean_cosine = clamp(g * flux / light.x, -MEAN_COSINE_MAX, MEAN_COSINE_MAX);
    float cos_theta = flux > 0.0 ? sign(mean_cosine) * dot(light.yzw, towards) / flux : 0.0;
    float a = abs(mean_cosine);
    float kappa = a * (3.0 - a * a) / (1.0 - a * a);
    float lobe = kappa > KAPPA_ISOTROPIC ? kappa / (2.0 * PI * (1.0 - exp(-2.0 * kappa))) * exp(kappa * (cos_theta - 1.0))
                                         : 1.0 / (4.0 * PI);
    return light.x * lobe;
}

static const float MEAN_COSINE_MAX = 0.99;
// Below it the lobe is isotropic to float precision.
static const float KAPPA_ISOTROPIC = 1e-4;

// The probes' light at `pos`, interpolated over the corners of its leaf, scattered
// along `-dir` per unit scattering coefficient through a phase of first moment `g`,
// which alone sees their flux: by `closure_radiance`, or with CLOUD_BENCH_P1_CLOSURE by
// P1, (phi + 3 g F . -dir) / 4 pi, which falls below zero where the light is strongly
// directional.
float3 probe_radiance(float3 pos, float3 dir, float g, float3 u)
{
    uint3 cells = bench.probe_cells.xyz;
    if ((bench.options.z & CLOUD_BENCH_PROBE_JITTER) != 0)
        pos += tent(u) * PROBE_JITTER_SPACINGS * bench.probe_spacing.xyz;
    // Clamped to the outermost probes.
    float3 lattice = clamp((pos - bench.probe_origin.xyz) / bench.probe_spacing.xyz, 0.0, float3(cells));
    uint3 cell = min(uint3(lattice), cells - 1);
    uint leaf = bench_probe_cells[cell.x + cells.x * (cell.y + cells.y * cell.z)];
    if (leaf == NO_LEAF)
        return 0.0;
    uint level = leaf >> LEAF_SLOT_BITS;
    uint slot = leaf & ((1u << LEAF_SLOT_BITS) - 1);
    float3 within = (lattice - float3((cell >> level) << level)) / float(1u << level);
    uint3 size;
    bench_probe_light.GetDimensions(size.x, size.y, size.z);
    uint channel_depth = size.z / 3;
    uint2 leaves = size.xy / 2;
    // Between the leaf's texel centres the filter interpolates its corners exactly.
    float3 texel = 2.0 * float3(slot % leaves.x, slot / leaves.x % leaves.y, slot / (leaves.x * leaves.y)) + 0.5 + within;
    bool p1 = (bench.options.z & CLOUD_BENCH_P1_CLOSURE) != 0;
    float3 radiance;
    for (uint c = 0; c < 3; c++)
    {
        float3 uvw = float3(texel.xy, float(c * channel_depth) + texel.z) / float3(size);
        float4 light = bench_probe_light.SampleLevel(linear_clamp, uvw, 0.0);
        radiance[c] = p1 ? max(light.x - 3.0 * g * dot(light.yzw, dir), 0.0) / (4.0 * PI) : closure_radiance(light, -dir, g);
    }
    return radiance;
}

// The camera ray of the pixel, in the cloud frame: through a point drawn within the
// pixel, which antialiases the reference, or through the pixel's centre, which spares
// the approximation the wisps' sub-pixel noise.
struct CameraRay
{
    PixelSample s;
    float3 pos;
    float3 dir;

    __init(uint2 thread, uint sample_index, bool pixel_jitter)
    {
        s = PixelSample(thread);
        // Each accumulation keys its own samples, as either restarts alone.
        s.rng = Rng(thread, sample_index + frame.frame.w);
        dir = camera_direction(float2(thread) + (pixel_jitter ? s.rng.next4().xy : float2(0.5)));
        pos = BenchVolume().to_cloud(frame.camera_position.xyz);
    }
};

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void reference_main(uint3 thread: SV_DispatchThreadID)
{
    CameraRay ray = CameraRay(thread.xy, bench.samples.x, true);
    if (!ray.s.valid)
        return;
    BenchVolume volume = BenchVolume();
    Light light = Light();
    float3 sun = frame.sun_direction.xyz;
    float3 pos = ray.pos;
    float3 dir = ray.dir;
    float3 radiance = 0.0;
    for (uint bounce = 0; bounce < bench.options.y; bounce++)
    {
        float t = sample_collision(volume, pos, dir, FLOAT_MAX, ray.s.rng);
        if (t < 0.0)
        {
            radiance += bounce == 0 ? light.sky_background(ray.s.pixel) : light.sky_radiance(dir);
            break;
        }
        pos += t * dir;
        bool reduce_peak = bounce > 0;
        if (any(light.sun > 0.0))
            radiance += light.sun * cloud_phase(dot(dir, sun), reduce_peak) * transmittance(volume, pos, sun, ray.s.rng);
        dir = sample_cloud_phase(ray.s.rng.next4().xyz, dir, reduce_peak);
    }
    float3 sum = bench.samples.x == 0 ? float3(0.0) : bench_reference[ray.s.pixel].rgb;
    bench_reference[ray.s.pixel] = float4(sum + (all(isfinite(radiance)) ? radiance : float3(0.0)), 1.0);
}

// The light scattered at `pos` along `-dir` per unit scattering coefficient that has
// scattered before, by one scatter followed exactly: towards a direction drawn from the
// phase, or from its broad part alone, so that the phase cancels, the light from the
// first collision, its sun single scattering marched and the rest read from the
// probes, or the sky's radiance where there is none. The probes' light then has
// scattered twice before reaching `pos`, which varies far more slowly than once.
float3 gathered_radiance(BenchVolume volume, Light light, float3 pos, float3 dir, bool broad, inout Rng rng)
{
    float3 u = rng.next4().xyz;
    float3 towards = broad ? sample_cloud_phase_broad(u, dir, false) : sample_cloud_phase(u, dir, false);
    float t = sample_collision(volume, pos, towards, FLOAT_MAX, rng);
    if (t < 0.0)
        return light.sky_radiance(towards);
    float3 collision = pos + t * towards;
    float3 sun = frame.sun_direction.xyz;
    float3 radiance = probe_radiance(collision, towards, cloud_phase_mean_cosine(true), rng.next4().xyz);
    if (any(light.sun > 0.0))
    {
        float sun_depth = optical_depth(volume, collision, sun, volume.voxel, BEAM_DEPTH_MAX, uint(bench.march.z));
        radiance += light.sun * cloud_phase(dot(towards, sun), true) * exp(-sun_depth);
    }
    return radiance;
}

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void approximation_main(uint3 thread: SV_DispatchThreadID)
{
    CameraRay ray = CameraRay(thread.xy, bench.samples.y, false);
    if (!ray.s.valid)
        return;
    BenchVolume volume = BenchVolume();
    Light light = Light();
    float3 sun = frame.sun_direction.xyz;
    bool sun_shines = any(light.sun > 0.0);
    float3 pos = ray.pos;
    float3 dir = ray.dir;
    Jitter jitter = Jitter(ray.s.pixel, bench.samples.y + frame.frame.w, ray.s.rng);
    float phase = jitter.next4().x;
    float cos_sun = dot(dir, sun);
    float single_phase = cloud_phase(cos_sun, false);
    float g = cloud_phase_mean_cosine(false);
    bool multiple_scattering = bench.options.y > 1;
    bool gather = (bench.options.z & CLOUD_BENCH_PROBE_GATHER) != 0;
    // The view's scatter splits into a forward peak, taken as straight on, and a broad
    // rest. Light the peak scatters keeps to the view, so with R the radiance arriving
    // along it, dR/dt = (1 - peak) sigma R - sigma (single scattering + (1 - peak) the
    // rest's in-scattering): the view attenuates at (1 - peak) sigma and scatters the
    // sun's beam by the whole phase, which keeps the peak's halo exact.
    bool forward_peak = multiple_scattering && (bench.options.z & CLOUD_BENCH_FORWARD_PEAK) != 0;
    float peak = forward_peak ? cloud_phase_peak(false) : 0.0;
    float broad_g = forward_peak ? cloud_phase_broad_mean_cosine(false) : g;
    // The weight of every step so far and the step chosen among them in proportion.
    float gather_weight = 0.0;
    float3 gather_position = 0.0;
    float step = MARCH_STEP_VOXELS * volume.voxel;

    float3 radiance = 0.0;
    float transmittance = 1.0;
    float2 span = grid_span(volume, pos, dir, FLOAT_MAX);
    if (span.x < span.y)
    {
        for (MajorantWalk walk = MajorantWalk(volume, pos, dir, span.x, span.y, uint(bench.march.y));
             walk.active() && transmittance > MARCH_TRANSMITTANCE_MIN; walk.advance(volume))
        {
            if (!(walk.majorant > 0.0))
                continue;
            uint count;
            float dt;
            walk.steps(step, bench.march.w * volume.voxel, -log(transmittance), bench.march.x, count, dt);
            for (uint i = 0; i < count; i++)
            {
                float3 position = pos + (walk.start + (float(i) + phase) * dt) * dir;
                float extinction = volume.extinction_cubic(position);
                if (!(extinction > 0.0))
                    continue;
                float sun_depth = sun_shines ? optical_depth(volume, position, sun, volume.voxel, BEAM_DEPTH_MAX, uint(bench.march.z)) : FLOAT_MAX;
                float sun_light = single_phase * exp(-sun_depth);
                float3 scattered = 0.0;
                if (multiple_scattering)
                {
                    if (gather)
                    {
                        float weight = extinction * transmittance * step_integral((1.0 - peak) * extinction * dt) * dt;
                        gather_weight += weight;
                        if (ray.s.rng.next4().x * gather_weight < weight)
                            gather_position = position;
                    }
                    else
                        scattered = (1.0 - peak) * probe_radiance(position, dir, broad_g, jitter.next4().xyz);
                }
                float3 in_scattering = extinction * (light.sun * sun_light + scattered);
                float step_depth = (1.0 - peak) * extinction * dt;
                radiance += in_scattering * transmittance * step_integral(step_depth) * dt;
                transmittance *= exp(-step_depth);
            }
        }
    }
    if (gather_weight > 0.0)
        radiance += gather_weight * (1.0 - peak) * gathered_radiance(volume, light, gather_position, dir, forward_peak, ray.s.rng);
    radiance += transmittance * light.sky_background(ray.s.pixel);
    float4 sum = bench.samples.y == 0 ? float4(0.0) : bench_approximation[ray.s.pixel];
    bench_approximation[ray.s.pixel] = sum + float4(radiance, 1.0 - transmittance);
}

static const float3 REC709_LUMINANCE = float3(0.2126, 0.7152, 0.0722);
// Tints of the error view at a factor of two too bright and too dark.
static const float3 BRIGHTER_TINT = float3(1.0, 0.15, 0.1);
static const float3 DARKER_TINT = float3(0.1, 0.35, 1.0);

// Writes the view's mean into `accum`, scaled so that the display pass's division by
// the frame count leaves it as it is.
[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void resolve_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 reference = bench.samples.x > 0 ? bench_reference[s.pixel].rgb / float(bench.samples.x) : float3(0.0);
    float4 approximation = bench.samples.y > 0 ? bench_approximation[s.pixel] / float(bench.samples.y) : float4(0.0);
    float3 shown;
    switch (bench.options.x)
    {
    case CLOUD_BENCH_VIEW_REFERENCE:
        shown = reference;
        break;
    case CLOUD_BENCH_VIEW_APPROXIMATION:
        shown = approximation.rgb;
        break;
    case CLOUD_BENCH_VIEW_SPLIT:
    {
        uint middle = frame.frame.y / 2;
        shown = s.pixel.x < middle ? reference : s.pixel.x > middle ? approximation.rgb : float3(0.0);
        break;
    }
    case CLOUD_BENCH_VIEW_ERROR:
    {
        float luminance = dot(reference, REC709_LUMINANCE);
        float error = clamp(log2(dot(approximation.rgb, REC709_LUMINANCE) / luminance), -1.0, 1.0);
        float3 tint = lerp(float3(1.0), error > 0.0 ? BRIGHTER_TINT : DARKER_TINT, abs(error));
        shown = luminance > 0.0 ? luminance * tint : float3(0.0);
        break;
    }
    default:
        shown = approximation.a;
        break;
    }
    accum[s.pixel] = float4(shown * float(frame.frame.x + 1), 1.0);
}
