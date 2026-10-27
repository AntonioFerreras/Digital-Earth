// The cloud bench's transfer probes: at the corners of src/render/ProbeLayout.zig's
// leaves over the volume, how the cloud carries
// distant light to each probe, independent of the light, so that only the density
// restarts their bake. A probe holds the fluence phi and the flux F of the light
// arriving at it, the radiance of the diffusion approximation's P1 form,
// (phi + 3 F . w) / 4 pi along w, as functions of the light's direction.
//
// The sky's light is smooth over directions, so the sky takes a linear transfer: per
// spherical harmonic j of its radiance, phi_j and F_j of the light reaching the probe,
// straight or scattered on its way, the sky's single and high order scattering where
// the approximation scatters it towards the viewer.
//
// The sun is one direction, and a probe in its shadow takes orders of magnitude less
// of it than from directions where the cloud is thin, which a linear transfer of few
// harmonics rings below zero. The sun's transfer is therefore fitted in the log: phi
// as exp(sum of harmonics), and F / phi, which lies within the unit ball, as linear
// harmonics. The sun's single scattering the approximation marches exactly, so this
// transfer holds light scattered on its way only.
//
// `bake_probes` adds one backward random walk per probe per dispatch, `fit_probes`
// fits the sun's transfer from the walks' direction bins and contracts the sky's from
// the sums and the bins, `light_probes` lights the probes with the present sun and sky
// from the fits alone, and `fill_probe_atlas` copies their light to their leaves'
// corners for the approximation (modes/cloud_bench.slang).

import atmosphere.phase;
import clouds.bench_volume;
import common.constants;
import common.octahedral;
import common.rng;
import common.sampling;
import common.spherical_harmonics;
import shared.bindings;
import shared.uniforms;

StructuredBuffer<float4> bench_probe_positions;
// Element probe * SH3_COUNT + j: the probe's sums over its walks of the sky's transfer
// for harmonic j, x: phi_j, yzw: F_j.
RWStructuredBuffer<float4> bench_probe_transfer;
// Element b * probes + probe: the probe's sums over its walks of the sun's light from
// directions in bin b of the equal-area octahedral map, x: phi, yzw: F. Bin-major, as a
// photon feeds one bin, so that its deposits at neighbouring probes share cache lines;
// src/render/ProbeCache.zig stores them probe-major.
RWStructuredBuffer<float4> bench_probe_sun_bins;
// The bins as raw bytes, for the photons' atomic adds.
RWByteAddressBuffer bench_probe_sun_bins_raw;
RWStructuredBuffer<SunTransfer> bench_probe_sun;
// Per quantised coefficient, its least value over the lit probes at 4 i and its greatest
// at 4 (SUN_QUANTIZED + i), as `orderable` uints for the atomics.
RWByteAddressBuffer bench_probe_sun_ranges;
RWStructuredBuffer<SkyTransfer> bench_probe_sky;
// Element probe * 3 + c: the probe's light in colour channel c, x: phi, yzw: F.
RWStructuredBuffer<float4> bench_probe_light_values;
// Element leaf * 8 + corner: the probe at the leaf's corner, x in bit 0 of the
// corner, y in bit 1, z in bit 2.
StructuredBuffer<uint> bench_probe_corners;
// modes/cloud_bench.slang's per finest cell leaves.
StructuredBuffer<uint> bench_probe_cells;
static const uint LEAF_SLOT_BITS = 27;
static const uint NO_LEAF = 0xffffffff;
// Per probe, the integral of its trilinear weight over its leaves: an eighth of each
// leaf's volume.
StructuredBuffer<float> bench_probe_volumes;
// modes/cloud_bench.slang's atlas: leaf slot s's corners at texels 2 (s's place in
// rows of width / 2 and layers of height / 2) + the corner's bits, colour channel c
// in the c-th third along z.
[format("rgba16f")]
RWTexture3D<float4> bench_probe_light_target;
// modes/cloud_bench_sky.slang's.
Texture2D<float4> bench_sun;
// Per cell of SUN_DEPTH_STRIDE voxels, the least optical depth from its centre towards
// any bin's centre, a little past SUN_DEPTH_BAKED where it exceeds it.
[format("r16f")]
RWTexture3D<float> bench_sun_least;
// Per cell, the least of `bench_sun_least` there and in the cells around it, so that it
// bounds the depth from anywhere within the cell.
[format("r16f")]
RWTexture3D<float> bench_sun_reach;

// Per unit irradiance from the sun's direction s: phi(s) = exp(sum_j c_j Y_j(s))
// through order 5 and F(s) = phi(s) sum_k r_k Y_k(s) over the first four harmonics.
// c_0 is a float; the other coefficients, c_1 on and then r, are bytes over the field's
// range of each in `bench_probe_sun_ranges`, four to a uint.
static const uint SUN_FLUX_HARMONICS = 4;
// The log fluence's harmonics fitted; the rest stay zero.
uint sun_harmonics()
{
    return (bench.options.z & CLOUD_BENCH_SUN_ORDER3) != 0 ? SH3_COUNT : SH5_COUNT;
}
static const uint SUN_QUANTIZED = SH5_COUNT - 1 + 3 * SUN_FLUX_HARMONICS;
static const uint SUN_PACKED_WORDS = (SUN_QUANTIZED + 3) / 4;
struct SunTransfer
{
    float dc;
    uint packed[SUN_PACKED_WORDS];

    float coefficient(uint i)
    {
        return dequantise(packed_byte(packed, i), coefficient_range(i));
    }
};

// Per harmonic j of the sky's radiance through order 3, phi_j and F_j of its light at
// the probe, single and high order: the walks' sums contracted with the bins. phi_0 is
// a float; the other values, j-major, are bytes over +-sqrt(2 l + 1) phi_0, which
// bounds each, as the transfer is a positive kernel with |F| <= phi and the harmonics
// of degree l lie within sqrt((2 l + 1) / 4 pi).
static const uint SKY_QUANTIZED = 4 * SH3_COUNT - 1;
static const uint SKY_PACKED_WORDS = (SKY_QUANTIZED + 3) / 4;
struct SkyTransfer
{
    float dc;
    uint packed[SKY_PACKED_WORDS];

    float2 range(uint i)
    {
        uint l = uint(sqrt(float((i + 1) / 4)));
        float bound = sqrt(2.0 * float(l) + 1.0) * dc;
        return float2(-bound, bound);
    }

    float4 light(uint j)
    {
        float4 light;
        for (uint k = 0; k < 4; k++)
        {
            uint i = 4 * j + k;
            light[k] = i == 0 ? dc : dequantise(packed_byte(packed, i - 1), range(i - 1));
        }
        return light;
    }
};

uint packed_byte<let N : uint>(uint packed[N], uint i)
{
    return packed[i / 4] >> (8 * (i % 4)) & 255;
}

void pack_byte<let N : uint>(inout uint packed[N], uint i, uint byte)
{
    packed[i / 4] |= byte << (8 * (i % 4));
}

static const uint PROBE_GROUP_SIZE = 64;

static const uint SUN_BIN_SIDE = 8;
static const uint SUN_BIN_COUNT = SUN_BIN_SIDE * SUN_BIN_SIDE;
// The dimmest share of a probe's brightest direction that the log fit resolves: bins
// below it, where few walks have reached yet, would otherwise pull the fit to -inf.
static const float SUN_FLOOR = 1e-5;
// Of a probe no walk has carried any sunlight to.
static const float SUN_DARK_LOG = -100.0;
// Optical depth towards the sun past which a collision counts as unlit: each such
// collision would add less than exp(-SUN_DEPTH_MAX), and marching on costs the most.
static const float SUN_DEPTH_MAX = 8.0;
// The least depths are baked past SUN_DEPTH_MAX, so that a collision whose reach lies
// past SUN_REACH_SKIP, which bounds how far the true depth may fall below the reach
// away from cell centres and bin centres, can skip the sun.
static const float SUN_DEPTH_BAKED = 16.0;
static const float SUN_REACH_SKIP = 12.0;
static const uint SUN_DEPTH_STRIDE = 8;


uint sun_bin(float3 dir)
{
    uint2 cell = min(uint2(equal_area_sphere_to_square(dir) * float(SUN_BIN_SIDE)), SUN_BIN_SIDE - 1);
    return cell.x + SUN_BIN_SIDE * cell.y;
}

// `offset` within bin b, in [0, 1)^2 of its cell of the equal-area map.
float3 bin_direction(uint b, float2 offset)
{
    return equal_area_square_to_sphere((float2(b % SUN_BIN_SIDE, b / SUN_BIN_SIDE) + offset) / float(SUN_BIN_SIDE));
}

float sun_reach(BenchVolume volume, float3 pos)
{
    uint3 cells;
    bench_sun_reach.GetDimensions(cells.x, cells.y, cells.z);
    int3 cell = int3(floor((pos - volume.grid_min) / (float(SUN_DEPTH_STRIDE) * volume.voxel)));
    return bench_sun_reach[clamp(cell, int3(0), int3(cells) - 1)];
}

[shader("compute")]
[numthreads(4, 4, 4)]
void bake_sun_least(uint3 thread: SV_DispatchThreadID)
{
    uint3 cells;
    bench_sun_least.GetDimensions(cells.x, cells.y, cells.z);
    if (any(thread >= cells))
        return;
    BenchVolume volume = BenchVolume();
    float3 pos = volume.grid_min + (float3(thread) + 0.5) * float(SUN_DEPTH_STRIDE) * volume.voxel;
    float least = FLOAT_MAX;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
        least = min(least, optical_depth(volume, pos, bin_direction(b, 0.5), volume.voxel, SUN_DEPTH_BAKED, 0));
    bench_sun_least[thread] = least;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void bake_sun_reach(uint3 thread: SV_DispatchThreadID)
{
    uint3 cells;
    bench_sun_reach.GetDimensions(cells.x, cells.y, cells.z);
    if (any(thread >= cells))
        return;
    float least = FLOAT_MAX;
    for (uint i = 0; i < 27; i++)
    {
        int3 neighbour = clamp(int3(thread) + int3(i % 3, i / 3 % 3, i / 9) - 1, int3(0), int3(cells) - 1);
        least = min(least, bench_sun_least[neighbour]);
    }
    bench_sun_reach[thread] = least;
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void clear_probe_sums(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
        bench_probe_sun_bins[b * bench.probe_counts.x + probe] = 0.0;
    for (uint j = 0; j < SH3_COUNT; j++)
        bench_probe_transfer[probe * SH3_COUNT + j] = 0.0;
}

void add_sum(RWByteAddressBuffer sums, uint element, float4 value)
{
    for (uint k = 0; k < 4; k++)
    {
        float original;
        sums.InterlockedAddF32(16 * element + 4 * k, value[k], original);
    }
}

// The scattered light of this thread's photon in the leaf it last crossed, per corner,
// not yet added to the probes.
static uint splat_slot = NO_LEAF;
static float4 splat[8] = { float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0) };

// Adds `weight` times the path from `pos` along `dir` up to `length` to `bin`, at
// points a finest cell apart from `next` on, which carries over to the path's next
// part: each point's cell length times its leaf's corners' trilinear weights, over each
// probe's integral of its weight, a kernel estimate of the fluence, summed in registers
// until the photon leaves the leaf, across collisions, as the noise of drawing one
// corner would otherwise dominate the bins'. Light that has not collided is the sky's
// single scattering, which `bake_sky_transfer` estimates directly.
void deposit_points(float3 pos, float3 dir, float length, inout float next, float weight, bool collided, uint bin)
{
    uint3 cells = bench.probe_cells.xyz;
    float step = min(bench.probe_spacing.x, min(bench.probe_spacing.y, bench.probe_spacing.z));
    if (!collided)
    {
        next -= length;
        next -= floor(next / step) * step;
        return;
    }
    for (; next < length; next += step)
    {
        float3 lattice = (pos + next * dir - bench.probe_origin.xyz) / bench.probe_spacing.xyz;
        if (any(lattice < 0.0) || any(lattice >= float3(cells)))
            continue;
        uint3 cell = uint3(lattice);
        uint leaf = bench_probe_cells[cell.x + cells.x * (cell.y + cells.y * cell.z)];
        if (leaf == NO_LEAF)
            continue;
        uint level = leaf >> LEAF_SLOT_BITS;
        uint slot = leaf & ((1u << LEAF_SLOT_BITS) - 1);
        float3 within = (lattice - float3((cell >> level) << level)) / float(1u << level);
        if (slot != splat_slot)
            flush_splat(bin);
        splat_slot = slot;
        float weights[8] = corner_weights(within);
        for (uint corner = 0; corner < 8; corner++)
            splat[corner] += weights[corner] * weight * step * float4(1.0, dir);
    }
    next -= length;
}

void flush_splat(uint bin)
{
    if (splat_slot == NO_LEAF)
        return;
    for (uint corner = 0; corner < 8; corner++)
    {
        uint probe = bench_probe_corners[splat_slot * 8 + corner];
        if (splat[corner].x > 0.0)
            add_sum(bench_probe_sun_bins_raw, bin * bench.probe_counts.x + probe, splat[corner] / bench_probe_volumes[probe]);
        splat[corner] = 0.0;
    }
    splat_slot = NO_LEAF;
}

// Of a leaf's corners, their trilinear weights at `within` the leaf, x in bit 0.
float[8] corner_weights(float3 within)
{
    float weights[8];
    for (uint corner = 0; corner < 8; corner++)
    {
        float3 side = select((uint3(corner) >> uint3(0, 1, 2) & 1u) != 0u, within, 1.0 - within);
        weights[corner] = side.x * side.y * side.z;
    }
    return weights;
}

// `deposit_points` exactly: the part of the path in each leaf it crosses to one of the
// leaf's corners, drawn in proportion to the corner's trilinear weight integrated
// along the chord, which Simpson's rule gives exactly as the weight is cubic along a
// line.
void deposit_path(float3 pos, float3 dir, float length, float weight, bool collided, uint bin, inout Rng rng)
{
    if (!collided)
        return;
    uint3 cells = bench.probe_cells.xyz;
    float3 start = (pos - bench.probe_origin.xyz) / bench.probe_spacing.xyz;
    float3 lattice_dir = dir / bench.probe_spacing.xyz;
    float3 inverse = safe_inverse(lattice_dir);
    bool3 moving = abs(lattice_dir) >= 1e-8;
    float3 bounds_a = -start * inverse;
    float3 bounds_b = (float3(cells) - start) * inverse;
    float3 near = select(moving, min(bounds_a, bounds_b), -FLOAT_MAX);
    float3 far = select(moving, max(bounds_a, bounds_b), FLOAT_MAX);
    float t = max(0.0, max(max(near.x, near.y), near.z));
    float end = min(length, min(min(far.x, far.y), far.z));
    // Along the path, a hair past t picks the cell the path goes on into.
    float nudge = 1e-4 * min(bench.probe_spacing.x, min(bench.probe_spacing.y, bench.probe_spacing.z));
    while (t < end)
    {
        float3 here = start + t * lattice_dir;
        uint3 cell = uint3(clamp(floor(here + nudge * lattice_dir), 0.0, float3(cells - 1)));
        uint leaf = bench_probe_cells[cell.x + cells.x * (cell.y + cells.y * cell.z)];
        uint level = leaf == NO_LEAF ? 0 : leaf >> LEAF_SLOT_BITS;
        float size = float(1u << level);
        float3 low = float3((cell >> level) << level);
        float3 exits = select(moving, (select(lattice_dir > 0.0, low + size, low) - here) * inverse, FLOAT_MAX);
        float next = min(end, max(t + min(min(exits.x, exits.y), exits.z), t + nudge));
        if (leaf != NO_LEAF)
        {
            float chord = next - t;
            float3 a = (here - low) / size;
            float3 b = (start + next * lattice_dir - low) / size;
            float at_a[8] = corner_weights(a);
            float at_middle[8] = corner_weights(0.5 * (a + b));
            float at_b[8] = corner_weights(b);
            float pick = rng.next4().x * 6.0;
            uint corner = 7;
            for (uint c = 0; c < 7; c++)
            {
                pick -= at_a[c] + 4.0 * at_middle[c] + at_b[c];
                if (pick < 0.0)
                {
                    corner = c;
                    break;
                }
            }
            uint slot = leaf & ((1u << LEAF_SLOT_BITS) - 1);
            uint probe = bench_probe_corners[slot * 8 + corner];
            add_sum(bench_probe_sun_bins_raw, bin * bench.probe_counts.x + probe, weight * chord / bench_probe_volumes[probe] * float4(1.0, dir));
        }
        t = next;
    }
}

// The adjoint of `bake_probes`: light enters the grid's box from a uniform direction,
// at a uniform point of the box's area seen from there, and `deposit_path`, with
// CLOUD_BENCH_LEAF_LENGTHS, or `deposit_points` adds its path past its first collision
// to the sun's bin of its entry direction; `light_probes` takes the sky's scattered light
// from the bins, and `bake_sky_transfer` its single scattering. A round adds as much as
// one walk per probe.
void trace_photon(uint photon, uint round)
{
    uint photon_count = bench.probe_counts.x;
    BenchVolume volume = BenchVolume();
    Rng rng = Rng(uint2(photon, 1), round);
    float3 from = sample_sphere_uniform(rng.next4().xy);
    float3 dir = -from;
    float3 extent = volume.grid_extent;
    float3 areas = abs(dir) * extent.yzx * extent.zxy;
    float area = areas.x + areas.y + areas.z;
    float4 u = rng.next4();
    uint axis = u.x * area < areas.x ? 0 : (u.x * area < areas.x + areas.y ? 1 : 2);
    float3 pos = volume.grid_min + u.yzw * extent;
    pos[axis] = volume.grid_min[axis] + (dir[axis] > 0.0 ? 0.0 : extent[axis]);
    uint bin = sun_bin(from);
    float weight = area / float(photon_count);
    bool leaf_lengths = (bench.options.z & CLOUD_BENCH_LEAF_LENGTHS) != 0;
    float next = rng.next4().x * min(bench.probe_spacing.x, min(bench.probe_spacing.y, bench.probe_spacing.z));
    bool collided = false;
    for (uint bounce = 0; bounce <= bench.options.y; bounce++)
    {
        float t = sample_collision(volume, pos, dir, FLOAT_MAX, rng);
        if (leaf_lengths)
            deposit_path(pos, dir, t >= 0.0 ? t : FLOAT_MAX, weight, collided, bin, rng);
        else
            deposit_points(pos, dir, t >= 0.0 ? t : grid_span(volume, pos, dir, FLOAT_MAX).y, next, weight, collided, bin);
        if (t < 0.0)
            break;
        pos += t * dir;
        collided = true;
        dir = sample_cloud_phase(rng.next4().xyz, dir, true);
    }
    flush_splat(bin);
}

// A dispatch traces `bench.samples.w` rounds from `bench.samples.z` on. Each thread takes
// their photons one after another from the counter, so that a warp stays full while its
// longest paths finish, and the rounds' tails overlap.
RWByteAddressBuffer bench_photon_counter;

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void trace_photons(uint3 thread: SV_DispatchThreadID)
{
    uint photon_count = bench.probe_counts.x;
    for (;;)
    {
        uint index;
        bench_photon_counter.InterlockedAdd(0, 1, index);
        if (index >= photon_count * bench.samples.w)
            return;
        trace_photon(index % photon_count, bench.samples.z + index / photon_count);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void reset_photon_counter()
{
    bench_photon_counter.Store(0, 0);
}

// The sky's single scattering transfer, estimated directly: per round, the
// transmittance from the probe along one direction, which is what the sky's radiance
// there reaches the probe by, into the transfer's harmonics, 4 pi times for the
// sphere. The directions are a Fibonacci set of SKY_RAY_SET, taken in a scrambled
// order and rotated at random per probe and set, so that the rounds stratify the
// sphere while any number of them stays unbiased.
static const uint SKY_RAY_SET = 1024;
static const uint SKY_RAY_STRIDE = 571;

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void bake_sky_transfer(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    BenchVolume volume = BenchVolume();
    float3 pos = bench_probe_positions[probe].xyz;
    float4 sums[SH3_COUNT];
    for (uint j = 0; j < SH3_COUNT; j++)
        sums[j] = bench_probe_transfer[probe * SH3_COUNT + j];
    for (uint round = bench.samples.z; round < bench.samples.z + bench.samples.w; round++)
    {
        Rng rng = Rng(uint2(probe, 2), round);
        // A uniform rotation (Shoemake) of the set.
        Rng set_rng = Rng(uint2(probe, 3), round / SKY_RAY_SET);
        float4 u = set_rng.next4();
        float4 q = float4(sqrt(1.0 - u.x) * sin(2.0 * PI * u.y), sqrt(1.0 - u.x) * cos(2.0 * PI * u.y),
                          sqrt(u.x) * sin(2.0 * PI * u.z), sqrt(u.x) * cos(2.0 * PI * u.z));
        uint i = (round % SKY_RAY_SET) * SKY_RAY_STRIDE % SKY_RAY_SET;
        float z = 1.0 - 2.0 * (float(i) + 0.5) / float(SKY_RAY_SET);
        float radius = sqrt(max(1.0 - z * z, 0.0));
        float angle = float(i) * PI * (3.0 - sqrt(5.0));
        float3 point = float3(radius * cos(angle), z, radius * sin(angle));
        float3 look = point + 2.0 * cross(q.xyz, cross(q.xyz, point) + q.w * point);
        float seen = transmittance(volume, pos, look, rng);
        float basis[SH3_COUNT] = sh_basis<3>(look);
        for (uint j = 0; j < SH3_COUNT; j++)
            sums[j] += 4.0 * PI * seen * basis[j] * float4(1.0, -look);
    }
    for (uint j = 0; j < SH3_COUNT; j++)
        bench_probe_transfer[probe * SH3_COUNT + j] = sums[j];
}

// A walk starts at the probe looking along a uniform direction and follows the light
// arriving from it backwards, scattering by the droplets' phase, which is symmetric,
// until it leaves the grid. The sky's light it then meets from its last direction
// reaches the probe, estimated over all directions as 4 pi times that radiance. At
// every collision the walk also looks for a sun it does not know: towards a direction
// drawn from the phase, so that the phase cancels, it adds the transmittance to that
// direction's bin, marched unless CLOUD_BENCH_RATIO_TRACKING, and skips collisions
// that no direction lights. A walk's light hides in its few collisions near a lit surface, and an exact transmittance
// there converges the bins far sooner than ratio tracking's noisy, unbiased one. Walks
// past the bounce limit are dropped, as in the reference.
[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void bake_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    uint probes = bench.probe_counts.x;
    if (bench.samples.z == 0)
    {
        for (uint b = 0; b < SUN_BIN_COUNT; b++)
            bench_probe_sun_bins[b * probes + probe] = 0.0;
    }
    BenchVolume volume = BenchVolume();
    bool ratio_tracking = (bench.options.z & CLOUD_BENCH_RATIO_TRACKING) != 0;
    Rng rng = Rng(uint2(probe, 0), bench.samples.z);
    float3 pos = bench_probe_positions[probe].xyz;
    float3 look = sample_sphere_uniform(rng.next4().xy);
    // The light travels opposite to the look.
    float4 arrival = float4(1.0, -look);
    float3 dir = look;
    float t = sample_collision(volume, pos, dir, FLOAT_MAX, rng);
    for (uint bounce = 0; t >= 0.0 && bounce < bench.options.y; bounce++)
    {
        pos += t * dir;
        // Every walk's scatter follows the approximation's own towards the viewer.
        if (sun_reach(volume, pos) < SUN_REACH_SKIP)
        {
            float3 towards_sun = sample_cloud_phase(rng.next4().xyz, dir, true);
            float sunlit;
            if (ratio_tracking)
                sunlit = transmittance(volume, pos, towards_sun, rng);
            else
            {
                float sun_depth = optical_depth(volume, pos, towards_sun, volume.voxel, SUN_DEPTH_MAX, 0);
                sunlit = sun_depth > SUN_DEPTH_MAX ? 0.0 : exp(-sun_depth);
            }
            if (sunlit > 0.0)
                bench_probe_sun_bins[sun_bin(towards_sun) * probes + probe] += sunlit * arrival;
        }
        dir = sample_cloud_phase(rng.next4().xyz, dir, true);
        t = sample_collision(volume, pos, dir, FLOAT_MAX, rng);
    }
    float weight = t < 0.0 ? 4.0 * PI : 0.0;
    float basis[SH3_COUNT] = sh_basis<3>(dir);
    for (uint j = 0; j < SH3_COUNT; j++)
    {
        uint element = probe * SH3_COUNT + j;
        float4 sum = bench.samples.z == 0 ? float4(0.0) : bench_probe_transfer[element];
        bench_probe_transfer[element] = sum + weight * basis[j] * arrival;
    }
}

// The harmonics of the sun's log fluence, or with CLOUD_BENCH_SUN_LINEAR the fluence,
// and flux ratio of `probe`'s bins by the bins' quadrature: they are of equal solid
// angle. Index 0 is the fluence's constant harmonic, then its others, then the flux
// ratio's; false if no walk has carried sunlight to the probe.
bool sun_coefficients(uint probe, out float coefficients[SUN_QUANTIZED + 1])
{
    uint probes = bench.probe_counts.x;
    for (uint i = 0; i <= SUN_QUANTIZED; i++)
        coefficients[i] = 0.0;
    float brightest = 0.0;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
        brightest = max(brightest, bench_probe_sun_bins[b * probes + probe].x);
    if (!(brightest > 0.0))
        return false;
    // Each walk's deposits in a bin estimate the bin's integral of phi / 4 pi.
    float to_fluence = float(SUN_BIN_COUNT) / float(max(bench.samples.z, 1));
    float bin_solid_angle = 4.0 * PI / float(SUN_BIN_COUNT);
    bool linear = (bench.options.z & CLOUD_BENCH_SUN_LINEAR) != 0;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
    {
        float4 sum = bench_probe_sun_bins[b * probes + probe];
        float basis[SH5_COUNT] = sh_basis<5>(bin_direction(b, 0.5));
        float fluence = max(sum.x, SUN_FLOOR * brightest) * to_fluence;
        float fitted = (linear ? fluence : log(fluence)) * bin_solid_angle;
        for (uint j = 0; j < sun_harmonics(); j++)
            coefficients[j] += fitted * basis[j];
        float3 flux_ratio = (sum.x > 0.0 ? sum.yzw / sum.x : float3(0.0)) * bin_solid_angle;
        for (uint k = 0; k < SUN_FLUX_HARMONICS; k++)
            for (uint c = 0; c < 3; c++)
                coefficients[SH5_COUNT + 3 * k + c] += flux_ratio[c] * basis[k];
    }
    return true;
}

// Floats as uints that order alike, for the ranges' atomics.
uint orderable(float value)
{
    uint bits = asuint(value);
    return (bits & 0x80000000) != 0 ? ~bits : bits | 0x80000000;
}

float from_orderable(uint bits)
{
    return asfloat((bits & 0x80000000) != 0 ? bits & 0x7FFFFFFF : ~bits);
}

float2 coefficient_range(uint i)
{
    return float2(from_orderable(bench_probe_sun_ranges.Load(4 * i)), from_orderable(bench_probe_sun_ranges.Load(4 * (SUN_QUANTIZED + i))));
}

uint quantise(float value, float2 range)
{
    return range.y > range.x ? uint(round(saturate((value - range.x) / (range.y - range.x)) * 255.0)) : 0;
}

float dequantise(uint byte, float2 range)
{
    return lerp(range.x, range.y, float(byte) / 255.0);
}

// Per harmonic of the sky's radiance, the light at `probe` per walk: the sums, and with
// photons the bins, as a bin's sum is the light from unit radiance over the bin and the
// sky's radiance is smooth across a bin.
void sky_light(uint probe, out float4 light[SH3_COUNT])
{
    uint probes = bench.probe_counts.x;
    float walks = float(max(bench.samples.z, 1));
    for (uint j = 0; j < SH3_COUNT; j++)
        light[j] = bench_probe_transfer[probe * SH3_COUNT + j] / walks;
    if ((bench.options.z & CLOUD_BENCH_PHOTONS) == 0)
        return;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
    {
        float basis[SH3_COUNT] = sh_basis<3>(bin_direction(b, 0.5));
        float4 bin = 4.0 * PI / walks * bench_probe_sun_bins[b * probes + probe];
        for (uint j = 0; j < SH3_COUNT; j++)
            light[j] += basis[j] * bin;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void reset_sun_ranges()
{
    for (uint i = 0; i < SUN_QUANTIZED; i++)
    {
        bench_probe_sun_ranges.Store(4 * i, 0xFFFFFFFF);
        bench_probe_sun_ranges.Store(4 * (SUN_QUANTIZED + i), 0);
    }
}

// The least and greatest of each quantised coefficient over the lit probes, reduced
// over the wave before the atomics.
[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void range_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    float coefficients[SUN_QUANTIZED + 1];
    if (probe >= bench.probe_counts.x || !sun_coefficients(probe, coefficients))
        return;
    for (uint i = 0; i < SUN_QUANTIZED; i++)
    {
        uint bits = orderable(coefficients[i + 1]);
        uint least = WaveActiveMin(bits);
        uint greatest = WaveActiveMax(bits);
        if (WaveIsFirstLane())
        {
            bench_probe_sun_ranges.InterlockedMin(4 * i, least);
            bench_probe_sun_ranges.InterlockedMax(4 * (SUN_QUANTIZED + i), greatest);
        }
    }
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void fit_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    float coefficients[SUN_QUANTIZED + 1];
    bool lit = sun_coefficients(probe, coefficients);
    SunTransfer sun;
    sun.dc = lit ? coefficients[0] : SUN_DARK_LOG;
    for (uint word = 0; word < SUN_PACKED_WORDS; word++)
        sun.packed[word] = 0;
    for (uint i = 0; i < SUN_QUANTIZED && lit; i++)
        pack_byte(sun.packed, i, quantise(coefficients[i + 1], coefficient_range(i)));
    bench_probe_sun[probe] = sun;

    float4 light[SH3_COUNT];
    sky_light(probe, light);
    SkyTransfer sky;
    sky.dc = light[0].x;
    for (uint word = 0; word < SKY_PACKED_WORDS; word++)
        sky.packed[word] = 0;
    for (uint i = 0; i < SKY_QUANTIZED; i++)
        pack_byte(sky.packed, i, quantise(light[(i + 1) / 4][(i + 1) % 4], sky.range(i)));
    bench_probe_sky[probe] = sky;
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void light_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    float3 sun = (bench.options.z & CLOUD_BENCH_SUN) != 0 ? bench_sun.Load(int3(0, 0, 0)).rgb : float3(0.0);
    float4 light[3] = { float4(0.0), float4(0.0), float4(0.0) };
    if ((bench.options.z & CLOUD_BENCH_SKY) != 0)
    {
        SkyTransfer sky = bench_probe_sky[probe];
        uint harmonics = (bench.options.z & CLOUD_BENCH_SKY_ORDER2) != 0 ? SH2_COUNT : SH3_COUNT;
        for (uint j = 0; j < harmonics; j++)
        {
            float3 radiance = bench_sun.Load(int3(1 + j, 0, 0)).rgb;
            float4 light_j = sky.light(j);
            for (uint c = 0; c < 3; c++)
                light[c] += radiance[c] * light_j;
        }
    }

    SunTransfer transfer = bench_probe_sun[probe];
    float4 sunlight = 0.0;
    if (transfer.dc > SUN_DARK_LOG && any(sun > 0.0))
    {
        float sun_basis[SH5_COUNT] = sh_basis<5>(frame.sun_direction.xyz);
        float fitted = transfer.dc * sun_basis[0];
        for (uint j = 1; j < SH5_COUNT; j++)
            fitted += transfer.coefficient(j - 1) * sun_basis[j];
        float3 flux_ratio = 0.0;
        for (uint k = 0; k < SUN_FLUX_HARMONICS; k++)
            for (uint c = 0; c < 3; c++)
                flux_ratio[c] += transfer.coefficient(SH5_COUNT - 1 + 3 * k + c) * sun_basis[k];
        bool linear = (bench.options.z & CLOUD_BENCH_SUN_LINEAR) != 0;
        sunlight = (linear ? max(fitted, 0.0) : exp(fitted)) * float4(1.0, flux_ratio);
    }
    for (uint c = 0; c < 3; c++)
        bench_probe_light_values[probe * 3 + c] = light[c] + sun[c] * sunlight;
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void fill_probe_atlas(uint3 thread: SV_DispatchThreadID)
{
    uint slot = thread.x;
    if (slot >= bench.probe_counts.y)
        return;
    uint3 size;
    bench_probe_light_target.GetDimensions(size.x, size.y, size.z);
    uint channel_depth = size.z / 3;
    uint2 leaves = size.xy / 2;
    uint3 first = 2 * uint3(slot % leaves.x, slot / leaves.x % leaves.y, slot / (leaves.x * leaves.y));
    for (uint corner = 0; corner < 8; corner++)
    {
        uint3 texel = first + uint3(corner & 1, (corner >> 1) & 1, corner >> 2);
        uint probe = bench_probe_corners[slot * 8 + corner];
        for (uint c = 0; c < 3; c++)
            bench_probe_light_target[uint3(texel.xy, c * channel_depth + texel.z)] = bench_probe_light_values[probe * 3 + c];
    }
}
