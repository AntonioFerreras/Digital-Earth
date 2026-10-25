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
// fits the sun's transfer from the walks' direction bins, `light_probes` lights the
// probes with the present sun and sky, and `fill_probe_atlas` copies their light to
// their leaves' corners for the approximation (modes/cloud_bench.slang).

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
// Element probe * SH_COUNT + j: the probe's sums over its walks of the sky's transfer
// for harmonic j, x: phi_j, yzw: F_j.
RWStructuredBuffer<float4> bench_probe_transfer;
// Element probe * SUN_BIN_COUNT + b: the probe's sums over its walks of the sun's light
// from directions in bin b of the equal-area octahedral map, x: phi, yzw: F.
RWStructuredBuffer<float4> bench_probe_sun_bins;
// The two sums above as raw bytes, for the photons' atomic adds.
RWByteAddressBuffer bench_probe_transfer_raw;
RWByteAddressBuffer bench_probe_sun_bins_raw;
RWStructuredBuffer<SunTransfer> bench_probe_sun;
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

// Per unit irradiance from the sun's direction s: phi(s) = exp(sum_j log_fluence[j]
// Y_j(s)) through order 2, or order 3 with CLOUD_BENCH_SUN_ORDER3, and
// F(s) = phi(s) sum_k flux_ratio[k] Y_k(s) over the first four harmonics. Scalar arrays
// only, so that the buffer packs it without padding.
struct SunTransfer
{
    float log_fluence[SH3_COUNT];
    float flux_ratio[3 * SUN_FLUX_HARMONICS];

    [mutating]
    void add_flux_ratio(uint k, float3 value)
    {
        for (uint i = 0; i < 3; i++)
            flux_ratio[3 * k + i] += value[i];
    }

    float3 flux_ratio_at(uint k)
    {
        return float3(flux_ratio[3 * k], flux_ratio[3 * k + 1], flux_ratio[3 * k + 2]);
    }
};

static const uint PROBE_GROUP_SIZE = 64;

// The harmonics of the sun's log fluence.
uint sun_harmonics()
{
    return (bench.options.z & CLOUD_BENCH_SUN_ORDER3) != 0 ? SH3_COUNT : SH_COUNT;
}
static const uint SUN_FLUX_HARMONICS = 4;
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
        bench_probe_sun_bins[probe * SUN_BIN_COUNT + b] = 0.0;
    for (uint j = 0; j < SH_COUNT; j++)
        bench_probe_transfer[probe * SH_COUNT + j] = 0.0;
}

void add_sum(RWByteAddressBuffer sums, uint element, float4 value)
{
    for (uint k = 0; k < 4; k++)
    {
        float original;
        sums.InterlockedAddF32(16 * element + 4 * k, value[k], original);
    }
}

// Light that has scattered feeds the sun's `bin`, light that has not the sky's transfer.
void deposit(uint probe, float4 value, bool collided, uint bin, float basis[SH_COUNT])
{
    if (collided)
        add_sum(bench_probe_sun_bins_raw, probe * SUN_BIN_COUNT + bin, value);
    else
    {
        for (uint j = 0; j < SH_COUNT; j++)
            add_sum(bench_probe_transfer_raw, probe * SH_COUNT + j, 4.0 * PI * basis[j] * value);
    }
}

// The scattered light of this thread's photon in the leaf it last crossed, per corner,
// not yet added to the probes.
static uint splat_slot = NO_LEAF;
static float4 splat[8] = { float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0), float4(0.0) };

// Adds `weight` times the path from `pos` along `dir` up to `length` to the probes, at
// points a finest cell apart from `next` on, which carries over to the path's next
// part: each point's cell length times its leaf's corners' trilinear weights, over each
// probe's integral of its weight, a kernel estimate of the fluence. Scattered light goes
// to all eight corners, summed in registers until the photon leaves the leaf, across
// collisions, as the noise of drawing one corner would otherwise dominate the bins'.
// Light that has not collided feeds nine harmonics per corner, too many for that, and
// goes to one corner drawn by its weight.
void deposit_points(float3 pos, float3 dir, float length, inout float next, float weight, bool collided, uint bin,
                    float basis[SH_COUNT], inout Rng rng)
{
    uint3 cells = bench.probe_cells.xyz;
    float step = min(bench.probe_spacing.x, min(bench.probe_spacing.y, bench.probe_spacing.z));
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
        if (collided)
        {
            if (slot != splat_slot)
                flush_splat(bin);
            splat_slot = slot;
            float weights[8] = corner_weights(within);
            for (uint corner = 0; corner < 8; corner++)
                splat[corner] += weights[corner] * weight * step * float4(1.0, dir);
            continue;
        }
        float4 pick = rng.next4();
        uint corner = (pick.x < within.x ? 1u : 0u) | (pick.y < within.y ? 2u : 0u) | (pick.z < within.z ? 4u : 0u);
        uint probe = bench_probe_corners[slot * 8 + corner];
        deposit(probe, weight * step / bench_probe_volumes[probe] * float4(1.0, dir), collided, bin, basis);
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
            add_sum(bench_probe_sun_bins_raw, probe * SUN_BIN_COUNT + bin, splat[corner] / bench_probe_volumes[probe]);
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
void deposit_path(float3 pos, float3 dir, float length, float weight, bool collided, uint bin, float basis[SH_COUNT],
                  inout Rng rng)
{
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
            deposit(probe, weight * chord / bench_probe_volumes[probe] * float4(1.0, dir), collided, bin, basis);
        }
        t = next;
    }
}

// The adjoint of `bake_probes`: light enters the grid's box from a uniform direction,
// at a uniform point of the box's area seen from there, and `deposit_path`, with
// CLOUD_BENCH_LEAF_LENGTHS, or `deposit_points` adds its path to the probes. Light past
// its first collision feeds the sun's bin of its entry direction, and before it the
// sky's transfer; `light_probes` takes the sky's scattered light from the bins. A
// round adds as much as one walk per probe.
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
    float basis[SH_COUNT] = sh_basis(from);
    float weight = area / float(photon_count);
    bool leaf_lengths = (bench.options.z & CLOUD_BENCH_LEAF_LENGTHS) != 0;
    float next = rng.next4().x * min(bench.probe_spacing.x, min(bench.probe_spacing.y, bench.probe_spacing.z));
    bool collided = false;
    for (uint bounce = 0; bounce <= bench.options.y; bounce++)
    {
        float t = sample_collision(volume, pos, dir, FLOAT_MAX, rng);
        if (leaf_lengths)
            deposit_path(pos, dir, t >= 0.0 ? t : FLOAT_MAX, weight, collided, bin, basis, rng);
        else
            deposit_points(pos, dir, t >= 0.0 ? t : grid_span(volume, pos, dir, FLOAT_MAX).y, next, weight, collided, bin, basis, rng);
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
    uint first_bin = probe * SUN_BIN_COUNT;
    if (bench.samples.z == 0)
    {
        for (uint b = 0; b < SUN_BIN_COUNT; b++)
            bench_probe_sun_bins[first_bin + b] = 0.0;
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
                bench_probe_sun_bins[first_bin + sun_bin(towards_sun)] += sunlit * arrival;
        }
        dir = sample_cloud_phase(rng.next4().xyz, dir, true);
        t = sample_collision(volume, pos, dir, FLOAT_MAX, rng);
    }
    float weight = t < 0.0 ? 4.0 * PI : 0.0;
    float basis[SH_COUNT] = sh_basis(dir);
    for (uint j = 0; j < SH_COUNT; j++)
    {
        uint element = probe * SH_COUNT + j;
        float4 sum = bench.samples.z == 0 ? float4(0.0) : bench_probe_transfer[element];
        bench_probe_transfer[element] = sum + weight * basis[j] * arrival;
    }
}

// Projects the bins' log fluence and flux ratio onto the harmonics by the bins'
// quadrature: they are of equal solid angle.
[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void fit_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    uint first_bin = probe * SUN_BIN_COUNT;
    float brightest = 0.0;
    for (uint b = 0; b < SUN_BIN_COUNT; b++)
        brightest = max(brightest, bench_probe_sun_bins[first_bin + b].x);
    SunTransfer fit;
    for (uint j = 0; j < SH3_COUNT; j++)
        fit.log_fluence[j] = 0.0;
    for (uint i = 0; i < 3 * SUN_FLUX_HARMONICS; i++)
        fit.flux_ratio[i] = 0.0;
    if (brightest > 0.0)
    {
        // Each walk's deposits in a bin estimate the bin's integral of phi / 4 pi.
        float to_fluence = float(SUN_BIN_COUNT) / float(max(bench.samples.z, 1));
        float bin_solid_angle = 4.0 * PI / float(SUN_BIN_COUNT);
        for (uint b = 0; b < SUN_BIN_COUNT; b++)
        {
            float4 sum = bench_probe_sun_bins[first_bin + b];
            float basis[SH3_COUNT] = sh3_basis(bin_direction(b, 0.5));
            float log_fluence = log(max(sum.x, SUN_FLOOR * brightest) * to_fluence);
            for (uint j = 0; j < sun_harmonics(); j++)
                fit.log_fluence[j] += log_fluence * basis[j] * bin_solid_angle;
            float3 flux_ratio = sum.x > 0.0 ? sum.yzw / sum.x : float3(0.0);
            for (uint k = 0; k < SUN_FLUX_HARMONICS; k++)
                fit.add_flux_ratio(k, flux_ratio * basis[k] * bin_solid_angle);
        }
    }
    else
        fit.log_fluence[0] = SUN_DARK_LOG;
    bench_probe_sun[probe] = fit;
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void light_probes(uint3 thread: SV_DispatchThreadID)
{
    uint probe = thread.x;
    if (probe >= bench.probe_counts.x)
        return;
    float3 sun = bench_sun.Load(int3(0, 0, 0)).rgb;
    bool sky = (bench.options.z & CLOUD_BENCH_SKY) != 0;
    float walks = float(max(bench.samples.z, 1));
    float4 light[3] = { float4(0.0), float4(0.0), float4(0.0) };
    for (uint j = 0; j < SH_COUNT; j++)
    {
        float3 sky_radiance = sky ? bench_sun.Load(int3(1 + j, 0, 0)).rgb / walks : float3(0.0);
        float4 transfer = bench_probe_transfer[probe * SH_COUNT + j];
        for (uint c = 0; c < 3; c++)
            light[c] += sky_radiance[c] * transfer;
    }
    // A bin's sum is the light from unit radiance over the bin, and the sky's radiance
    // is smooth across a bin.
    if (sky && (bench.options.z & CLOUD_BENCH_PHOTONS) != 0)
    {
        for (uint b = 0; b < SUN_BIN_COUNT; b++)
        {
            float basis[SH_COUNT] = sh_basis(bin_direction(b, 0.5));
            float3 radiance = 0.0;
            for (uint j = 0; j < SH_COUNT; j++)
                radiance += bench_sun.Load(int3(1 + j, 0, 0)).rgb * basis[j];
            float4 bin = 4.0 * PI / walks * bench_probe_sun_bins[probe * SUN_BIN_COUNT + b];
            for (uint c = 0; c < 3; c++)
                light[c] += radiance[c] * bin;
        }
    }

    SunTransfer transfer = bench_probe_sun[probe];
    float sun_basis[SH3_COUNT] = sh3_basis(frame.sun_direction.xyz);
    float log_fluence = 0.0;
    for (uint j = 0; j < sun_harmonics(); j++)
        log_fluence += transfer.log_fluence[j] * sun_basis[j];
    float3 flux_ratio = 0.0;
    for (uint k = 0; k < SUN_FLUX_HARMONICS; k++)
        flux_ratio += transfer.flux_ratio_at(k) * sun_basis[k];
    float4 sunlight = exp(log_fluence) * float4(1.0, flux_ratio);
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
