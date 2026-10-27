// The cloud bench's volume (src/render/CloudBench.zig): a density grid in its own
// frame, cloud units, with a majorant per cell of `bench.grid_size.w` voxels per edge
// that bounds the filtered density anywhere in the cell.
//
// Importing this module declares the bench's globals next to those of
// shared/bindings.slang.

module bench_volume;

import atmosphere.phase;
import common.constants;
import common.rng;
import common.sampling;
import shared.bindings;
import shared.uniforms;

public ConstantBuffer<CloudBenchUniforms> bench;
public Texture3D<float> bench_density;
public Texture3D<float> bench_majorant;
// Of cells 2 and 4 times as large per axis, for walks of level 1 and 2.
public Texture3D<float> bench_majorant_2;
public Texture3D<float> bench_majorant_4;
// Direction components below it count as zero: a ray that crosses the grid moves
// far less than a voxel along them, and their crossing distances along the walk would
// exceed what floats resolve, so that the walk never finishes.
static const float PARALLEL_COMPONENT = 1e-8;

bool3 crosses(float3 dir)
{
    return abs(dir) >= PARALLEL_COMPONENT;
}

// 1 / dir, finite: shader float division by zero need not give infinity, and axis
// rays (the light grid's up and down, a sun on the horizon) have zero components.
public float3 safe_inverse(float3 dir)
{
    return 1.0 / select(crosses(dir), dir, float3(1e-30));
}

public struct BenchVolume
{
    public float3 grid_min;
    public float3 grid_extent;
    public float voxel;
    public uint3 cells;
    // Of a majorant cell.
    public float cell_length;
    public float extinction;

    public __init()
    {
        voxel = bench.grid_min.w;
        grid_min = bench.grid_min.xyz;
        grid_extent = float3(bench.grid_size.xyz) * voxel;
        cells = (bench.grid_size.xyz + bench.grid_size.w - 1) / bench.grid_size.w;
        cell_length = float(bench.grid_size.w) * voxel;
        extinction = bench.medium.x * bench.origin.w;
    }

    // The grid's centre in planet-centred metres, where the sky and the sun are taken.
    public float3 world_centre()
    {
        return bench.origin.xyz + bench.origin.w * (grid_min + 0.5 * grid_extent);
    }

    // `world` in planet-centred metres.
    public float3 to_cloud(float3 world)
    {
        return (world - bench.origin.xyz) / bench.origin.w;
    }

    // Texture coordinates of the grid, and of any grid covering the same box.
    public float3 uvw(float3 pos)
    {
        return (pos - grid_min) / grid_extent;
    }

    public float extinction_at(float3 pos)
    {
        return extinction * bench_density.SampleLevel(linear_clamp, uvw(pos), 0.0);
    }

    // The tricubic B-spline's eight filtered fetches. Its support reaches two voxels,
    // so it may exceed the majorant slightly, which only paces a march.
    public float extinction_cubic(float3 pos)
    {
        float3 size = float3(bench.grid_size.xyz);
        CubicTaps<3> taps = CubicTaps<3>(uvw(pos) * size - 0.5, size);
        float density = 0.0;
        for (uint lobe = 0; lobe < 8; lobe++)
        {
            bool3 up = bool3((lobe & 1) != 0, (lobe & 2) != 0, (lobe & 4) != 0);
            float3 weight = select(up, taps.g1, taps.g0);
            density += weight.x * weight.y * weight.z * bench_density.SampleLevel(linear_clamp, select(up, taps.upper, taps.lower), 0.0);
        }
        return extinction * density;
    }

    // Where the ray enters and leaves the grid's box; entry >= exit for a miss.
    public float2 box_range(float3 pos, float3 dir)
    {
        float3 inverse = safe_inverse(dir);
        float3 t0 = (grid_min - pos) * inverse;
        float3 t1 = (grid_min + grid_extent - pos) * inverse;
        float3 near = min(t0, t1);
        float3 far = max(t0, t1);
        return float2(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
    }
};

// The sky's table (modes/cloud_bench_sky.slang): radiance by direction about +Y, the
// vertical at the cloud, in Hillaire's latitude mapping, which spends texels near the
// horizon.
public static const uint2 BENCH_SKY_SIZE = uint2(256, 128);

public float2 sky_uv(float3 dir)
{
    float elevation = asin(clamp(dir.y, -1.0, 1.0));
    float v = 0.5 + 0.5 * sign(elevation) * sqrt(abs(elevation) / (0.5 * PI));
    return float2(atan2(dir.z, dir.x) / (2.0 * PI) + 0.5, v);
}

public float3 sky_direction(float2 uv)
{
    float signed_v = 2.0 * uv.y - 1.0;
    float elevation = sign(signed_v) * signed_v * signed_v * 0.5 * PI;
    float azimuth = (uv.x - 0.5) * 2.0 * PI;
    return float3(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
}

// A ray's walk through the majorant cells (Amanatides and Woo 1987), or at `level`
// through cells 2^level times as large, over [start, end): the current cell spans
// [start, stop) with extinction majorant `majorant`.
public struct MajorantWalk
{
    public float start;
    public float stop;
    public float end;
    public float majorant;
    int3 cell;
    int3 step;
    float3 t_next;
    float3 t_delta;
    int3 cells;
    uint level;

    public __init(BenchVolume volume, float3 pos, float3 dir, float start, float end, uint level)
    {
        this.end = end;
        this.level = level;
        cells = int3((volume.cells + (1u << level) - 1) >> level);
        float cell_length = volume.cell_length * float(1u << level);
        float3 inverse = safe_inverse(dir);
        // Clamped, as rounding may put the entry point just outside the box.
        float3 entry = (pos + start * dir - volume.grid_min) / cell_length;
        cell = clamp(int3(floor(entry)), int3(0), cells - 1);
        step = select(crosses(dir), int3(sign(dir)), int3(0));
        float3 boundary = volume.grid_min + (float3(cell) + float3(step > 0)) * cell_length;
        // An axis the ray runs along never steps.
        t_next = select(crosses(dir), (boundary - pos) * inverse, float3(FLOAT_MAX));
        t_delta = select(crosses(dir), abs(cell_length * inverse), float3(FLOAT_MAX));
        enter(start, volume);
    }

    public bool active()
    {
        return start < end;
    }

    // Steps of at least `step` and at most `step_max` through the current cell, `count`
    // of `length` each, longer where its majorant optical depth per step stays below
    // STEP_MAJORANT_DEPTH times 1 + `depth`, the optical depth already marched, and
    // `scale` times as many. Thin cloud takes few, and so does a march deep into
    // cloud, whose light falls as exp(-depth) so that an error in it counts the less.
    public void steps(float step, float step_max, float depth, float scale, out uint count, out float length)
    {
        float span = stop - start;
        count = uint(ceil(scale * span / clamp(STEP_MAJORANT_DEPTH * (1.0 + depth) / majorant, step, step_max)));
        length = span / float(count);
    }

    // Moves to the next cell along the ray.
    [mutating]
    public void advance(BenchVolume volume)
    {
        uint axis = t_next.x < t_next.y ? (t_next.x < t_next.z ? 0 : 2) : (t_next.y < t_next.z ? 1 : 2);
        cell[axis] += step[axis];
        t_next[axis] += t_delta[axis];
        if (cell[axis] < 0 || cell[axis] >= cells[axis])
        {
            start = end;
            return;
        }
        enter(stop, volume);
    }

    [mutating]
    void enter(float t, BenchVolume volume)
    {
        start = t;
        stop = min(min(min(t_next.x, t_next.y), t_next.z), end);
        float bound = level == 0 ? bench_majorant.Load(int4(cell, 0))
                    : level == 1 ? bench_majorant_2.Load(int4(cell, 0))
                                 : bench_majorant_4.Load(int4(cell, 0));
        majorant = volume.extinction * bound;
    }
};

static const float STEP_MAJORANT_DEPTH = 0.25;

// The ray's part inside the grid's box before `t_max`; empty when start >= end.
public float2 grid_span(BenchVolume volume, float3 pos, float3 dir, float t_max)
{
    float2 range = volume.box_range(pos, dir);
    return float2(max(range.x, 0.0), min(range.y, t_max));
}

// The distance to the ray's first real collision before `t_max` by delta tracking,
// or a negative value for none.
public float sample_collision(BenchVolume volume, float3 pos, float3 dir, float t_max, inout Rng rng)
{
    float2 span = grid_span(volume, pos, dir, t_max);
    if (!(span.x < span.y))
        return -1.0;
    for (MajorantWalk walk = MajorantWalk(volume, pos, dir, span.x, span.y, 0); walk.active(); walk.advance(volume))
    {
        if (!(walk.majorant > 0.0))
            continue;
        float t = walk.start;
        while (true)
        {
            float4 u = rng.next4();
            t -= log(1.0 - u.x) / walk.majorant;
            if (t >= walk.stop)
                break;
            if (u.y * walk.majorant < volume.extinction_at(pos + t * dir))
                return t;
        }
    }
    return -1.0;
}

// Below it ratio tracking plays Russian roulette, so that deep rays stop early.
static const float ROULETTE_TRANSMITTANCE = 0.1;

// An unbiased estimate of the transmittance through the grid along the whole ray,
// by ratio tracking.
public float transmittance(BenchVolume volume, float3 pos, float3 dir, inout Rng rng)
{
    float2 span = grid_span(volume, pos, dir, FLOAT_MAX);
    if (!(span.x < span.y))
        return 1.0;
    float result = 1.0;
    for (MajorantWalk walk = MajorantWalk(volume, pos, dir, span.x, span.y, 0); walk.active(); walk.advance(volume))
    {
        if (!(walk.majorant > 0.0))
            continue;
        float t = walk.start;
        while (true)
        {
            float4 u = rng.next4();
            t -= log(1.0 - u.x) / walk.majorant;
            if (t >= walk.stop)
                break;
            result *= 1.0 - volume.extinction_at(pos + t * dir) / walk.majorant;
            if (result < ROULETTE_TRANSMITTANCE)
            {
                if (u.y * ROULETTE_TRANSMITTANCE >= result)
                    return 0.0;
                result = ROULETTE_TRANSMITTANCE;
            }
        }
    }
    return result;
}

// The optical depth through the grid along the whole ray, by midpoint steps of at
// most `step` in the cells of walk level `level` that hold cloud, or a value past
// `depth_max` once it exceeds it.
public float optical_depth(BenchVolume volume, float3 pos, float3 dir, float step, float depth_max, uint level)
{
    float2 span = grid_span(volume, pos, dir, FLOAT_MAX);
    if (!(span.x < span.y))
        return 0.0;
    float depth = 0.0;
    for (MajorantWalk walk = MajorantWalk(volume, pos, dir, span.x, span.y, level); walk.active() && depth <= depth_max;
         walk.advance(volume))
    {
        if (!(walk.majorant > 0.0))
            continue;
        uint count;
        float dt;
        walk.steps(step, FLOAT_MAX, depth, 1.0, count, dt);
        for (uint i = 0; i < count; i++)
            depth += volume.extinction_at(pos + (walk.start + (float(i) + 0.5) * dt) * dir) * dt;
    }
    return depth;
}
