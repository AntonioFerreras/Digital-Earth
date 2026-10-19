// The decoder behind an encoding: a fully connected network whose shape is a
// compile-time constant, so loops unroll and activations stay in registers.
// IN inputs, HIDDEN layers of WIDTH neurons, 3 ReLU outputs (linear Rec.709);
// HIDDEN = 0 is a linear head.
//
// Weights, in float4 rows: per layer, OUT rows of ceil(in / 4) float4 holding
// the row-major [out][in] matrix zero-padded to whole float4s, then
// ceil(OUT / 4) float4 of bias, zero-padded.

module mlp;

public static const uint MLP_OUTPUTS = 3;
// Threads per group of every caller: all of them call `forward` together, from
// uniform control flow, because staging weight sources load cooperatively.
public static const uint MLP_GROUP_THREADS = 64;
// Layers this wide evaluate four outputs per loop iteration instead of fully
// unrolled: a 64-wide network unrolled spills registers and runs slower.
static const int ROLLED_WIDTH_MIN = 64;

public interface IActivation
{
    static float apply(float x);
}

public struct Identity : IActivation
{
    public static float apply(float x) { return x; }
}

public struct Relu : IActivation
{
    public static float apply(float x) { return max(x, 0.0); }
}

// erf with absolute error below 2e-7 in f32: x * P(x^2) below 1 and
// 1 - exp(Q(|x|)) above, with P and Q weighted minimax fits of erf(x) / x on
// [0, 1] and log(erfc(t)) on [1, 4]. erfc(4) is below half an ulp of 1.
public float erf(float x)
{
    float s = x * x;
    float p = -0.000566019968;
    p = p * s + 0.00492480164;
    p = p * s - 0.0267177187;
    p = p * s + 0.112804122;
    p = p * s - 0.376123548;
    p = p * s + 1.12837911;

    float t = min(abs(x), 4.0);
    float q = 0.00018557407;
    q = q * t - 0.00293960795;
    q = q * t + 0.0218053199;
    q = q * t - 0.103037372;
    q = q * t - 0.638225675;
    q = q * t - 1.12704003;
    q = q * t - 0.000353718759;

    // Both branches run: neurons of one warp straddle the split.
    return abs(x) < 1.0 ? x * p : sign(x) * (1.0 - exp(q));
}

// Exact GELU, x (1 + erf(x / sqrt(2))) / 2.
public struct Gelu : IActivation
{
    public static float apply(float x) { return 0.5 * x * (1.0 + erf(x * 0.707106781)); }
}

// Where a layer's float4 rows are read from. Every thread of the group calls
// `stage` from uniform control flow before reading the layer.
public interface IWeightSource
{
    static void stage<let ROWS : int>(uint first, uint group_thread);
    static float4 row(uint first, uint index);
}

public StructuredBuffer<float4> mlp_rows;

// Each layer copied into groupshared memory by the whole group; CAPACITY is the
// largest layer's row count, which Slang cannot derive in a generic argument.
public struct GroupStaged<let CAPACITY : int> : IWeightSource
{
    static groupshared float4 staged[CAPACITY];

    public static void stage<let ROWS : int>(uint first, uint group_thread)
    {
        static_assert(ROWS <= CAPACITY, "layer larger than the staging capacity");
        GroupMemoryBarrierWithGroupSync();
        for (uint k = group_thread; k < ROWS; k += MLP_GROUP_THREADS)
            staged[k] = mlp_rows[first + k];
        GroupMemoryBarrierWithGroupSync();
    }

    public static float4 row(uint first, uint index) { return staged[index]; }
}

// At most 64 KiB, NVIDIA's uniform buffer range, indexed at compile-time
// offsets: the compiler may read the weights as constant operands of the FMAs,
// with no loads and no barriers. Only tiny networks fit the constant cache.
public static const uint MLP_CONSTANT_ROWS = 4096;
public struct MlpConstantRows
{
    public float4 rows[MLP_CONSTANT_ROWS];
};
public ConstantBuffer<MlpConstantRows> mlp_constants;

public struct Constant : IWeightSource
{
    public static void stage<let ROWS : int>(uint first, uint group_thread) {}
    public static float4 row(uint first, uint index) { return mlp_constants.rows[first + index]; }
}

// `x` through one layer whose rows start at `first`, which advances past it.
void dense<let IN : int, let OUT : int, A : IActivation, W : IWeightSource>(
    float x[IN], out float y[OUT], inout uint first, uint group_thread)
{
    static const int IN4 = (IN + 3) / 4;
    static const int ROWS = OUT * IN4 + (OUT + 3) / 4;
    W.stage<ROWS>(first, group_thread);
    float4 x4[IN4];
    [ForceUnroll]
    for (int i = 0; i < 4 * IN4; i++)
        x4[i / 4][i % 4] = i < IN ? x[i] : 0.0;
    if (OUT >= ROLLED_WIDTH_MIN)
    {
        static_assert(OUT % 4 == 0, "rolled layers evaluate whole float4s of outputs");
        // Four independent sums share each input vector and keep the FMA pipes busy.
        [loop]
        for (uint o = 0; o < OUT; o += 4)
        {
            float4 sum = W.row(first, OUT * IN4 + o / 4);
            [ForceUnroll]
            for (int j = 0; j < IN4; j++)
            {
                [ForceUnroll]
                for (uint r = 0; r < 4; r++)
                    sum[r] += dot(W.row(first, (o + r) * IN4 + j), x4[j]);
            }
            [ForceUnroll]
            for (uint r = 0; r < 4; r++)
                y[o + r] = A.apply(sum[r]);
        }
    }
    else
    {
        [ForceUnroll]
        for (int o = 0; o < OUT; o++)
        {
            float sum = W.row(first, OUT * IN4 + o / 4)[o % 4];
            [ForceUnroll]
            for (int j = 0; j < IN4; j++)
                sum += dot(W.row(first, o * IN4 + j), x4[j]);
            y[o] = A.apply(sum);
        }
    }
    first += ROWS;
}

// The network behind an encoding. All threads of the group call `forward`,
// from uniform control flow.
public interface INetwork
{
    static float3 forward<let IN : int>(float features[IN], uint group_thread);
}

// The first hidden layer applies AFirst, later hidden layers A, the output ReLU.
public struct Mlp<let WIDTH : int, let HIDDEN : int, AFirst : IActivation, A : IActivation, W : IWeightSource> : INetwork
{
    public static float3 forward<let IN : int>(float features[IN], uint group_thread)
    {
        uint first = 0;
        float rgb[MLP_OUTPUTS];
        if (HIDDEN == 0)
        {
            dense<IN, MLP_OUTPUTS, Relu, W>(features, rgb, first, group_thread);
        }
        else
        {
            float h[WIDTH];
            dense<IN, WIDTH, AFirst, W>(features, h, first, group_thread);
            [ForceUnroll]
            for (int layer = 1; layer < HIDDEN; layer++)
            {
                float next[WIDTH];
                dense<WIDTH, WIDTH, A, W>(h, next, first, group_thread);
                h = next;
            }
            dense<WIDTH, MLP_OUTPUTS, Relu, W>(h, rgb, first, group_thread);
        }
        return float3(rgb[0], rgb[1], rgb[2]);
    }
}
