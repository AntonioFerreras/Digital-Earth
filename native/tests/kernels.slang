// Fixture kernels for test_gpu.c.

// Mirrored by struct frame in test_gpu.c.
struct Frame
{
    float4 tint;
    uint4 counts;
    float scale;
    uint offset;
};

uniform Frame frame;
ConstantBuffer<Frame> frame_cb;
RWStructuredBuffer<uint> values;
StructuredBuffer<uint> input_values;
RWStructuredBuffer<float> results;
RWTexture2D<float4> target;
Texture2D<float> source;
SamplerState linear_sampler;
Texture2D<float> maps[2];

[shader("compute")]
[numthreads(64, 1, 1)]
void fill(uint3 id: SV_DispatchThreadID)
{
    values[id.x] = id.x * frame.counts.x + frame.offset;
}

// Reads what the previous dispatch in the same command list wrote.
[shader("compute")]
[numthreads(64, 1, 1)]
void double_values(uint3 id: SV_DispatchThreadID)
{
    values[id.x] = values[id.x] * 2 + frame_cb.offset;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void copy_plus_one(uint3 id: SV_DispatchThreadID)
{
    values[id.x] = input_values[id.x] + 1;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void paint(uint3 id: SV_DispatchThreadID)
{
    target[id.xy] = float4(id.x, id.y, frame.scale, 1) * frame.tint;
}

// Samples halfway between texel centres 0 and 1 of row 0.
[shader("compute")]
[numthreads(1, 1, 1)]
void sample_between(uint3 id: SV_DispatchThreadID)
{
    uint width, height;
    source.GetDimensions(width, height);
    results[0] = source.SampleLevel(linear_sampler, float2(1.0 / width, 0.5 / height), 0);
}

interface IOffset
{
    static uint offset();
}

struct AddSeven : IOffset
{
    static uint offset() { return 7; }
}

// Specialized by the host: test_generic_entry.
[shader("compute")]
[numthreads(64, 1, 1)]
void fill_generic<T : IOffset, let SCALE : uint>(uint3 id: SV_DispatchThreadID)
{
    values[id.x] = id.x * SCALE + T.offset();
}
