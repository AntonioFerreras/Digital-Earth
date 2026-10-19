// The neural mode's network evaluation on given uvwz, and the shader's erf,
// through the golden-test I/O of harness.slang. src/render/test_goldens.zig
// specializes test_network per network as the neural mode does.

import neural.encoding;
import neural.fourier;
import neural.hashgrid;
import neural.kplanes;
import neural.mlp;
import neural.network;
import tests.harness;

// Input uvwz; output linear Rec.709. Every thread joins its group's forward
// pass, so the threads past the last case run it on zeros.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_network<E : IEncoding, N : INetwork>(uint3 id: SV_DispatchThreadID, uint group_thread: SV_GroupIndex)
{
    static_assert(GROUP_SIZE == MLP_GROUP_THREADS, "networks stage weights with the whole group");
    Case c = Case(id.x, 4, 3);
    float4 uvwz = float4(0.0);
    if (c.valid)
        uvwz = c.load<4>(0);
    float3 rgb = network_radiance<E, N>(uvwz, group_thread);
    if (c.valid)
        c.store(0, rgb);
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_erf(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 1, 1);
    if (!c.valid)
        return;
    outputs[c.output_base] = erf(inputs[c.input_base]);
}
