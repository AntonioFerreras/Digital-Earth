// A trained network's estimate of the symmetric path tracer, one forward pass
// per camera ray. The host specializes the entry point per network
// architecture: E is the encoding, N the network (src/render/NeuralMode.zig).

import atmosphere.bruneton;
import neural.encoding;
import neural.fourier;
import neural.hashgrid;
import neural.kplanes;
import neural.mlp;
import neural.network;
import shared.bindings;

groupshared uint group_enters;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void neural_main<E : IEncoding, N : INetwork>(uint3 thread: SV_DispatchThreadID, uint group_thread: SV_GroupIndex)
{
    static_assert(MODE_GROUP_SIZE * MODE_GROUP_SIZE == MLP_GROUP_THREADS, "networks stage weights with the whole group");
    PixelSample s = PixelSample(thread.xy);
    float3 dir = s.camera_ray_direction();
    float4 uvwz;
    bool enters = camera_ray_uvwz(frame.camera_position.xyz, dir, frame.sun_direction.xyz, uvwz) && s.valid;

    // The forward pass needs the whole group; groups without an atmosphere ray skip it.
    if (group_thread == 0)
        group_enters = 0;
    GroupMemoryBarrierWithGroupSync();
    if (enters)
        InterlockedOr(group_enters, 1);
    GroupMemoryBarrierWithGroupSync();
    float3 rgb = float3(0.0);
    if (group_enters != 0)
        rgb = network_radiance<E, N>(uvwz, group_thread);
    if (s.valid)
        s.accumulate(enters ? rgb : float3(0.0));
}
