// A trained network's estimate of the symmetric scene's linear Rec.709 at uvwz:
// the optional u_mu rotation, the encoding E, and the network N, scaled to the
// renderer's radiance. src/render/GpuNetwork.zig binds the globals.

module network;

import neural.encoding;
import neural.mlp;

public struct NetworkUniforms
{
    // From the network's output to the renderer's radiance.
    public float output_scale;
    // Nonzero: the network learned u_mu rotated by half.
    public uint mu_rotation;
};
public ConstantBuffer<NetworkUniforms> network_uniforms;

// All threads of the group call it, from uniform control flow.
public float3 network_radiance<E : IEncoding, N : INetwork>(float4 uvwz, uint group_thread)
{
    // u_mu = 0.5 separates rays that hit the ground from rays that miss it;
    // rotated, that discontinuity sits on the domain edges, where no
    // interpolation crosses it.
    if (network_uniforms.mu_rotation != 0)
        uvwz.z = frac(uvwz.z + 0.5);
    float features[E.FEATURES];
    E.encode(uvwz, features);
    return network_uniforms.output_scale * N.forward<E.FEATURES>(features, group_thread);
}
