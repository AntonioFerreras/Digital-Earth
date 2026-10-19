// lut4d_sample at given uvwz, and the uvwz decoding at the ends of each range,
// through the golden-test I/O of harness.slang (src/render/test_lut4d.zig drives it).

import atmosphere.bruneton;
import atmosphere.lut4d;
import common.constants;
import tests.harness;

SamplerState linear_clamp;
Texture3D<float4> scattering_lut;
uniform uint4 scattering_lut_shape;

// Input uvwz; output rgb.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void lut4d_lookup(uint3 thread: SV_DispatchThreadID)
{
    Case c = Case(thread.x, 4, 3);
    if (!c.valid)
        return;
    c.store<3>(0, lut4d_sample(scattering_lut, linear_clamp, scattering_lut_shape, c.load<4>(0)));
}

// Input uvwz; output scattering_params_from_uvwz as (altitude, mu, mu_s, nu,
// intersects_ground), then the cosine of the horizon at that radius.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void decode_uvwz(uint3 thread: SV_DispatchThreadID)
{
    Case c = Case(thread.x, 4, 6);
    if (!c.valid)
        return;
    ScatteringParams p = scattering_params_from_uvwz(c.load<4>(0));
    float horizon = -sqrt(max(p.r * p.r - PLANET_RADIUS * PLANET_RADIUS, 0.0)) / p.r;
    c.store<3>(0, float3(p.r - PLANET_RADIUS, p.mu, p.mu_s));
    c.store<3>(3, float3(p.nu, p.intersects_ground ? 1.0 : 0.0, horizon));
}
