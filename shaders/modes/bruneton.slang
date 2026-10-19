// The symmetric path tracer's radiance looked up per camera ray in a dense uvwz
// table baked by `earth datagen --sampling grid`: neural.slang with the network
// replaced by the table.

import atmosphere.bruneton;
import atmosphere.lut4d;
import shared.bindings;

// Layout in atmosphere/lut4d.slang.
Texture3D<float4> scattering_lut;
uniform uint4 scattering_lut_shape;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void bruneton_main(uint3 thread: SV_DispatchThreadID)
{
    PixelSample s = PixelSample(thread.xy);
    if (!s.valid)
        return;
    float3 dir = s.camera_ray_direction();
    float4 uvwz;
    float3 rgb = float3(0.0);
    if (camera_ray_uvwz(frame.camera_position.xyz, dir, frame.sun_direction.xyz, uvwz))
        rgb = lut4d_sample(scattering_lut, linear_clamp, scattering_lut_shape, uvwz);
    s.accumulate(rgb);
}
