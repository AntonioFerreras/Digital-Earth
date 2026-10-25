// The ray marcher's cloud shadow map (earth/cloud_shadow.slang) for the sun of
// `frame`, one thread per line.

import earth.cloud_shadow;
import earth.cloud_shell;
import shared.bindings;
import shared.uniforms;

[format("r16f")]
RWTexture2DArray<float> cloud_shadow_target;

// Midpoint samples per slice of each line.
static const uint STEPS_PER_SLICE = 16;

[shader("compute")]
[numthreads(MODE_GROUP_SIZE, MODE_GROUP_SIZE, 1)]
void bake_cloud_shadow(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= uint2(CLOUD_SHADOW_WIDTH, CLOUD_SHADOW_HEIGHT)))
        return;
    float transmittance[CLOUD_SHADOW_SLICES];
    bake_cloud_shadow_line(thread.xy, SunFrame(frame.sun_direction.xyz), CloudMedium(scene), STEPS_PER_SLICE, transmittance);
    for (uint k = 0; k < CLOUD_SHADOW_SLICES; k++)
        cloud_shadow_target[uint3(thread.xy, k)] = transmittance[k];
}
