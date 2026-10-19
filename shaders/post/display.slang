// Accumulated scene-linear samples to the rgba8 display image: average,
// vignette, exposure, OpenDRT, camera response, gamma, and the sRGB OETF.
// This is the one place where rows flip: `accum` counts rows from the bottom
// (shared/bindings.slang), `image` from the top as presentation and PNGs expect.

import post.crf;
import post.opendrt;
import post.output;
import shared.uniforms;
import spectral.cie;

public static const uint DISPLAY_GROUP_SIZE = 8;

ConstantBuffer<PostUniforms> post;
Texture2D<float4> accum;
Texture2D<float4> camera_responses;
SamplerState linear_clamp;
[format("rgba8")]
RWTexture2D<float4> image;

[shader("compute")]
[numthreads(DISPLAY_GROUP_SIZE, DISPLAY_GROUP_SIZE, 1)]
void display_main(uint3 thread: SV_DispatchThreadID)
{
    uint2 size;
    image.GetDimensions(size.x, size.y);
    if (any(thread.xy >= size))
        return;
    uint2 pixel = uint2(thread.x, size.y - 1 - thread.y);

    float samples = float(post.state.x);
    // Keeps the user-facing exposure on the original's scale.
    float exposure = exp2(post.tone.x + ORIGINAL_ESTIMATOR_STOPS);
    float3 linear = accum[pixel].rgb / samples * vignette(pixel, size) * exposure;
    float3 display = opendrt(linear, post.tone.z, post.tone.w, post.state.w);
    float3 response = camera_response(camera_responses, linear_clamp, display, post.state.y, post.state.z);
    image[thread.xy] = float4(srgb_oetf(pow(response, post.tone.y)), 1.0);
}
