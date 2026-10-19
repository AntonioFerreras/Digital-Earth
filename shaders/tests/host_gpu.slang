// Exercises the host gpu wrappers: FrameUniforms layout check, data and buffer
// bindings, and group counts derived from the thread-group size.

import shared.uniforms;

ConstantBuffer<FrameUniforms> frame;
RWStructuredBuffer<uint> output;

[shader("compute")]
[numthreads(64, 1, 1)]
void echo_frame(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < 4)
    {
        output[thread.x] = frame.frame[thread.x] + thread.x;
    }
}
