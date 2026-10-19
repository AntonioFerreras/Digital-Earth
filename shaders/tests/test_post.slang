import tests.harness;
import post.crf;
import post.opendrt;
import post.output;

// Inputs: scene-linear rgb, saturation, contrast, tonescale preset.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_opendrt(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 6, 3);
    if (!c.valid)
        return;
    float3 controls = c.load<3>(3);
    c.store(0, opendrt(c.load<3>(0), controls.x, controls.y, uint(controls.z)));
}

// Inputs: display rgb, index, count. Outputs: u (per channel), v.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_crf_coords(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 5, 4);
    if (!c.valid)
        return;
    CrfCoords coords = crf_coords(c.load<3>(0), uint(inputs[c.input_base + 3]), uint(inputs[c.input_base + 4]));
    c.store(0, float4(coords.u, coords.v));
}

// Inputs: pixel x, y, width, height.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_vignette(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 1);
    if (!c.valid)
        return;
    uint4 in = uint4(c.load<4>(0));
    outputs[c.output_base] = vignette(in.xy, in.zw);
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_srgb_oetf(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 3);
    if (!c.valid)
        return;
    c.store(0, srgb_oetf(c.load<3>(0)));
}
