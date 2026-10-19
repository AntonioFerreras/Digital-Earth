import tests.harness;
import shared.earth_bindings;

// Inputs: map uv. Outputs: hardware and f32 bilinear samples of `earth_topography`.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_tiled_sampling(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 2);
    if (!c.valid)
        return;
    float2 uv = c.load<2>(0);
    c.store(0, float2(sample_tiled(earth_topography, earth_maps.topography, uv),
                      sample_tiled_exact(earth_topography, earth_maps.topography, uv)));
}
