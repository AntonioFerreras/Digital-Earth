import tests.harness;
import earth.tiles;

// Inputs: a direction and the level wanted. Outputs: height, water, albedo rgb, face, u, v.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_tile_lookups(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 8);
    if (!c.valid)
        return;
    float4 input = c.load<4>(0);
    float3 n = normalize(input.xyz);
    uint level = uint(input.w);
    FacePoint p = FacePoint(n);
    c.store(0, float2(tile_height(n, level), tile_water(n, level)));
    c.store(2, tile_albedo(n, level));
    c.store(5, float3(float(p.face), p.uv));
}
