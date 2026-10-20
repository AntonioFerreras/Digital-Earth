import tests.harness;
import earth.tiles;

// Inputs: a direction, not necessarily unit. Outputs: face, u, v.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_cube_face_uv(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 3);
    if (!c.valid)
        return;
    FacePoint p = FacePoint(normalize(c.load<3>(0)));
    c.store(0, float3(float(p.face), p.uv));
}

// Inputs: face, u, v. Outputs: the unit direction.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_cube_direction(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 3);
    if (!c.valid)
        return;
    float3 input = c.load<3>(0);
    c.store(0, face_direction(uint(input.x), input.yz));
}
