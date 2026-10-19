import tests.harness;
import common.geometry;
import common.rng;
import common.sampling;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_ray_sphere(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 7, 2);
    if (!c.valid)
        return;
    c.store(0, ray_sphere(c.load<3>(0), c.load<3>(3), inputs[c.input_base + 6]));
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_rng(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 4);
    if (!c.valid)
        return;
    uint4 key = uint4(c.load<4>(0));
    Rng rng = Rng(key.xy, key.z);
    rng.dimension = key.w;
    c.store(0, rng.next4());
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sample_cone(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 6, 3);
    if (!c.valid)
        return;
    c.store(0, sample_cone(c.load<2>(0), inputs[c.input_base + 2], c.load<3>(3)));
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sample_hemisphere_cosine(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 5, 3);
    if (!c.valid)
        return;
    c.store(0, sample_hemisphere_cosine(c.load<2>(0), c.load<3>(2)));
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sample_sphere_uniform(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 3);
    if (!c.valid)
        return;
    c.store(0, sample_sphere_uniform(c.load<2>(0)));
}
