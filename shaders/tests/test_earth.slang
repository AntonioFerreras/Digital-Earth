import common.constants;
import common.rng;
import common.sampling;
import tests.harness;
import earth.brdf;
import earth.clouds;
import earth.maps;
import earth.material;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sphere_uv(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 2);
    if (!c.valid)
        return;
    c.store(0, sphere_uv(c.load<3>(0)));
}

// Inputs: albedo map rgb, ocean mask.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_grade_albedo(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 3);
    if (!c.valid)
        return;
    c.store(0, grade_albedo(c.load<3>(0), inputs[c.input_base + 3]));
}

// Inputs: albedo, ocean, bathymetry, v, n, l. Outputs: brdf, n_dot_l.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_earth_brdf(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 12, 2);
    if (!c.valid)
        return;
    float3 surface = c.load<3>(0);
    SurfaceResponse response = earth_brdf(surface.x, surface.y, surface.z, c.load<3>(3), c.load<3>(6), c.load<3>(9));
    c.store(0, float2(response.brdf, response.n_dot_l));
}

// Inputs: r, coverage, max_density.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_cloud_density(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 1);
    if (!c.valid)
        return;
    float3 in = c.load<3>(0);
    outputs[c.output_base] = cloud_density(in.x, in.y, in.z);
}

// Inputs: ocean, bathymetry, v, n (normalized here). Outputs: the directional albedo
// (integral of brdf n.l at unit albedo) estimated by sample_earth_bounce and by
// uniform sphere sampling, the bounce pdf integrated over the sphere, and the share
// of bounces reflected below the surface, which the pdf leaves out.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_earth_bounce(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 8, 4);
    if (!c.valid)
        return;
    float2 surface = c.load<2>(0);
    float3 v = normalize(c.load<3>(2));
    float3 n = normalize(c.load<3>(5));
    const uint sample_count = 1 << 20;
    Rng rng = Rng(uint2(id.x, 0), 0);
    float bounce_sum = 0.0;
    float uniform_sum = 0.0;
    float pdf_sum = 0.0;
    uint below_count = 0;
    for (uint i = 0; i < sample_count; i++)
    {
        float4 u = rng.next4();
        BounceSample bounce = sample_earth_bounce(u.xyz, surface.x, surface.y, v, n);
        if (bounce.pdf > 0.0)
        {
            SurfaceResponse response = earth_brdf(1.0, surface.x, surface.y, v, n, bounce.dir);
            bounce_sum += response.brdf * response.n_dot_l / bounce.pdf;
        }
        else
        {
            below_count += 1;
        }
        float3 l = sample_sphere_uniform(rng.next4().xy);
        SurfaceResponse response = earth_brdf(1.0, surface.x, surface.y, v, n, l);
        uniform_sum += response.brdf * response.n_dot_l * 4.0 * PI;
        pdf_sum += earth_bounce_pdf(surface.x, surface.y, v, n, l) * 4.0 * PI;
    }
    c.store(0, float4(bounce_sum, uniform_sum, pdf_sum, float(below_count)) / float(sample_count));
}
