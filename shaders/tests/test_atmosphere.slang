import tests.harness;
import atmosphere.bruneton;
import atmosphere.media;
import atmosphere.phase;

// lut: LUT/ozone_profile.dat as f32.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_media_density(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 3);
    if (!c.valid)
        return;
    float3 in = c.load<3>(0);
    c.store(0, media_density(lut, in.x, in.y, in.z));
}

// lut: LUT/ozone_cross_section.dat as f32.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_media_extinction(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 3);
    if (!c.valid)
        return;
    c.store(0, media_extinction(lut, inputs[c.input_base], inputs[c.input_base + 1]));
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_step_integral(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 1, 1);
    if (!c.valid)
        return;
    outputs[c.output_base] = step_integral(inputs[c.input_base]);
}

// Outputs: Rayleigh, HG(g), Klein-Nishina(3000), Draine(g, alpha), cloud, cloud with reduced peak.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_phase(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 6);
    if (!c.valid)
        return;
    float3 in = c.load<3>(0);
    float cos_theta = in.x;
    c.store(0, float3(rayleigh_phase(cos_theta), hg_phase(cos_theta, in.y), klein_nishina_phase(cos_theta, MIE_KLEIN_NISHINA_E)));
    c.store(3, float3(draine_phase(cos_theta, in.y, in.z), cloud_phase(cos_theta, false), cloud_phase(cos_theta, true)));
}

// Outputs: HG(0.8), Klein-Nishina(3000), cloud, cloud with reduced peak directions.
// HG and Klein-Nishina use u.xy; the cloud lobes use u.xyz.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sample_phase(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 6, 12);
    if (!c.valid)
        return;
    float3 u = c.load<3>(0);
    float3 view = c.load<3>(3);
    c.store(0, sample_hg_phase(u.xy, view, MIE_HG_G));
    c.store(3, sample_klein_nishina_phase(u.xy, view, MIE_KLEIN_NISHINA_E));
    c.store(6, sample_cloud_phase(u, view, false));
    c.store(9, sample_cloud_phase(u, view, true));
}

// Outputs: uvwz, intersects_ground (0 or 1).
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_scattering_uvwz(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 5);
    if (!c.valid)
        return;
    float4 in = c.load<4>(0);
    ScatteringParams p;
    p.r = in.x;
    p.mu = in.y;
    p.mu_s = in.z;
    p.nu = in.w;
    p.intersects_ground = ray_intersects_ground(p.r, p.mu);
    c.store(0, scattering_uvwz(p));
    outputs[c.output_base + 4] = p.intersects_ground ? 1.0 : 0.0;
}

// Outputs: r, mu, mu_s, nu, intersects_ground (0 or 1).
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_scattering_params_from_uvwz(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 5);
    if (!c.valid)
        return;
    ScatteringParams p = scattering_params_from_uvwz(c.load<4>(0));
    c.store(0, float4(p.r, p.mu, p.mu_s, p.nu));
    outputs[c.output_base + 4] = p.intersects_ground ? 1.0 : 0.0;
}

// Inputs: pos, dir, sun. Outputs: r, mu, mu_s, nu, intersects_ground (0 or 1).
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_scattering_params_from_ray(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 9, 5);
    if (!c.valid)
        return;
    ScatteringParams p = scattering_params_from_ray(c.load<3>(0), c.load<3>(3), c.load<3>(6));
    c.store(0, float4(p.r, p.mu, p.mu_s, p.nu));
    outputs[c.output_base + 4] = p.intersects_ground ? 1.0 : 0.0;
}

// Outputs: pos, dir, sun.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_canonical_ray(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 9);
    if (!c.valid)
        return;
    float4 in = c.load<4>(0);
    CanonicalRay ray = canonical_ray(in.x, in.y, in.z, in.w);
    c.store(0, ray.pos);
    c.store(3, ray.dir);
    c.store(6, ray.sun);
}

// Inputs: mu, r on a 64 x 64 LUT. Outputs: uv.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_transmittance_uv(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 2);
    if (!c.valid)
        return;
    c.store(0, transmittance_uv(inputs[c.input_base], inputs[c.input_base + 1], float2(64.0)));
}

// Inputs: uv on a 64 x 64 LUT. Outputs: mu, r.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_transmittance_mu_r(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 2);
    if (!c.valid)
        return;
    c.store(0, transmittance_mu_r(c.load<2>(0), float2(64.0)));
}
