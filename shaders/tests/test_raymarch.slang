// The ray marcher's baked LUTs against direct evaluations, through the
// golden-test I/O of harness.slang (src/render/test_raymarch.zig drives it).

import atmosphere.bruneton;
import atmosphere.media;
import atmosphere.scattering_luts;
import common.constants;
import shared.uniforms;
import spectral.cie;
import tests.harness;

ConstantBuffer<SceneUniforms> scene;
StructuredBuffer<float> ozone_cross_section;
StructuredBuffer<float> ozone_profile;
SamplerState linear_clamp;
Texture2D<float4> column_density_lut;
Texture3D<float4> multiple_scattering_lut;
StructuredBuffer<float> cie;

static const uint REFERENCE_STEP_COUNT = 1 << 16;

// Input (mu, altitude, wavelength); output (LUT lookup, reference). The
// reference is a fine midpoint integral with compensated summation, 0 for
// rays that hit the planet.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void transmittance_reference(uint3 thread: SV_DispatchThreadID)
{
    Case c = Case(thread.x, 3, 2);
    if (!c.valid)
        return;
    float3 input = c.load<3>(0);
    float mu = input.x;
    float r = PLANET_RADIUS + input.y;
    SpectralMedium medium = SpectralMedium(ozone_cross_section, ozone_profile, input.z, scene.media);
    float3 position = float3(0.0, r, 0.0);
    float3 dir = float3(sqrt(1.0 - mu * mu), mu, 0.0);
    float lookup = ColumnDensity(column_density_lut, position, dir).transmittance(medium.extinction);

    float reference = 0.0;
    if (!ray_intersects_ground(r, mu))
    {
        float dt = distance_to_atmosphere_top(r, mu) / float(REFERENCE_STEP_COUNT);
        float3 sum = 0.0;
        float3 compensation = 0.0;
        for (uint i = 0; i < REFERENCE_STEP_COUNT; i++)
        {
            float3 term = medium.density(position + (float(i) + 0.5) * dt * dir) - compensation;
            float3 next = sum + term;
            compensation = (next - sum) - term;
            sum = next;
        }
        reference = exp(-dot(medium.extinction, sum * dt));
    }
    c.store<2>(0, float2(lookup, reference));
}

// Input (u, v, bundle u, member) at multiple-scattering texel centres; output (LUT
// lookup at the decoded point for that member, its direct radiance, second order, f_ms).
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void multiple_scattering_reference(uint3 thread: SV_DispatchThreadID)
{
    Case c = Case(thread.x, 4, 4);
    if (!c.valid)
        return;
    float4 input = c.load<4>(0);
    uint member = uint(input.w);
    float2 mu_s_h = multiple_scattering_mu_s_h(input.xy);
    float wavelength = sample_wavelength_bundle(cie, input.z).wavelength[member];
    SpectralMedium medium = SpectralMedium(ozone_cross_section, ozone_profile, wavelength, scene.media);
    float3 position = float3(0.0, PLANET_RADIUS + mu_s_h.y, 0.0);
    float3 sun = float3(sqrt(1.0 - mu_s_h.x * mu_s_h.x), mu_s_h.x, 0.0);
    float4 bundle_lookup = MultipleScatteringLookup(position, sun).sample(multiple_scattering_lut, linear_clamp, input.z);
    float lookup = bundle_lookup[member];
    MultipleScattering direct = multiple_scattering(column_density_lut, medium, mu_s_h.x, mu_s_h.y);
    c.store<4>(0, float4(lookup, direct.radiance(), direct.second_order, direct.transfer));
}
