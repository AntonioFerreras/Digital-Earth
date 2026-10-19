// The ray marcher's LUT bakes (atmosphere/scattering_luts.slang), one dispatch
// each, one thread per texel. The multiple-scattering bake reads the
// column-density table, so it runs second.

import atmosphere.bruneton;
import atmosphere.media;
import atmosphere.scattering_luts;
import common.constants;
import spectral.cie;
import shared.uniforms;

ConstantBuffer<SceneUniforms> scene;
StructuredBuffer<float> ozone_cross_section;
StructuredBuffer<float> ozone_profile;
Texture2D<float4> column_density_lut;
[format("rgba32f")]
RWTexture2D<float4> column_density_target;
StructuredBuffer<float> cie;
// The texel format follows the texture, rgba32f or rgba16f where rgba32f cannot be filtered.
[format("unknown")]
RWTexture3D<float4> multiple_scattering_target;

[shader("compute")]
[numthreads(8, 8, 1)]
void bake_column_density(uint3 texel: SV_DispatchThreadID)
{
    if (any(texel.xy >= COLUMN_DENSITY_LUT_SIZE))
        return;
    float2 uv = (float2(texel.xy) + 0.5) / float2(COLUMN_DENSITY_LUT_SIZE);
    float2 mu_r = transmittance_mu_r(uv, float2(COLUMN_DENSITY_LUT_SIZE));
    // The densities do not depend on the wavelength.
    SpectralMedium medium = SpectralMedium(ozone_cross_section, ozone_profile, WAVELENGTH_MIN, scene.media);
    column_density_target[texel.xy] = float4(log_column_density_to_top(medium, mu_r.y, mu_r.x), 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bake_multiple_scattering(uint3 texel: SV_DispatchThreadID)
{
    if (any(texel >= MULTIPLE_SCATTERING_LUT_SIZE))
        return;
    float2 mu_s_h = multiple_scattering_mu_s_h(lut_texel_centre(texel, MULTIPLE_SCATTERING_LUT_SIZE).xy);
    WavelengthBundle bundle = sample_wavelength_bundle(cie, multiple_scattering_slice_u(texel.z));
    float4 radiance;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
    {
        SpectralMedium medium = SpectralMedium(ozone_cross_section, ozone_profile, bundle.wavelength[k], scene.media);
        radiance[k] = multiple_scattering(column_density_lut, medium, mu_s_h.x, mu_s_h.y).radiance();
    }
    multiple_scattering_target[texel] = radiance;
}
