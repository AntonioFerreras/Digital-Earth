// The sun's light through the atmosphere, linear Rec.709, averaged over wavelength
// bundles that stratify the CIE sampling.

module sunlight;

import atmosphere.media;
import atmosphere.scattering_luts;
import shared.bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;

static const uint SUN_BUNDLE_COUNT = 64;

// The irradiance of the sun along `sun` at `pos`, planet-centred metres, through the
// air above it; zero where the planet hides it.
public float3 sun_irradiance(float3 pos, float3 sun)
{
    ColumnDensity column = ColumnDensity(column_density_lut, pos, sun);
    float3 rgb = 0.0;
    for (uint k = 0; column.visible && k < SUN_BUNDLE_COUNT; k++)
    {
        WavelengthBundle w = sample_wavelength_bundle(cie, (float(k) + 0.5) / float(SUN_BUNDLE_COUNT));
        SpectralMediumBundle media = SpectralMediumBundle(ozone_cross_section, ozone_profile, w.wavelength, scene.media);
        Spectrum irradiance;
        for (int j = 0; j < BUNDLE_WAVELENGTH_COUNT; j++)
            irradiance[j] = solar_irradiance(w.wavelength[j]) * column.transmittance(media.extinction[j]);
        rgb += xyz_to_rec709(mul(w.xyz_weight, irradiance));
    }
    return rgb / float(SUN_BUNDLE_COUNT);
}
