module planck;

import common.constants;

// Blackbody spectral radiance at `wavelength` nm. The constants are rescaled
// SI values (h in J s x 1e18, c in nm/s truncated to 2.9e17, k in J/K x 1e18);
// they fix the scene's absolute brightness, which exposure and the MLP are
// tuned to, so they stay as the original wrote them.
public float planck(float temperature, float wavelength)
{
    const float h = 6.62607015e-16;
    const float c = 2.9e17;
    const float k = 1.38e-5;
    float w2 = wavelength * wavelength;
    return 2.0 * h * c * c / (w2 * w2 * wavelength) / (exp(h * c / (wavelength * k * temperature)) - 1.0);
}

// Irradiance at `wavelength` from the whole sun disk at normal incidence: its
// radiance times 2 pi (1 - cos r), written as 4 pi sin^2(r / 2) because
// 1 - cos r cancels in f32.
public float solar_irradiance(float wavelength)
{
    float sin_half = sin(0.5 * SUN_ANGULAR_RADIUS);
    return planck(SUN_TEMPERATURE, wavelength) * 4.0 * PI * sin_half * sin_half;
}
