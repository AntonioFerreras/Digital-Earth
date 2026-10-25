// Phase functions (per steradian) of the cosine between the propagation
// direction and the scattered direction, with exact importance sampling.
// Samplers take uniform numbers `u` in [0, 1) and the unit propagation `view`.

module phase;

import common.constants;
import common.geometry;

public static const float ISOTROPIC_PHASE = 1.0 / (4.0 * PI);
// Klein-Nishina asymmetry of the aerosol phase, a sharp forward peak.
public static const float MIE_KLEIN_NISHINA_E = 3000.0;
// Henyey-Greenstein g of the aerosol phase after the first scatter.
public static const float MIE_HG_G = 0.8;

public float rayleigh_phase(float cos_theta)
{
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

float3 phase_direction(float cos_theta, float u_phi, float3 view)
{
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = 2.0 * PI * u_phi;
    return local_to_world(float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta), view);
}

public float hg_phase(float cos_theta, float g)
{
    return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

public float3 sample_hg_phase(float2 u, float3 view, float g)
{
    float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
    return phase_direction((1.0 + g * g - s * s) / (2.0 * g), u.y, view);
}

public float klein_nishina_phase(float cos_theta, float e)
{
    return e / (2.0 * PI * (e * (1.0 - cos_theta) + 1.0) * log(2.0 * e + 1.0));
}

public float3 sample_klein_nishina_phase(float2 u, float3 view, float e)
{
    return phase_direction((-pow(2.0 * e + 1.0, 1.0 - u.x) + e + 1.0) / e, u.y, view);
}

// Jendersie and d'Eon 2023, "An Approximate Mie Scattering Function for Fog and
// Cloud Rendering" (NVIDIA, MIT licence).
public float draine_phase(float cos_theta, float g, float alpha)
{
    return ((1.0 - g * g) * (1.0 + alpha * cos_theta * cos_theta))
        / (4.0 * (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0) * PI * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

public float draine_mean_cosine(float g, float alpha)
{
    return g * (1.0 + alpha * (3.0 + 2.0 * g * g) / 5.0) / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
}

public float3 sample_draine_phase(float2 u, float3 view, float g, float a)
{
    float xi = u.x;
    float g2 = g * g;
    float g3 = g * g2;
    float g4 = g2 * g2;
    float g6 = g2 * g4;
    float pgp1_2 = (1.0 + g2) * (1.0 + g2);
    float t1a = -a + a * g4;
    float t1a3 = t1a * t1a * t1a;
    float t2 = -1296.0 * (-1.0 + g2) * (a - a * g2) * t1a * (4.0 * g2 + a * pgp1_2);
    float t3 = 3.0 * g2 * (1.0 + g * (-1.0 + 2.0 * xi)) + a * (2.0 + g2 + g3 * (1.0 + 2.0 * g2) * (-1.0 + 2.0 * xi));
    float t4a = 432.0 * t1a3 + t2 + 432.0 * (a - a * g2) * t3 * t3;
    float t4b = -144.0 * a * g2 + 288.0 * a * g4 - 144.0 * a * g6;
    float t4b3 = t4b * t4b * t4b;
    float t4 = t4a + sqrt(-4.0 * t4b3 + t4a * t4a);
    float t4p3 = pow(t4, 1.0 / 3.0);
    float cbrt2 = pow(2.0, 1.0 / 3.0);
    float t6 = (2.0 * t1a + (48.0 * cbrt2 * (-(a * g2) + 2.0 * a * g4 - a * g6)) / t4p3 + t4p3 / (3.0 * cbrt2)) / (a - a * g2);
    float t5 = 6.0 * (1.0 + g2) + t6;
    float root = -0.5 * sqrt(t5) + sqrt(6.0 * (1.0 + g2) - (8.0 * t3) / (a * (-1.0 + g2) * sqrt(t5)) - t6) / 2.0;
    return phase_direction((1.0 + g2 - root * root) / (2.0 * g), u.y, view);
}

// HG + Draine mixture fitted to water droplets of diameter `d` um, 5 to 50.
struct CloudPhaseFit
{
    float g_hg;
    float g_draine;
    float alpha_draine;
    float w_draine;

    __init(float d)
    {
        g_hg = exp(-0.0990567 / (d - 1.67154));
        g_draine = exp(-2.20679 / (d + 3.91029) - 0.428934);
        alpha_draine = exp(3.62489 - 8.29288 / (d + 5.52825));
        w_draine = exp(-0.599085 / (d - 0.641583) - 0.665888);
    }

    float evaluate(float cos_theta)
    {
        return lerp(hg_phase(cos_theta, g_hg), draine_phase(cos_theta, g_draine, alpha_draine), w_draine);
    }

    float mean_cosine()
    {
        return lerp(g_hg, draine_mean_cosine(g_draine, alpha_draine), w_draine);
    }
};

// The clouds' droplets of 8 um. `reduce_peak` (scatters after the first) swaps the
// fitted HG g of ~0.984 for a broader 0.91.
CloudPhaseFit cloud_phase_fit(bool reduce_peak)
{
    CloudPhaseFit fit = CloudPhaseFit(8.0);
    if (reduce_peak)
        fit.g_hg = 0.91;
    return fit;
}

// The share of the forward peak, the Henyey-Greenstein lobe, and the rest, Draine's
// lobe, for a delta-M split (Wiscombe 1977) that takes the peak as scattered straight on.
public float cloud_phase_peak(bool reduce_peak)
{
    return 1.0 - cloud_phase_fit(reduce_peak).w_draine;
}

public float cloud_phase_broad_mean_cosine(bool reduce_peak)
{
    CloudPhaseFit fit = cloud_phase_fit(reduce_peak);
    return draine_mean_cosine(fit.g_draine, fit.alpha_draine);
}

public float3 sample_cloud_phase_broad(float3 u, float3 view, bool reduce_peak)
{
    CloudPhaseFit fit = cloud_phase_fit(reduce_peak);
    return sample_draine_phase(u.yz, view, fit.g_draine, fit.alpha_draine);
}

public float cloud_phase(float cos_theta, bool reduce_peak)
{
    return cloud_phase_fit(reduce_peak).evaluate(cos_theta);
}

public float cloud_phase_mean_cosine(bool reduce_peak)
{
    return cloud_phase_fit(reduce_peak).mean_cosine();
}

// The mixture for droplets of diameter `d` um with its HG lobe's g replaced by `g_hg`.
public float cloud_phase_for_droplets(float cos_theta, float d, float g_hg)
{
    CloudPhaseFit fit = CloudPhaseFit(d);
    fit.g_hg = g_hg;
    return fit.evaluate(cos_theta);
}

// u.x picks the lobe, u.yz sample it.
public float3 sample_cloud_phase(float3 u, float3 view, bool reduce_peak)
{
    CloudPhaseFit fit = cloud_phase_fit(reduce_peak);
    if (u.x < fit.w_draine)
        return sample_draine_phase(u.yz, view, fit.g_draine, fit.alpha_draine);
    return sample_hg_phase(u.yz, view, fit.g_hg);
}
