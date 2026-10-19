// Earth surface reflectance: Burley diffuse, GGX land specular, and an ocean
// lobe blending Beckmann and GGX by view angle; and bounce directions sampled
// in proportion to it.

module brdf;

import common.constants;
import common.geometry;
import common.sampling;

float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

float burley_diffuse(float roughness, float n_dot_l, float n_dot_v, float l_dot_h)
{
    float retro_reflection = 2.0 * roughness * l_dot_h * l_dot_h;
    float f_l = pow5(1.0 - n_dot_l);
    float f_v = pow5(1.0 - n_dot_v);
    float retro = retro_reflection * (f_l + f_v + f_l * f_v * (retro_reflection - 1.0));
    return ((1.0 - 0.5 * f_l) * (1.0 - 0.5 * f_v) + retro) / PI;
}

// Unpolarized Fresnel reflectance of a dielectric with normal-incidence
// reflectance f0.
float fresnel_dielectric(float v_dot_h, float f0)
{
    float sqrt_f0 = sqrt(f0);
    float eta = (1.0 + sqrt_f0) / (1.0 - sqrt_f0);
    float sin_i = sqrt(saturate(1.0 - v_dot_h * v_dot_h));
    float sin_t = sin_i / max(eta, 1e-8);
    float cos_t = sqrt(1.0 - sin_t * sin_t);
    float r_s = (v_dot_h - eta * cos_t) / max(v_dot_h + eta * cos_t, 1e-8);
    float r_p = (cos_t - eta * v_dot_h) / max(cos_t + eta * v_dot_h, 1e-8);
    return saturate((r_s * r_s + r_p * r_p) * 0.5);
}

float smith_lambda(float n_dot_x, float alpha2)
{
    float c2 = n_dot_x * n_dot_x;
    return (-1.0 + sqrt(alpha2 * (1.0 - c2) / c2 + 1.0)) * 0.5;
}

// The GGX (Trowbridge-Reitz) distribution of normals at squared alpha `alpha2`.
float ggx_distribution(float alpha2, float n_dot_h)
{
    float den = (alpha2 - 1.0) * n_dot_h * n_dot_h + 1.0;
    return alpha2 / (PI * den * den);
}

// GGX with height-correlated Smith masking; `roughness` is the GGX alpha.
float ggx_specular(float roughness, float f0, float n_dot_l, float n_dot_v, float l_dot_h, float n_dot_h)
{
    float alpha2 = roughness * roughness;
    float d = ggx_distribution(alpha2, n_dot_h);
    float g = 1.0 / (1.0 + smith_lambda(n_dot_v, alpha2) + smith_lambda(n_dot_l, alpha2));
    return d * g * fresnel_dielectric(l_dot_h, f0) / max(4.0 * n_dot_l * n_dot_v, 1e-5);
}

// The Beckmann alpha of the ocean lobe at `roughness`.
float beckmann_alpha(float roughness)
{
    return roughness * roughness * 2.0;
}

// Beckmann NDF with V-cavity masking. It deliberately omits 1 / (4 nl nv):
// the ocean glint was tuned with this brighter, wider grazing lobe.
float beckmann_specular(float roughness, float f0, float n_dot_l, float n_dot_v, float l_dot_h, float n_dot_h)
{
    float alpha = beckmann_alpha(roughness);
    float alpha2 = alpha * alpha;
    float c2 = n_dot_h * n_dot_h;
    float d = exp(-(1.0 - c2) / (alpha2 * c2)) / max(PI * alpha2 * c2 * c2, 1e-5);
    float g = min(1.0, min(2.0 * n_dot_v * n_dot_h / l_dot_h, 2.0 * n_dot_l * n_dot_h / l_dot_h));
    return d * g * fresnel_dielectric(l_dot_h, f0);
}

float ocean_roughness_at(float bathymetry)
{
    return lerp(0.25, 0.19, smoothstep(0.3, 0.7, bathymetry));
}

// The ocean lobe's share of the specular term, for the ocean mask `ocean`.
float ocean_share(float ocean)
{
    return smoothstep(0.6, 1.0, ocean);
}

// GGX's share of the ocean lobe; Beckmann has the rest.
float ocean_ggx_share(float n_dot_v)
{
    return clamp(smoothstep(0.2, 0.95, n_dot_v), 0.05, 0.94);
}

public struct SurfaceResponse
{
    public float brdf;
    public float n_dot_l;
};

// `albedo` is spectral reflectance, `ocean` the ocean mask, `bathymetry` the
// bathymetry map (1 = shallow); v, n, l are unit vectors away from the surface.
public SurfaceResponse earth_brdf(float albedo, float ocean, float bathymetry, float3 v, float3 n, float3 l)
{
    float3 h = normalize(v + l);
    float n_dot_l = saturate(dot(n, l));
    float n_dot_v = saturate(dot(n, v));
    float l_dot_h = saturate(dot(l, h));
    float n_dot_h = saturate(dot(n, h));

    const float land_roughness = 0.73;
    float ocean_roughness = ocean_roughness_at(bathymetry);

    float diffuse = burley_diffuse(land_roughness, n_dot_l, n_dot_v, l_dot_h);
    float land_specular = ggx_specular(land_roughness, 0.04, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
    float ocean_ggx = ggx_specular(ocean_roughness, 0.02, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
    float ocean_beckmann = 0.65 * beckmann_specular(ocean_roughness, 0.02, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
    float ocean_specular = lerp(ocean_beckmann, ocean_ggx, ocean_ggx_share(n_dot_v));

    SurfaceResponse response;
    response.brdf = albedo * diffuse * 0.28 + lerp(land_specular, ocean_specular, ocean_share(ocean)) * 0.5;
    response.n_dot_l = n_dot_l;
    return response;
}

// Share of bounces that sample the ocean lobe's microfacet normals over full ocean;
// the rest are cosine weighted, for the diffuse term and the broad land lobe.
static const float OCEAN_BOUNCE_SHARE = 0.7;

// The techniques of `sample_earth_bounce` at one surface point and view: the
// selection probability and squared alpha of each microfacet lobe.
struct EarthBounceLobes
{
    float ggx_probability;
    float beckmann_probability;
    float ggx_alpha2;
    // GGX at the Beckmann alpha: its tails are longer, so it covers the narrow lobe.
    float beckmann_alpha2;

    __init(float ocean, float bathymetry, float n_dot_v)
    {
        float roughness = ocean_roughness_at(bathymetry);
        float specular_probability = OCEAN_BOUNCE_SHARE * ocean_share(ocean);
        ggx_probability = specular_probability * ocean_ggx_share(n_dot_v);
        beckmann_probability = specular_probability - ggx_probability;
        ggx_alpha2 = roughness * roughness;
        beckmann_alpha2 = beckmann_alpha(roughness) * beckmann_alpha(roughness);
    }

    // The mixture's solid-angle density of `l`.
    float pdf(float3 v, float3 n, float3 l)
    {
        float n_dot_l = dot(n, l);
        if (!(n_dot_l > 0.0))
            return 0.0;
        float3 h = normalize(v + l);
        float n_dot_h = saturate(dot(n, h));
        // A reflected direction's density is the normal's over 4 v.h.
        float specular = (ggx_probability * ggx_distribution(ggx_alpha2, n_dot_h) +
                          beckmann_probability * ggx_distribution(beckmann_alpha2, n_dot_h)) *
                         n_dot_h / (4.0 * max(dot(v, h), 1e-6));
        return (1.0 - ggx_probability - beckmann_probability) * n_dot_l / PI + specular;
    }
};

// A normal from the GGX distribution of squared alpha `alpha2` about the unit `n`,
// with density ggx_distribution * n.h.
float3 sample_ggx_normal(float2 u, float alpha2, float3 n)
{
    float cos2_theta = (1.0 - u.x) / (1.0 + (alpha2 - 1.0) * u.x);
    float sin_theta = sqrt(saturate(1.0 - cos2_theta));
    float phi = 2.0 * PI * u.y;
    return local_to_world(float3(sin_theta * cos(phi), sin_theta * sin(phi), sqrt(cos2_theta)), n);
}

public struct BounceSample
{
    // Unit direction away from the surface.
    public float3 dir;
    // Its solid-angle density; 0 when a reflection about a sampled normal points
    // below the surface, where the response is 0 too.
    public float pdf;
};

// A bounce direction from unit `v` off the Earth surface with shading normal `n`:
// one-sample MIS over cosine weighting and the ocean lobe's two microfacet
// distributions, chosen by `u.z`, so the estimate brdf n.l / pdf stays bounded
// where cosine weighting alone would see the sky's glare in the ocean rarely.
public BounceSample sample_earth_bounce(float3 u, float ocean, float bathymetry, float3 v, float3 n)
{
    EarthBounceLobes lobes = EarthBounceLobes(ocean, bathymetry, saturate(dot(n, v)));
    BounceSample bounce;
    if (u.z < lobes.ggx_probability)
        bounce.dir = reflect(-v, sample_ggx_normal(u.xy, lobes.ggx_alpha2, n));
    else if (u.z < lobes.ggx_probability + lobes.beckmann_probability)
        bounce.dir = reflect(-v, sample_ggx_normal(u.xy, lobes.beckmann_alpha2, n));
    else
        bounce.dir = sample_hemisphere_cosine(u.xy, n);
    bounce.pdf = lobes.pdf(v, n, bounce.dir);
    return bounce;
}

// The density with which `sample_earth_bounce` picks `l`.
public float earth_bounce_pdf(float ocean, float bathymetry, float3 v, float3 n, float3 l)
{
    return EarthBounceLobes(ocean, bathymetry, saturate(dot(n, v))).pdf(v, n, l);
}
