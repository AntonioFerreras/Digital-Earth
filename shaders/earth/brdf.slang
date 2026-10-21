// Earth surface reflectance: Burley diffuse and GGX specular over land; over water,
// the Cox-Munk glint of a wind-roughened sea and the light scattered up out of the
// water, or the previous look, a Beckmann and GGX blend by view angle over the land
// terms; and bounce directions sampled in proportion to it.

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

// Normal-incidence reflectance of water.
static const float WATER_F0 = 0.02;

// The sea's mean square slopes along and across the wind at `wind_speed` m/s at 10 m:
// Guerin, Capelle and Hartmann's (2026) piecewise-linear fits to sun glint that IASI
// saw over ERA5 winds, within the scatter of Cox and Munk's (1954) clean-sea fit.
// Their fits start at 2 m/s; below it the slopes fall linearly to Cox and Munk's
// calm-sea intercepts.
float2 sea_mean_square_slopes(float wind_speed)
{
    if (wind_speed < 2.0)
        return lerp(float2(1e-4, 3e-3), float2(8.27e-3, 7.61e-3), wind_speed / 2.0);
    float upwind = wind_speed < 6.5 ? 2.12e-3 * wind_speed + 4.03e-3
                 : wind_speed < 11.0 ? 4.02e-3 * wind_speed - 7.76e-3
                                     : 2.59e-3 * wind_speed + 8.24e-3;
    return float2(upwind, 1.82e-3 * wind_speed + 3.97e-3);
}

// The decay rate of the slope tail, 1.2 ln 10 per unit slope.
static const float SEA_TAIL_DECAY = 2.763;

// Smith's masking Lambda of the Beckmann distribution, in Walter et al.'s (2007)
// rational fit.
float beckmann_lambda(float n_dot_x, float alpha2)
{
    float c2 = n_dot_x * n_dot_x;
    float a = sqrt(c2 / (alpha2 * (1.0 - c2)));
    return a < 1.6 ? (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a) : 0.0;
}

// The distribution of the sea's facet slopes at one point (Guerin, Capelle and
// Hartmann 2026): a Gaussian core, stretched along the wind, plus the exponential
// tail they found beyond about two standard deviations. The sum bends smoothly from
// one into the other, where their maximum would kink. Part of the tail they measured
// may be light that foam and aerosols scatter, so it enters with a weight.
struct SeaSlopes
{
    float3 normal;
    float3 upwind;
    float3 crosswind;
    // Mean square slopes along `upwind` and `crosswind`.
    float2 variance;
    // The tail's density at slope 0, per unit slope area.
    float tail_amplitude;
    // 1 over the integral of the core and the tail together.
    float normalization;

    __init(Water water, float3 n)
    {
        normal = n;
        float2 calm = sea_mean_square_slopes(0.0);
        float2 mean_square = (calm + water.wind_roughness * (sea_mean_square_slopes(water.wind_speed) - calm)) * water.slope_scale;
        float3 along = water.along - dot(water.along, n) * n;
        if (dot(along, along) > 1e-6)
        {
            upwind = normalize(along);
            variance = mean_square;
        }
        else
        {
            upwind = local_to_world(float3(1.0, 0.0, 0.0), n);
            variance = 0.5 * (mean_square.x + mean_square.y);
        }
        crosswind = cross(n, upwind);
        tail_amplitude = water.tail * (0.021 * water.wind_speed + 0.244);
        // The tail integrates to 2 pi amplitude / decay^2 over the slope plane.
        normalization = 1.0 / (1.0 + 2.0 * PI * tail_amplitude / (SEA_TAIL_DECAY * SEA_TAIL_DECAY));
    }

    // The core's distribution of unit facet normals `h`, which integrates to 1 over
    // the hemisphere weighted by n.h.
    float core(float3 h)
    {
        float c = max(dot(h, normal), 1e-4);
        float2 slope = float2(dot(h, upwind), dot(h, crosswind)) / c;
        float density = exp(-0.5 * dot(slope * slope, 1.0 / variance)) / (2.0 * PI * sqrt(variance.x * variance.y));
        return density / (c * c * c * c);
    }

    // The full distribution of unit facet normals `h`.
    float distribution(float3 h)
    {
        float c = max(dot(h, normal), 1e-4);
        float tail = tail_amplitude * exp(-SEA_TAIL_DECAY * sqrt(max(1.0 - c * c, 0.0)) / c) / (c * c * c * c);
        return (core(h) + tail) * normalization;
    }

    // Smith's Lambda of the core for unit `x`, through Beckmann's at the alpha of the
    // core's slope variance in x's azimuth.
    float lambda(float3 x)
    {
        float2 azimuth = float2(dot(x, upwind), dot(x, crosswind));
        float alpha2 = 2.0 * dot(azimuth * azimuth, variance) / max(dot(azimuth, azimuth), 1e-12);
        return beckmann_lambda(dot(x, normal), alpha2);
    }

    // A unit facet normal from the core, with density core * n.h: Beckmann's
    // anisotropic sampling at alphas squared twice the variances.
    float3 sample_core(float2 u)
    {
        float2 alpha = sqrt(2.0 * variance);
        float angle = 2.0 * PI * u.y;
        float2 direction = normalize(float2(alpha.x * cos(angle), alpha.y * sin(angle)));
        float tan2_theta = -log(1.0 - u.x) / dot(direction * direction, 1.0 / (alpha * alpha));
        float cos_theta = rsqrt(1.0 + tan2_theta);
        float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
        return normalize(sin_theta * (direction.x * upwind + direction.y * crosswind) + cos_theta * normal);
    }
};

// Sun and sky glitter off a wind-roughened sea: the sea's slope distribution with
// height-correlated Smith masking of its core.
float sea_glint(SeaSlopes slopes, float3 v, float3 l, float3 h, float n_dot_l, float n_dot_v, float l_dot_h)
{
    float g = 1.0 / (1.0 + slopes.lambda(v) + slopes.lambda(l));
    return slopes.distribution(h) * g * fresnel_dielectric(l_dot_h, WATER_F0) / max(4.0 * n_dot_l * n_dot_v, 1e-5);
}

// Light refracted into the water, scattered back up evenly, and refracted out again.
// `reflectance` is what a view from straight above sees, so both crossings count
// relative to normal incidence.
float water_body(float reflectance, float n_dot_l, float n_dot_v)
{
    float transmitted = (1.0 - fresnel_dielectric(n_dot_l, WATER_F0)) * (1.0 - fresnel_dielectric(n_dot_v, WATER_F0));
    return reflectance / PI * transmitted / ((1.0 - WATER_F0) * (1.0 - WATER_F0));
}

// The water at a surface point.
public struct Water
{
    // Water fraction in [0, 1].
    public float fraction;
    // Wind speed in m/s at 10 m, which roughens the water.
    public float wind_speed;
    // Factor on the slope variance that wind speed gives, for roughness the wind
    // field does not resolve.
    public float slope_scale;
    // The wind's direction in the tangent plane, or 0 where it has none.
    public float3 along;
    // The weight of the slope distribution's exponential tail, 0 to 1.
    public float tail;
    // Temporary: the factor on the slope variance the wind adds over a calm sea's.
    public float wind_roughness;
    // Bathymetry map, 1 = shallow, which roughens the previous ocean lobe instead.
    public float bathymetry;
    // Temporary switches from the previous look to the physical model: the glint
    // from the wind, and the water's colour from light scattered up out of it.
    public bool glint;
    public bool body;

    // The share of the surface that reflects as water.
    float share()
    {
        return body ? fraction : ocean_share(fraction);
    }
};

// Spectral reflectances at one wavelength: the land's albedo, and what the water
// body returns seen from straight above.
public struct Reflectance
{
    public float land;
    public float water;
};

public struct SurfaceResponse
{
    public float brdf;
    public float n_dot_l;
};

// v, n, l are unit vectors away from the surface.
public SurfaceResponse earth_brdf(Reflectance albedo, Water water, float3 v, float3 n, float3 l)
{
    float3 h = normalize(v + l);
    float n_dot_l = saturate(dot(n, l));
    float n_dot_v = saturate(dot(n, v));
    float l_dot_h = saturate(dot(l, h));
    float n_dot_h = saturate(dot(n, h));

    const float land_roughness = 0.73;
    float diffuse = burley_diffuse(land_roughness, n_dot_l, n_dot_v, l_dot_h);
    float land_specular = 0.5 * ggx_specular(land_roughness, 0.04, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
    float water_specular;
    if (water.glint)
    {
        water_specular = sea_glint(SeaSlopes(water, n), v, l, h, n_dot_l, n_dot_v, l_dot_h);
    }
    else
    {
        float roughness = ocean_roughness_at(water.bathymetry);
        float ocean_ggx = ggx_specular(roughness, WATER_F0, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
        float ocean_beckmann = 0.65 * beckmann_specular(roughness, WATER_F0, n_dot_l, n_dot_v, l_dot_h, n_dot_h);
        water_specular = 0.5 * lerp(ocean_beckmann, ocean_ggx, ocean_ggx_share(n_dot_v));
    }

    SurfaceResponse response;
    float land = albedo.land * diffuse * 0.28;
    if (water.body)
        response.brdf = lerp(land + land_specular, water_specular + water_body(albedo.water, n_dot_l, n_dot_v), water.fraction);
    else
        response.brdf = land + lerp(land_specular, water_specular, water.share());
    response.n_dot_l = n_dot_l;
    return response;
}

// Share of bounces that sample the water's microfacet normals over full water; the
// rest are cosine weighted, for the diffuse terms and the broad land lobe.
static const float OCEAN_BOUNCE_SHARE = 0.7;

// The techniques of `sample_earth_bounce` at one surface point and view: the
// selection probability and squared alpha of each microfacet lobe.
struct EarthBounceLobes
{
    float ggx_probability;
    float beckmann_probability;
    float ggx_alpha2;
    float beckmann_alpha2;
    // The Beckmann technique samples the sea's core normals; otherwise GGX normals at
    // the previous lobe's Beckmann alpha, whose longer tails cover it.
    bool sea;
    SeaSlopes slopes;

    __init(Water water, float3 n, float n_dot_v)
    {
        float specular_probability = OCEAN_BOUNCE_SHARE * water.share();
        sea = water.glint;
        slopes = SeaSlopes(water, n);
        if (sea)
        {
            ggx_probability = 0.0;
            beckmann_probability = specular_probability;
            ggx_alpha2 = 1.0;
            beckmann_alpha2 = 1.0;
        }
        else
        {
            float roughness = ocean_roughness_at(water.bathymetry);
            ggx_probability = specular_probability * ocean_ggx_share(n_dot_v);
            beckmann_probability = specular_probability - ggx_probability;
            ggx_alpha2 = roughness * roughness;
            beckmann_alpha2 = beckmann_alpha(roughness) * beckmann_alpha(roughness);
        }
    }

    // The mixture's solid-angle density of `l`.
    float pdf(float3 v, float3 n, float3 l)
    {
        float n_dot_l = dot(n, l);
        if (!(n_dot_l > 0.0))
            return 0.0;
        float3 h = normalize(v + l);
        float n_dot_h = saturate(dot(n, h));
        float beckmann = sea ? slopes.core(h) : ggx_distribution(beckmann_alpha2, n_dot_h);
        // A reflected direction's density is the normal's over 4 v.h.
        float specular = (ggx_probability * ggx_distribution(ggx_alpha2, n_dot_h) + beckmann_probability * beckmann) *
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
// one-sample MIS over cosine weighting and the water's microfacet distributions,
// chosen by `u.z`, so the estimate brdf n.l / pdf stays bounded where cosine
// weighting alone would see the sky's glare in the water rarely.
public BounceSample sample_earth_bounce(float3 u, Water water, float3 v, float3 n)
{
    EarthBounceLobes lobes = EarthBounceLobes(water, n, saturate(dot(n, v)));
    BounceSample bounce;
    if (u.z < lobes.ggx_probability)
        bounce.dir = reflect(-v, sample_ggx_normal(u.xy, lobes.ggx_alpha2, n));
    else if (u.z < lobes.ggx_probability + lobes.beckmann_probability)
        bounce.dir = reflect(-v, lobes.sea ? lobes.slopes.sample_core(u.xy) : sample_ggx_normal(u.xy, lobes.beckmann_alpha2, n));
    else
        bounce.dir = sample_hemisphere_cosine(u.xy, n);
    bounce.pdf = lobes.pdf(v, n, bounce.dir);
    return bounce;
}

// The density with which `sample_earth_bounce` picks `l`.
public float earth_bounce_pdf(Water water, float3 v, float3 n, float3 l)
{
    return EarthBounceLobes(water, n, saturate(dot(n, v))).pdf(v, n, l);
}
