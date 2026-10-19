// Spectral volumetric path tracing of the planet, one wavelength bundle
// (spectral/cie.slang) per path: delta tracking through the Rayleigh, aerosol,
// and ozone media and the cloud shell, ratio-tracked sun transmittance,
// next-event estimation towards the sun cone, phase sampling, the surface of
// earth/surface.slang with a BRDF-sampled bounce, and Russian roulette. `scene`
// flags select between the textured and the symmetric scene (shared/uniforms.slang).
//
// One member of the bundle, the hero, decides every collision, and each
// member's estimate is weighted by spectral MIS (Wilkie et al. 2014, "Hero
// Wavelength Spectral Sampling"; pbrt-v4's VolPathIntegrator): the balance
// heuristic over the bundle's members as heroes. That needs every member to be
// the hero equally often, so the hero is drawn uniformly per path. All members
// share one majorant, so free-flight distances have the same density for every
// member and only the collision events' probabilities differ.
//
// Importing this module declares the globals of shared/bindings.slang and
// shared/earth_bindings.slang, which the kernel must bind.

module path_radiance;

import atmosphere.media;
import atmosphere.phase;
import common.constants;
import common.geometry;
import common.rng;
import common.sampling;
import earth.brdf;
import earth.clouds;
import earth.maps;
import earth.surface;
import shared.bindings;
import shared.earth_bindings;
import shared.uniforms;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

// Interaction ids: Rayleigh, aerosol, and ozone index the media's columns; then the cloud.
static const uint RAYLEIGH = 0;
static const uint MIE = 1;
static const uint CLOUD = 3;
static const float4 SCATTERING_ALBEDO = float4(RAYLEIGH_ALBEDO, AEROSOL_ALBEDO, OZONE_ALBEDO, CLOUD_ALBEDO);

static const float RATIO_TRACKING_CUTOFF = 1e-5;
static const float ROULETTE_PROBABILITY_MIN = 0.05;

enum Event
{
    None,
    Absorb,
    Scatter,
};

struct Interaction
{
    Event event;
    float t;
    uint id;
};

// What one path's bundle sees of the media.
struct Media
{
    SpectralMediumBundle rmo;
    // Bounds every member's extinction, so that all members share the free flights.
    float rmo_majorant;
    bool clouds;
    // Cloud extinction at unit density and the cloud density maximum; clouds are grey.
    float cloud_extinction;
    float cloud_density_max;

    __init(SceneUniforms scene, Spectrum wavelength)
    {
        rmo = SpectralMediumBundle(ozone_cross_section, ozone_profile, wavelength, scene.media);
        // Rayleigh and aerosol densities peak at sea level.
        float3 density_max = float3(rayleigh_density(0.0, rmo.first.rayleigh_sea_level_density),
                                    mie_density(0.0, rmo.first.turbidity), ozone_density_max(ozone_profile));
        Spectrum majorants = mul(rmo.extinction, density_max);
        rmo_majorant = 0.0;
        for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
            rmo_majorant = max(rmo_majorant, majorants[k]);
        clouds = (scene.config.x & SCENE_CLOUDS) != 0;
        cloud_extinction = scene.media.y;
        cloud_density_max = scene.media.z;
    }

    float cloud_extinction_at(float3 pos)
    {
        float r = length(pos);
        float coverage = sample_tiled(earth_clouds, earth_maps.clouds, sphere_uv(pos / r));
        return cloud_extinction * cloud_density(r, coverage, cloud_density_max);
    }
};

// The balance heuristic's weights (spectral MIS): every member's path density
// over the hero's, scaled to a mean of 1 after each event so that long tracks
// stay in range.
struct SpectralWeights
{
    Spectrum ratio;
    // The member whose probabilities decide the collisions.
    uint hero;

    // Applies an event whose probability for each member is proportional to `density`.
    [mutating]
    void observe(Spectrum density)
    {
        ratio *= density / density[hero];
        ratio /= dot(ratio, Spectrum(1.0 / float(BUNDLE_WAVELENGTH_COUNT)));
    }
};

// The atmosphere along the ray up to the surface hit `t_surface` (or SURFACE_MISS),
// as (start, end); empty (end <= start) when the ray misses the atmosphere.
float2 atmosphere_segment(float3 pos, float3 dir, float t_surface)
{
    float2 t = ray_sphere(pos, dir, ATMOSPHERE_RADIUS);
    return float2(max(t.x, 0.0), t_surface >= 0.0 ? t_surface : t.y);
}

// The ray's two passes through the cloud shell, (start, end) each, cut at the
// surface hit so that terrain inside the shell occludes the clouds behind it.
float4 cloud_segments(float3 pos, float3 dir, float t_surface)
{
    float2 top = ray_sphere(pos, dir, CLOUD_TOP_RADIUS);
    float2 base = ray_sphere(pos, dir, CLOUD_BASE_RADIUS);
    float4 segments = float4(max(top.x, 0.0), top.y, 0.0, 0.0);
    if (base.y > 0.0)
    {
        // The inner sphere splits the pass: before it (if ahead) and after it.
        segments = base.x > 0.0 ? float4(segments.x, base.x, base.y, top.y) : float4(base.y, top.y, 0.0, 0.0);
    }
    float end = t_surface >= 0.0 ? t_surface : top.y;
    return float4(segments.x, min(segments.y, end), segments.z, min(segments.w, end));
}

// Delta tracking through the clouds' two passes: the first real collision before
// `t_end`, and whether the particle there scatters or absorbs.
Interaction track_clouds(float3 pos, float3 dir, float t_surface, Media media, inout Rng rng)
{
    Interaction interaction = { Event.None, 0.0, CLOUD };
    float majorant = media.cloud_extinction * media.cloud_density_max;
    if (!(majorant > 0.0))
        return interaction;
    float4 segments = cloud_segments(pos, dir, t_surface);
    for (uint i = 0; i < 2; i++)
    {
        float t = segments[2 * i];
        while (true)
        {
            float4 u = rng.next4();
            t -= log(1.0 - u.x) / majorant;
            if (t >= segments[2 * i + 1])
                break;
            if (u.y * majorant < media.cloud_extinction_at(pos + t * dir))
            {
                interaction.event = u.z < CLOUD_ALBEDO ? Event.Scatter : Event.Absorb;
                interaction.t = t;
                return interaction;
            }
        }
    }
    return interaction;
}

// Delta tracking through the RMO media over `segment` at the hero member of
// `weights`: the first real collision and whether the particle there scatters or
// absorbs. `weights` observes every tentative collision.
Interaction track_rmo(float3 pos, float3 dir, float2 segment, Media media, inout Rng rng,
                      inout SpectralWeights weights)
{
    Interaction interaction = { Event.None, segment.y, 0 };
    float majorant = media.rmo_majorant;
    float t = segment.x;
    while (true)
    {
        float4 u = rng.next4();
        t -= log(1.0 - u.x) / majorant;
        if (t >= segment.y)
            break;
        float3 density = media.rmo.first.density(pos + t * dir);
        Spectrum extinction = mul(media.rmo.extinction, density);
        float threshold = u.y * majorant;
        if (threshold < extinction[weights.hero])
        {
            float3 hero = media.rmo.extinction[weights.hero] * density;
            uint id = 0;
            float cumulative = hero.x;
            while (id < 2 && threshold >= cumulative)
            {
                id += 1;
                cumulative += hero[id];
            }
            // Scattering or absorbing takes the same share of each member's collisions.
            Spectrum medium_extinction;
            for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
                medium_extinction[k] = media.rmo.extinction[k][id] * density[id];
            weights.observe(medium_extinction);
            interaction.event = u.z < SCATTERING_ALBEDO[id] ? Event.Scatter : Event.Absorb;
            interaction.t = t;
            interaction.id = id;
            break;
        }
        weights.observe(majorant - extinction);
    }
    return interaction;
}

// The nearest collision in front of the surface hit `t_surface`. The cloud
// collision comes first, so that the RMO tracking stops there and `weights`
// observes only the collisions on the path.
Interaction sample_interaction(float3 pos, float3 dir, float t_surface, Media media, inout Rng rng,
                               inout SpectralWeights weights)
{
    float2 segment = atmosphere_segment(pos, dir, t_surface);
    Interaction cloud = { Event.None, 0.0, CLOUD };
    if (media.clouds)
        cloud = track_clouds(pos, dir, t_surface, media, rng);
    if (cloud.event != Event.None)
        segment.y = min(segment.y, cloud.t);
    Interaction interaction = track_rmo(pos, dir, segment, media, rng, weights);
    return interaction.event == Event.None ? cloud : interaction;
}

// Ratio-tracking estimates of every member's transmittance towards the unit `dir`,
// with the shared majorant.
Spectrum transmittance(float3 pos, float3 dir, Media media, inout Rng rng)
{
    float2 segment = atmosphere_segment(pos, dir, SURFACE_MISS);
    Spectrum result = 1.0;
    float t = segment.x;
    while (any(result >= RATIO_TRACKING_CUTOFF))
    {
        t -= log(1.0 - rng.next4().x) / media.rmo_majorant;
        if (t >= segment.y)
            break;
        result *= 1.0 - mul(media.rmo.extinction, media.rmo.first.density(pos + t * dir)) / media.rmo_majorant;
    }
    if (!media.clouds)
        return result;
    float majorant = media.cloud_extinction * media.cloud_density_max;
    if (!(majorant > 0.0))
        return result;
    float4 segments = cloud_segments(pos, dir, SURFACE_MISS);
    for (uint i = 0; i < 2; i++)
    {
        float cloud = 1.0;
        float t_cloud = segments[2 * i];
        while (cloud >= RATIO_TRACKING_CUTOFF)
        {
            t_cloud -= log(1.0 - rng.next4().x) / majorant;
            if (t_cloud >= segments[2 * i + 1])
                break;
            cloud *= 1.0 - media.cloud_extinction_at(pos + t_cloud * dir) / majorant;
        }
        result *= cloud;
    }
    return result;
}

// How the phase function of a collision is chosen on bounce `bounce`.
struct PhaseChoice
{
    uint id;
    // The aerosol switches to Henyey-Greenstein, clouds to their broader peak.
    bool reduce_peak;
    // Hack of the original: deep bounces see clouds as isotropic.
    bool isotropic_cloud;

    __init(uint id, uint bounce)
    {
        this.id = id;
        reduce_peak = bounce > 0;
        isotropic_cloud = bounce >= CLOUD_DEEP_BOUNCE;
    }

    bool mie_hg(SceneUniforms scene)
    {
        return reduce_peak && (scene.config.x & SCENE_MIE_HG_AFTER_FIRST) != 0;
    }

    float evaluate(float cos_theta, SceneUniforms scene)
    {
        if (id == RAYLEIGH)
            return rayleigh_phase(cos_theta);
        if (id == MIE)
            return mie_hg(scene) ? hg_phase(cos_theta, MIE_HG_G) : klein_nishina_phase(cos_theta, MIE_KLEIN_NISHINA_E);
        return isotropic_cloud ? ISOTROPIC_PHASE : cloud_phase(cos_theta, reduce_peak);
    }

    // A new direction for propagation `dir`; `weight` is phase / pdf.
    float3 sample(float3 u, float3 dir, SceneUniforms scene, out float weight)
    {
        weight = 1.0;
        if (id == RAYLEIGH)
        {
            float3 sampled = sample_sphere_uniform(u.xy);
            weight = rayleigh_phase(dot(dir, sampled)) * 4.0 * PI;
            return sampled;
        }
        if (id == MIE)
            return mie_hg(scene) ? sample_hg_phase(u.xy, dir, MIE_HG_G) : sample_klein_nishina_phase(u.xy, dir, MIE_KLEIN_NISHINA_E);
        return isotropic_cloud ? sample_sphere_uniform(u.xy) : sample_cloud_phase(u, dir, reduce_peak);
    }
};

public struct PathSample
{
    // Per bundle member, MIS weighted; finite and non-negative.
    public Spectrum radiance;
    // The ray left the atmosphere without any interaction; its `sky_radiance` is
    // not in `radiance`.
    public bool escaped;
};

// One path's estimate of the spectral radiance (the original's units) arriving
// at `pos` from the unit direction `-dir` at the bundle's `wavelength`s in nm, lit
// by a sun in the unit direction `sun`. It draws from `rng` only, so callers
// choose the sample keying. An estimate of XYZ is mul(WavelengthBundle.xyz_weight,
// radiance) (spectral/cie.slang).
//
// For the full estimate, add `sky_radiance(dir, sun, ...)` when `escaped`; a
// caller that canonicalized the ray passes the world `dir` and `sun` there, since
// the uvwz round trip cannot place the sun disk to a fraction of a degree.
// Requires the globals of shared/bindings.slang (LUT buffers) and, when `scene`
// sets SCENE_EARTH_MAPS, SCENE_TERRAIN, or SCENE_CLOUDS, those of
// shared/earth_bindings.slang.
public PathSample path_radiance(float3 pos, float3 dir, float3 sun, Spectrum wavelength, inout Rng rng,
                                SceneUniforms scene)
{
    float cos_sun_radius = cos(SUN_ANGULAR_RADIUS);
    Spectrum sun_irradiance;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        sun_irradiance[k] = solar_irradiance(wavelength[k]);
    Media media = Media(scene, wavelength);

    Spectrum radiance = 0.0;
    // The path's measurement over its density, except for the factors the
    // collisions put into `weights`.
    Spectrum throughput = 1.0;
    SpectralWeights weights = { Spectrum(1.0), min(uint(rng.next4().x * float(BUNDLE_WAVELENGTH_COUNT)),
                                                   uint(BUNDLE_WAVELENGTH_COUNT - 1)) };
    PathSample sample;
    sample.escaped = false;
    for (uint bounce = 0; bounce < scene.config.y; bounce++)
    {
        if (bounce >= CLOUD_DEEP_BOUNCE)
            media.cloud_extinction = CLOUD_EXTINCTION_DEEP;
        float t_surface = intersect_surface(pos, dir, scene);
        Interaction interaction = sample_interaction(pos, dir, t_surface, media, rng, weights);
        float3 light = sample_cone(rng.next4().xy, cos_sun_radius, sun);

        if (interaction.event == Event.Absorb)
            break;
        if (interaction.event == Event.Scatter)
        {
            float3 scatter_pos = pos + interaction.t * dir;
            PhaseChoice phase = PhaseChoice(interaction.id, bounce);
            // The heightfield's shadow is negligible at atmosphere heights.
            if (!(ray_sphere(scatter_pos, light, PLANET_RADIUS).y > 0.0))
            {
                Spectrum sun_transmittance = transmittance(scatter_pos, light, media, rng);
                radiance += throughput * weights.ratio * sun_transmittance * sun_irradiance *
                            phase.evaluate(dot(dir, light), scene);
            }
            float weight;
            dir = phase.sample(rng.next4().xyz, dir, scene, weight);
            pos = scatter_pos;
            throughput *= weight;
        }
        else if (t_surface > 0.0)
        {
            float3 surface_pos = pos + t_surface * dir;
            SurfaceSample s = sample_surface(surface_pos, scene);
            float3 lifted = offset_surface_point(surface_pos, scene);
            float3 view = -dir;
            bool sun_visible = surface_visible(lifted, light, scene);
            Spectrum sun_transmittance = sun_visible ? transmittance(lifted, light, media, rng) : Spectrum(0.0);
            BounceSample bounce = sample_bounce(s, view, rng.next4().xyz, scene);
            for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
            {
                float albedo = srgb_to_spectrum(srgb_to_spectrum_basis, s.albedo_srgb, wavelength[k]);
                SurfaceResponse direct = surface_brdf(s, albedo, view, light, scene);
                float lit = surface_emission(s, wavelength[k], scene) +
                            sun_transmittance[k] * sun_irradiance[k] * direct.brdf * direct.n_dot_l;
                radiance[k] += throughput[k] * weights.ratio[k] * lit;
                SurfaceResponse bounced = surface_brdf(s, albedo, view, bounce.dir, scene);
                throughput[k] *= bounce.pdf > 0.0 ? bounced.brdf * bounced.n_dot_l / bounce.pdf : 0.0;
            }
            if (!(bounce.pdf > 0.0))
                break;
            dir = bounce.dir;
            pos = lifted;
        }
        else
        {
            sample.escaped = bounce == 0;
            break;
        }

        if (bounce >= scene.config.z)
        {
            Spectrum weighted = throughput * weights.ratio;
            float weighted_max = 0.0;
            for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
                weighted_max = max(weighted_max, weighted[k]);
            float termination = max(ROULETTE_PROBABILITY_MIN, 1.0 - weighted_max);
            if (rng.next4().x < termination)
                break;
            throughput /= 1.0 - termination;
        }
    }

    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
        sample.radiance[k] = isfinite(radiance[k]) && radiance[k] > 0.0 ? radiance[k] : 0.0;
    return sample;
}

// What a ray escaping in the unit direction `dir` sees at the bundle's `wavelength`s:
// the sun disk around the unit `sun` and, with SCENE_STARS, the star map.
public Spectrum sky_radiance(float3 dir, float3 sun, Spectrum wavelength, SceneUniforms scene)
{
    bool sun_disk = dot(sun, dir) > cos(SUN_ANGULAR_RADIUS);
    bool stars = (scene.config.x & SCENE_STARS) != 0;
    float3 stars_srgb = stars ? sample_tiled(earth_stars, earth_maps.stars, sphere_uv(dir)).rgb : float3(0.0);
    Spectrum radiance;
    for (int k = 0; k < BUNDLE_WAVELENGTH_COUNT; k++)
    {
        float sun_radiance = planck(SUN_TEMPERATURE, wavelength[k]);
        radiance[k] = sun_disk ? sun_radiance : 0.0;
        if (stars)
        {
            radiance[k] += srgb_to_spectrum(srgb_to_spectrum_basis, stars_srgb, wavelength[k]) * sun_radiance *
                           scene.surface.w;
        }
    }
    return radiance;
}
