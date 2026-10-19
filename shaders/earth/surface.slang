// The planet surface as both render modes see it: intersection, sun
// visibility, and the material at a hit. Every mode reaches the surface only
// through this module, so the heightfield and the Earth maps change in one place.
//
// With SCENE_TERRAIN the surface is the sphere displaced outwards by the
// topography map times `scene.surface.y`; otherwise it is the bare sphere. With
// SCENE_EARTH_MAPS the material comes from the maps; otherwise it is the
// constant grey `scene.surface.x`.

module surface;

import common.constants;
import common.geometry;
import common.sampling;
import earth.brdf;
import earth.maps;
import earth.material;
import shared.earth_bindings;
import shared.uniforms;
import spectral.planck;

public static const float SURFACE_MISS = -1.0;

public struct SurfaceSample
{
    // Unit shading normal.
    public float3 normal;
    // Graded albedo, encoded sRGB in [0, 1] as the Taichi renderer used it.
    public float3 albedo_srgb;
    // 1 over water, 0 over land.
    public float ocean;
    // 1 for land or shallow water, 0 for the deepest ocean.
    public float bathymetry;
    // Night-light intensity in [0, 1].
    public float emissive;
};

static const uint SPHERE_TRACE_STEPS = 250;
// A march stops once the field is within this fraction of the distance marched
// inside the atmosphere. Measured from the camera instead, the tolerance grows to
// kilometres from orbit and stops hits in bands above the sea surface.
static const float SPHERE_TRACE_RELATIVE_EPSILON = 1e-4;

// Not a true distance bound on steep slopes; the march and the normals use it
// exactly as the original did, so overshoots land where they did there.
float land_sdf(float3 position, SceneUniforms scene)
{
    float height = sample_tiled_exact(earth_topography, earth_maps.topography, sphere_uv(normalize(position)));
    return length(position) - PLANET_RADIUS - scene.surface.y * height;
}

// Backward differences along the world axes, one step being half an equatorial
// texel of the topography map.
float3 land_normal(float3 position, SceneUniforms scene)
{
    float e = PI * PLANET_RADIUS / float(earth_maps.topography.x);
    float d = land_sdf(position, scene);
    return normalize(d - float3(land_sdf(position - float3(e, 0.0, 0.0), scene),
                                land_sdf(position - float3(0.0, e, 0.0), scene),
                                land_sdf(position - float3(0.0, 0.0, e), scene)));
}

// Distance along the unit `dir` to the first surface hit, or SURFACE_MISS.
public float intersect_surface(float3 origin, float3 dir, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_TERRAIN) == 0)
    {
        float2 t = ray_sphere(origin, dir, PLANET_RADIUS);
        return t.x > 0.0 ? t.x : SURFACE_MISS;
    }
    // The terrain lies inside the atmosphere, so the march spans the atmosphere
    // segment only; past it the original marched on to a miss.
    float2 atmosphere = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    if (atmosphere.y <= 0.0)
        return SURFACE_MISS;
    float entry = max(atmosphere.x, 0.0);
    float t = entry;
    for (uint step = 0; step < SPHERE_TRACE_STEPS; step++)
    {
        float d = land_sdf(origin + t * dir, scene);
        t += d;
        if (t > atmosphere.y || abs(d) < (t - entry) * SPHERE_TRACE_RELATIVE_EPSILON)
            break;
    }
    return t <= atmosphere.y ? t : SURFACE_MISS;
}

// Whether the sun direction `dir` is unoccluded by the surface from `origin`.
public bool surface_visible(float3 origin, float3 dir, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_TERRAIN) != 0)
        return intersect_surface(origin, dir, scene) < 0.0;
    return ray_sphere(origin, dir, PLANET_RADIUS).x < 0.0;
}

// The surface lifted along its normal far enough to start shadow and bounce rays
// without self-intersection.
public float3 offset_surface_point(float3 position, SceneUniforms scene)
{
    return position * (1.0 + 1e-4 * scene.surface.y / 12000.0);
}

public SurfaceSample sample_surface(float3 position, SceneUniforms scene)
{
    float3 up = normalize(position);
    SurfaceSample s;
    s.normal = (scene.config.x & SCENE_TERRAIN) != 0 ? land_normal(position, scene) : up;
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
    {
        float2 uv = sphere_uv(up);
        s.ocean = sample_tiled(earth_ocean_mask, earth_maps.ocean_mask, uv);
        s.albedo_srgb = grade_albedo(sample_tiled(earth_albedo, earth_maps.albedo, uv).rgb, s.ocean);
        s.bathymetry = sample_tiled(earth_bathymetry, earth_maps.bathymetry, uv);
        s.emissive = sample_tiled(earth_night_lights, earth_maps.night_lights, uv);
    }
    else
    {
        s.albedo_srgb = scene.surface.xxx;
        s.ocean = 0.0;
        s.bathymetry = 1.0;
        s.emissive = 0.0;
    }
    return s;
}

// Reflectance of `s` with spectral `albedo` for unit `v` and `l` pointing away
// from the surface. Without the Earth maps the ground is Lambertian at half its
// albedo, the shading the symmetric scene (and the MLP trained on it) used.
public SurfaceResponse surface_brdf(SurfaceSample s, float albedo, float3 v, float3 l, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
        return earth_brdf(albedo, s.ocean, s.bathymetry, v, s.normal, l);
    SurfaceResponse response;
    response.brdf = albedo / PI * 0.5;
    response.n_dot_l = saturate(dot(s.normal, l));
    return response;
}

// A bounce direction from unit `v` off `s`, sampled in proportion to its
// reflectance where the BRDF has a sharp lobe.
public BounceSample sample_bounce(SurfaceSample s, float3 v, float3 u, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_EARTH_MAPS) != 0)
        return sample_earth_bounce(u, s.ocean, s.bathymetry, v, s.normal);
    BounceSample bounce;
    bounce.dir = sample_hemisphere_cosine(u.xy, s.normal);
    bounce.pdf = saturate(dot(s.normal, bounce.dir)) / PI;
    return bounce;
}

// Spectral radiance the night lights of `s` emit.
public float surface_emission(SurfaceSample s, float wavelength, SceneUniforms scene)
{
    if ((scene.config.x & SCENE_NIGHT_LIGHTS) == 0)
        return 0.0;
    return s.emissive * planck(NIGHT_LIGHT_TEMPERATURE, wavelength) * scene.surface.z;
}
