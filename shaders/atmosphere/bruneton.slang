// Bruneton and Neyret 2008 parameterizations: the 4D scattering domain
// (r, mu, mu_s, nu) <-> uvwz that the MLP takes as input, and the 2D
// transmittance domain (mu, r) <-> uv.
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html

module bruneton;

import common.constants;
import common.geometry;

// The uvwz texel counts the MLP was trained with; mu splits in two halves.
static const float SCATTERING_R_SIZE = 4096.0;
static const float SCATTERING_MU_HALF_SIZE = 2048.0;
static const float SCATTERING_MU_S_SIZE = 4096.0;

// The uvwz that scattering_uvwz reaches at the ends of each domain, where the
// unit ranges are 0 and 1; src/assets/datagen_format.zig mirrors them. u spans
// [0, 1] (nu from -1 to 1), v [V_LOW, V_HIGH] (mu_s from MU_S_MIN to 1), and z
// [Z_LOW, Z_HIGH] (r from the ground to the top of the atmosphere). w spans
// [W_GROUND_LOW, W_GROUND_HIGH] on rays that hit the ground, horizon to nadir,
// and [W_SKY_LOW, W_SKY_HIGH] on the others, zenith to horizon.
public static const float V_LOW = 0.5 / SCATTERING_MU_S_SIZE;
public static const float V_HIGH = 1.0 - V_LOW;
public static const float W_GROUND_LOW = 0.25 / SCATTERING_MU_HALF_SIZE;
public static const float W_GROUND_HIGH = 0.5 - W_GROUND_LOW;
public static const float W_SKY_LOW = 0.5 + W_GROUND_LOW;
public static const float W_SKY_HIGH = 1.0 - W_GROUND_LOW;
public static const float Z_LOW = 0.5 / SCATTERING_R_SIZE;
public static const float Z_HIGH = 1.0 - Z_LOW;

// Lowest sun cosine the uvwz mapping distinguishes; lower suns map to it. It
// defines the MLP's input domain, so the night side is lit as at -11.5 degrees.
public static const float MU_S_MIN = -0.2;

// Distance from the ground to the top of the atmosphere along the horizon.
static const float H = sqrt(ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - PLANET_RADIUS * PLANET_RADIUS);

float safe_sqrt(float a)
{
    return sqrt(max(a, 0.0));
}

float unit_range_to_texel_coord(float x, float size)
{
    return 0.5 / size + x * (1.0 - 1.0 / size);
}

float texel_coord_to_unit_range(float u, float size)
{
    return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

public float distance_to_atmosphere_top(float r, float mu)
{
    float discriminant = r * r * (mu * mu - 1.0) + ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS;
    return max(-r * mu + safe_sqrt(discriminant), 0.0);
}

public bool ray_intersects_ground(float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + PLANET_RADIUS * PLANET_RADIUS >= 0.0;
}

// Moves `pos` along the unit `dir` onto the top of the atmosphere when it
// starts above it, since r clamps there; false when the ray misses the atmosphere.
public bool enter_atmosphere(inout float3 pos, float3 dir)
{
    if (length(pos) <= ATMOSPHERE_RADIUS)
        return true;
    float t = ray_sphere(pos, dir, ATMOSPHERE_RADIUS).x;
    pos += max(t, 0.0) * dir;
    return t > 0.0;
}

public struct ScatteringParams
{
    public float r;
    public float mu;
    public float mu_s;
    public float nu;
    public bool intersects_ground;
};

public ScatteringParams scattering_params_from_ray(float3 pos, float3 dir, float3 sun)
{
    ScatteringParams p;
    p.r = length(pos);
    p.mu = dot(pos, dir) / p.r;
    p.mu_s = dot(pos, sun) / p.r;
    p.nu = dot(dir, sun);
    p.intersects_ground = ray_intersects_ground(p.r, p.mu);
    return p;
}

// A ray with the given parameters in a canonical frame: origin on +Y, view
// direction rotated about Y from the XY plane, sun in the XY plane.
public struct CanonicalRay
{
    public float3 pos;
    public float3 dir;
    public float3 sun;
};

public CanonicalRay canonical_ray(float r, float mu, float mu_s, float nu)
{
    float sin_theta = safe_sqrt(1.0 - mu * mu);
    float sin_theta_s = safe_sqrt(1.0 - mu_s * mu_s);
    float cos_phi = 0.0;
    if (abs(sin_theta) > 1e-8 && abs(sin_theta_s) > 1e-8)
        cos_phi = clamp((nu - mu * mu_s) / (sin_theta * sin_theta_s), -1.0, 1.0);
    float sin_phi = safe_sqrt(1.0 - cos_phi * cos_phi);

    CanonicalRay ray;
    ray.pos = float3(0.0, r, 0.0);
    ray.dir = normalize(float3(sin_theta * cos_phi, mu, sin_theta * sin_phi));
    ray.sun = normalize(float3(sin_theta_s, mu_s, 0.0));
    return ray;
}

// Returns (u_nu, u_mu_s, u_mu, u_r), the component order the MLP was trained on.
public float4 scattering_uvwz(ScatteringParams p)
{
    float rho = safe_sqrt(p.r * p.r - PLANET_RADIUS * PLANET_RADIUS);
    float u_r = unit_range_to_texel_coord(rho / H, SCATTERING_R_SIZE);

    float r_mu = p.r * p.mu;
    float discriminant = r_mu * r_mu - p.r * p.r + PLANET_RADIUS * PLANET_RADIUS;
    float u_mu;
    if (p.intersects_ground)
    {
        float d = -r_mu - safe_sqrt(discriminant);
        float d_min = p.r - PLANET_RADIUS;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * unit_range_to_texel_coord(d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min),
                                                     SCATTERING_MU_HALF_SIZE);
    }
    else
    {
        float d = -r_mu + safe_sqrt(discriminant + H * H);
        float d_min = ATMOSPHERE_RADIUS - p.r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * unit_range_to_texel_coord((d - d_min) / (d_max - d_min), SCATTERING_MU_HALF_SIZE);
    }

    float d_min = ATMOSPHERE_RADIUS - PLANET_RADIUS;
    float d_max = H;
    float a = (distance_to_atmosphere_top(PLANET_RADIUS, p.mu_s) - d_min) / (d_max - d_min);
    float A = (distance_to_atmosphere_top(PLANET_RADIUS, MU_S_MIN) - d_min) / (d_max - d_min);
    float u_mu_s = unit_range_to_texel_coord(max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_MU_S_SIZE);

    float u_nu = (p.nu + 1.0) / 2.0;
    return float4(u_nu, u_mu_s, u_mu, u_r);
}

// The uvwz of a camera ray from `pos` in the symmetric scene, where radiance
// depends on (r, mu, mu_s, nu) alone; false when the ray misses the atmosphere.
public bool camera_ray_uvwz(float3 pos, float3 dir, float3 sun, out float4 uvwz)
{
    bool enters = enter_atmosphere(pos, dir);
    // The original mapped the canonical ray of (r, mu, mu_s, nu), not the
    // world ray; the round trip clamps nu to the range mu and mu_s allow.
    ScatteringParams world = scattering_params_from_ray(pos, dir, sun);
    CanonicalRay ray = canonical_ray(world.r, world.mu, world.mu_s, world.nu);
    uvwz = scattering_uvwz(scattering_params_from_ray(ray.pos, ray.dir, ray.sun));
    return enters;
}

public ScatteringParams scattering_params_from_uvwz(float4 uvwz)
{
    ScatteringParams p;
    float rho = H * texel_coord_to_unit_range(uvwz.w, SCATTERING_R_SIZE);
    p.r = sqrt(rho * rho + PLANET_RADIUS * PLANET_RADIUS);

    p.intersects_ground = uvwz.z < 0.5;
    if (p.intersects_ground)
    {
        float d_min = p.r - PLANET_RADIUS;
        float d_max = rho;
        float d = d_min + (d_max - d_min) * texel_coord_to_unit_range(1.0 - 2.0 * uvwz.z, SCATTERING_MU_HALF_SIZE);
        p.mu = d == 0.0 ? -1.0 : clamp(-(rho * rho + d * d) / (2.0 * p.r * d), -1.0, 1.0);
    }
    else
    {
        float d_min = ATMOSPHERE_RADIUS - p.r;
        float d_max = rho + H;
        float d = d_min + (d_max - d_min) * texel_coord_to_unit_range(2.0 * uvwz.z - 1.0, SCATTERING_MU_HALF_SIZE);
        p.mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * p.r * d), -1.0, 1.0);
    }

    float x_mu_s = texel_coord_to_unit_range(uvwz.y, SCATTERING_MU_S_SIZE);
    float d_min = ATMOSPHERE_RADIUS - PLANET_RADIUS;
    float d_max = H;
    float A = (distance_to_atmosphere_top(PLANET_RADIUS, MU_S_MIN) - d_min) / (d_max - d_min);
    float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    float d = d_min + min(a, A) * (d_max - d_min);
    p.mu_s = d == 0.0 ? 1.0 : clamp((H * H - d * d) / (2.0 * PLANET_RADIUS * d), -1.0, 1.0);

    // nu is clamped to the range reachable from mu and mu_s.
    float nu = clamp(uvwz.x * 2.0 - 1.0, -1.0, 1.0);
    float spread = sqrt((1.0 - p.mu * p.mu) * (1.0 - p.mu_s * p.mu_s));
    p.nu = clamp(nu, p.mu * p.mu_s - spread, p.mu * p.mu_s + spread);
    return p;
}

// Transmittance LUT coordinates for a ray at radius `r` with zenith cosine
// `mu`, on a LUT of `size` texels. The LUT stores only rays that reach the
// top of the atmosphere; callers test ray_intersects_ground first.
public float2 transmittance_uv(float mu, float r, float2 size)
{
    float rho = safe_sqrt(r * r - PLANET_RADIUS * PLANET_RADIUS);
    float d_min = ATMOSPHERE_RADIUS - r;
    float d_max = rho + H;
    float x_mu = (distance_to_atmosphere_top(r, mu) - d_min) / (d_max - d_min);
    return float2(unit_range_to_texel_coord(x_mu, size.x), unit_range_to_texel_coord(rho / H, size.y));
}

// Inverse of transmittance_uv, returning (mu, r). The bake evaluates it at
// texel centres, uv = (texel + 0.5) / size.
public float2 transmittance_mu_r(float2 uv, float2 size)
{
    float x_mu = texel_coord_to_unit_range(uv.x, size.x);
    float rho = H * texel_coord_to_unit_range(uv.y, size.y);
    float r = sqrt(rho * rho + PLANET_RADIUS * PLANET_RADIUS);
    float d_min = ATMOSPHERE_RADIUS - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    float mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    return float2(clamp(mu, -1.0, 1.0), r);
}
