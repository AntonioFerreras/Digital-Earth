import math

# Constants
planet_r = 6371e3
atmos_height = 110e3
atmos_upper_limit = planet_r + atmos_height
mu_s_min = -0.2

SCATTERING_TEXTURE_R_SIZE = 4096
SCATTERING_TEXTURE_MU_SIZE = 4096
SCATTERING_TEXTURE_MU_S_SIZE = 4096
SCATTERING_TEXTURE_NU_SIZE = 4096

# Helper functions
def clamp(x, a, b):
    return max(a, min(x, b))

def ClampCosine(mu):
    return clamp(mu, -1.0, 1.0)

def ClampDistance(d):
    return max(d, 0.0)

def ClampRadius(r):
    return clamp(r, planet_r, atmos_upper_limit)

def SafeSqrt(a):
    return math.sqrt(max(a, 0.0))

def GetTextureCoordFromUnitRange(x, texture_size):
    return 0.5 / texture_size + x * (1.0 - 1.0 / texture_size)

def GetUnitRangeFromTextureCoord(u, texture_size):
    return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size)

def DistanceToTopAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + atmos_upper_limit * atmos_upper_limit
    return ClampDistance(-r * mu + SafeSqrt(discriminant))

def DistanceToBottomAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + planet_r * planet_r
    return ClampDistance(-r * mu - SafeSqrt(discriminant))

# Main function
def GetRMuMuSNuFromScatteringTextureUvwz(u, v, z, w):
    H = math.sqrt(atmos_upper_limit ** 2 - planet_r ** 2)
    rho = H * GetUnitRangeFromTextureCoord(w, SCATTERING_TEXTURE_R_SIZE)
    r = math.sqrt(rho * rho + planet_r * planet_r)

    ray_r_mu_intersects_ground = False
    mu = 0.0

    if z < 0.5:
        d_min = r - planet_r
        d_max = rho
        d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            1.0 - 2.0 * z, SCATTERING_TEXTURE_MU_SIZE / 2)
        mu = -1.0 if d == 0.0 else ClampCosine(-(rho * rho + d * d) / (2.0 * r * d))
        ray_r_mu_intersects_ground = True
    else:
        d_min = atmos_upper_limit - r
        d_max = rho + H
        d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            2.0 * z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2)
        mu = 1.0 if d == 0.0 else ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d))
        ray_r_mu_intersects_ground = False

    x_mu_s = GetUnitRangeFromTextureCoord(v, SCATTERING_TEXTURE_MU_S_SIZE)
    d_min = atmos_upper_limit - planet_r
    d_max = H
    D = DistanceToTopAtmosphereBoundary(planet_r, mu_s_min)
    A = (D - d_min) / (d_max - d_min)
    a = (A - x_mu_s * A) / (1.0 + x_mu_s * A)
    d = d_min + min(a, A) * (d_max - d_min)
    mu_s = 1.0 if d == 0.0 else ClampCosine((H * H - d * d) / (2.0 * planet_r * d))

    nu = ClampCosine(u * 2.0 - 1.0)
    nu = clamp(nu, mu * mu_s - math.sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
                    mu * mu_s + math.sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)))

    return r, mu, mu_s, nu, ray_r_mu_intersects_ground

# Test calls
N = 100
for i in range(N):
    d = i / (N - 1)
    print(GetRMuMuSNuFromScatteringTextureUvwz(d, d, d, d))
