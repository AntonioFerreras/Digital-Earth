import numpy as np
import matplotlib.pyplot as plt

# Constants from volume_rendering_models.py
planet_r = 6371e3
atmos_height = 110e3
atmos_upper_limit = planet_r + atmos_height

# Constants from bruneton_mappings.py
mu_s_min = -0.2
SCATTERING_TEXTURE_R_SIZE = 4096
SCATTERING_TEXTURE_MU_SIZE = 4096
SCATTERING_TEXTURE_MU_S_SIZE = 4096
SCATTERING_TEXTURE_NU_SIZE = 4096

# Helper functions converted from Taichi to Python
def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)

def ClampCosine(mu):
    return clamp(mu, -1.0, 1.0)

def ClampDistance(d):
    return max(d, 0.0)

def ClampRadius(r):
    return clamp(r, planet_r, atmos_upper_limit)

def SafeSqrt(a):
    return np.sqrt(max(a, 0.0))

def GetTextureCoordFromUnitRange(x, texture_size):
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size))

def GetUnitRangeFromTextureCoord(u, texture_size):
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size))

def DistanceToTopAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + atmos_upper_limit * atmos_upper_limit
    return ClampDistance(-r * mu + SafeSqrt(discriminant))

def DistanceToBottomAtmosphereBoundary(r, mu):
    discriminant = r * r * (mu * mu - 1.0) + planet_r * planet_r
    return ClampDistance(-r * mu - SafeSqrt(discriminant))

def RayIntersectsGround(r, mu):
    return mu < 0.0 and r * r * (mu * mu - 1.0) + planet_r * planet_r >= 0.0

def mu_s_mapping(x):
    # Add safeguards but keep the power function
    normalized = clamp((x - mu_s_min) / (1.0 - mu_s_min), 0.0, 1.0)
    # Avoid exact zeros that might cause issues elsewhere
    if normalized < 1e-6:
        normalized = 1e-6
    return pow(normalized, 0.85)

def inverse_mu_s_mapping(y):
    # Add similar safeguards for the inverse
    y_safe = clamp(y, 1e-6, 1.0)
    return mu_s_min + (1.0 - mu_s_min) * pow(y_safe, 1.0 / 0.85)

def GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground):
    # Distance to top atmosphere boundary for a horizontal ray at ground level.
    H = np.sqrt(atmos_upper_limit * atmos_upper_limit - planet_r * planet_r)
    
    # Distance to the horizon.
    rho = SafeSqrt(r * r - planet_r * planet_r)
    u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE)

    # Discriminant of the quadratic equation for the intersections of the ray
    # (r,mu) with the ground (see RayIntersectsGround).
    r_mu = r * mu
    discriminant = r_mu * r_mu - r * r + planet_r * planet_r
    u_mu = 0.0
    
    if ray_r_mu_intersects_ground:
        # Distance to the ground for the ray (r,mu), and its minimum and maximum
        # values over all mu - obtained for (r,-1) and (r,mu_horizon).
        d = -r_mu - SafeSqrt(discriminant)
        d_min = r - planet_r
        d_max = rho
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
            0.0 if d_max == d_min else (d - d_min) / (d_max - d_min), 
            SCATTERING_TEXTURE_MU_SIZE / 2)
    else:
        # Distance to the top atmosphere boundary for the ray (r,mu), and its
        # minimum and maximum values over all mu - obtained for (r,1) and
        # (r,mu_horizon).
        d = -r_mu + SafeSqrt(discriminant + H * H)
        d_min = atmos_upper_limit - r
        d_max = rho + H
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2)

    # This is the part that maps mu_s to texture coordinate uvwz.y
    d = DistanceToTopAtmosphereBoundary(planet_r, mu_s)
    d_min = atmos_upper_limit - planet_r
    d_max = H
    a = (d - d_min) / (d_max - d_min)
    D = DistanceToTopAtmosphereBoundary(planet_r, mu_s_min)
    A = (D - d_min) / (d_max - d_min)
    
    # An ad-hoc function equal to 0 for mu_s = mu_s_min (because then d = D and
    # thus a = A), equal to 1 for mu_s = 1 (because then d = d_min and thus
    # a = 0), and with a large slope around mu_s = 0, to get more texture 
    # samples near the horizon.
    u_mu_s = GetTextureCoordFromUnitRange(
        max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE)

    u_nu = (nu + 1.0) / 2.0
    
    # Return a list instead of vec4 since we're not using Taichi
    return [u_nu, u_mu_s, u_mu, u_r]

# Test the function with various mu_s values
def test_uvwz_y_for_mu_s():
    # Fixed parameters for testing
    r = planet_r + 1000.0  # 1km above ground
    mu = 0.5               # Looking upward at an angle
    nu = 0.0               # Perpendicular to sun
    ray_intersects_ground = RayIntersectsGround(r, mu)
    
    # Test a range of mu_s values from mu_s_min to 1.0
    mu_s_values = np.linspace(mu_s_min, 1.0, 100)
    uvwz_y_values = []
    
    print(f"Testing mu_s mapping to uvwz.y (texture coordinate for sun zenith angle)")
    print(f"{'mu_s':<10} | {'uvwz.y':<10}")
    print("-" * 23)
    
    for mu_s in mu_s_values:
        uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_intersects_ground)
        uvwz_y = uvwz[1]  # The y component of uvwz
        uvwz_y_values.append(uvwz_y)
        
        # Print values at regular intervals
        if len(uvwz_y_values) % 10 == 1:
            print(f"{mu_s:<10.4f} | {uvwz_y:<10.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(mu_s_values, uvwz_y_values)
    plt.title('Mapping of mu_s to uvwz.y (Texture Coordinate)')
    plt.xlabel('mu_s (cosine of sun zenith angle)')
    plt.ylabel('uvwz.y (texture coordinate)')
    plt.grid(True)
    plt.axvline(x=0.0, color='r', linestyle='--', label='Horizon (mu_s = 0)')
    plt.legend()
    plt.savefig('mu_s_mapping.png')
    print("\nGraph saved as 'mu_s_mapping.png'")
    
    return mu_s_values, uvwz_y_values

# Run the test
if __name__ == '__main__':
    mu_s_values, uvwz_y_values = test_uvwz_y_for_mu_s()
