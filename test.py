import numpy as np

def SafeSqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

def clamp(x, a, b):
    return np.minimum(np.maximum(x, a), b)

def test_reconstruction_with_rotation():
    EPSILON = 1e-5

    # Step 1: Random values in [0, 1]
    r = np.random.uniform(6371000.0, 6471000.0)
    mu = np.random.uniform(-1.0, 1.0)
    mu_s = np.random.uniform(-1.0, 1.0)
    nu = np.random.uniform(-1.0, 1.0)
    nu = clamp(nu, mu * mu_s - SafeSqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
        mu * mu_s + SafeSqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)))

    # Step 3: Construct initial vectors
    ray_pos = np.array([0.0, r, 0.0])

    sin_theta = SafeSqrt(1.0 - mu * mu)
    ray_dir = np.array([sin_theta, mu, 0.0])

    sin_theta_s = SafeSqrt(1.0 - mu_s * mu_s)
    sun_dir = np.array([sin_theta_s, mu_s, 0.0])

    # Step 4: Adjust ray_dir to match desired nu
    cos_phi = 0.0
    cos_phi = (nu - mu * mu_s) / (sin_theta * sin_theta_s)
    cos_phi = clamp(cos_phi, -1.0, 1.0)
    sin_phi = SafeSqrt(1.0 - cos_phi * cos_phi)

    ray_dir = np.array([
        ray_dir[0] * cos_phi,
        ray_dir[1],
        ray_dir[0] * sin_phi
    ])
    ray_dir /= np.linalg.norm(ray_dir)
    
    sun_dir /= np.linalg.norm(sun_dir)

    # Step 5: Recompute quantities
    r_prime = np.linalg.norm(ray_pos)
    mu_prime = np.dot(ray_pos, ray_dir) / r_prime
    mu_s_prime = np.dot(ray_pos, sun_dir) / r_prime
    nu_prime = np.dot(ray_dir, sun_dir)

    print(f"Original r = {r}, Reconstructed r = {r_prime}")
    print(f"Original mu = {mu}, Reconstructed mu = {mu_prime}")
    print(f"Original mu_s = {mu_s}, Reconstructed mu_s = {mu_s_prime}")
    print(f"Original nu = {nu}, Reconstructed nu = {nu_prime}")

    assert abs(r - r_prime) < EPSILON
    assert abs(mu - mu_prime) < EPSILON
    assert abs(mu_s - mu_s_prime) < EPSILON
    assert abs(nu - nu_prime) < EPSILON

test_reconstruction_with_rotation()
