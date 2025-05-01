# import numpy as np

# def SafeSqrt(x):
#     return np.sqrt(np.maximum(x, 0.0))

# def clamp(x, a, b):
#     return np.minimum(np.maximum(x, a), b)

# def test_reconstruction_with_rotation():
#     EPSILON = 1e-5

#     # Step 1: Random values in [0, 1]
#     r = np.random.uniform(6371000.0, 6471000.0)
#     mu = np.random.uniform(-1.0, 1.0)
#     mu_s = np.random.uniform(-1.0, 1.0)
#     nu = np.random.uniform(-1.0, 1.0)
#     nu = clamp(nu, mu * mu_s - SafeSqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
#         mu * mu_s + SafeSqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)))

#     # Step 3: Construct initial vectors
#     ray_pos = np.array([0.0, r, 0.0])

#     sin_theta = SafeSqrt(1.0 - mu * mu)
#     ray_dir = np.array([sin_theta, mu, 0.0])

#     sin_theta_s = SafeSqrt(1.0 - mu_s * mu_s)
#     sun_dir = np.array([sin_theta_s, mu_s, 0.0])

#     # Step 4: Adjust ray_dir to match desired nu
#     cos_phi = 0.0
#     cos_phi = (nu - mu * mu_s) / (sin_theta * sin_theta_s)
#     cos_phi = clamp(cos_phi, -1.0, 1.0)
#     sin_phi = SafeSqrt(1.0 - cos_phi * cos_phi)

#     ray_dir = np.array([
#         ray_dir[0] * cos_phi,
#         ray_dir[1],
#         ray_dir[0] * sin_phi
#     ])
#     ray_dir /= np.linalg.norm(ray_dir)
    
#     sun_dir /= np.linalg.norm(sun_dir)

#     # Step 5: Recompute quantities
#     r_prime = np.linalg.norm(ray_pos)
#     mu_prime = np.dot(ray_pos, ray_dir) / r_prime
#     mu_s_prime = np.dot(ray_pos, sun_dir) / r_prime
#     nu_prime = np.dot(ray_dir, sun_dir)

#     print(f"Original r = {r}, Reconstructed r = {r_prime}")
#     print(f"Original mu = {mu}, Reconstructed mu = {mu_prime}")
#     print(f"Original mu_s = {mu_s}, Reconstructed mu_s = {mu_s_prime}")
#     print(f"Original nu = {nu}, Reconstructed nu = {nu_prime}")

#     assert abs(r - r_prime) < EPSILON
#     assert abs(mu - mu_prime) < EPSILON
#     assert abs(mu_s - mu_s_prime) < EPSILON
#     assert abs(nu - nu_prime) < EPSILON

# test_reconstruction_with_rotation()

import numpy as np

# ---------- forward map ------------------------------------------------------
def biased_smooth_map(x: float) -> float:
    """
    Smooth, strictly-increasing mapping from [-0.2, 1.0] to [0, 1].

       [-0.2, 0.1]  → [0.0, 0.4]
       (0.1,  1.0]  → (0.4, 1.0]
    """
    if not -0.2 <= x <= 1.0:
        raise ValueError("x must be in [-0.2, 1.0]")

    t     = (x + 0.2) / 1.2           # affine → t∈[0,1]
    split = 0.25                      # t for x = 0.1

    def S(s: float) -> float:         # cubic C¹ smooth-step
        return (3*s - 2*s*s) * s      # = 3s² − 2s³

    if t <= split:                    # first segment
        s = t / split
        return 0.4 * S(s)
    else:                             # second segment
        s = (t - split) / (1 - split)
        return 0.4 + 0.6 * S(s)

# ---------- inverse map ------------------------------------------------------
def inverse_biased_smooth_map(y: float, tol: float = 1e-12) -> float:
    """
    Inverse of biased_smooth_map on [0,1] → [-0.2, 1.0].

    Uses bisection (monotone cubic ⇒ one root in [0,1]).
    """
    if not 0.0 <= y <= 1.0:
        raise ValueError("y must be in [0, 1]")

    split = 0.25

    def S(s: float) -> float:
        return (3*s - 2*s*s) * s      # same cubic

    # helper: invert S(s)=v by bisection on s∈[0,1]
    def inv_S(v: float) -> float:
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            (lo, hi) = (mid, hi) if S(mid) < v else (lo, mid)
        return (lo + hi) / 2.0

    if y <= 0.4:                      # came from first segment
        v = y / 0.4
        s = inv_S(v)
        t = s * split
    else:                             # came from second segment
        v = (y - 0.4) / 0.6
        s = inv_S(v)
        t = split + s * (1 - split)

    return t * 1.2 - 0.2              # back to x

# ---------- round-trip tests -------------------------------------------------
xs = np.linspace(-0.2, 1.0, 2001)
ys = np.array([biased_smooth_map(float(x)) for x in xs])

err_forward_then_inverse = np.max(
    np.abs([inverse_biased_smooth_map(float(y)) for y in ys] - xs)
)

ys_grid = np.linspace(0.0, 1.0, 2001)
err_inverse_then_forward = np.max(
    np.abs([biased_smooth_map(inverse_biased_smooth_map(float(y))) for y in ys_grid] - ys_grid)
)

print(f"max |inv(f(x)) − x|  : {err_forward_then_inverse:.2e}")
print(f"max |f(inv(y)) − y| : {err_inverse_then_forward:.2e}")
