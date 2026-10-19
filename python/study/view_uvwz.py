"""Per-pixel MLP inputs of a camera view, as the renderer's `mlp` mode computes them.

Ports the host side (`viewer/Config.zig` defaults and presets, `viewer/Camera.zig`
basis, `render/View.zig` frame uniforms) and the shader side
(`shared/bindings.slang` camera rays, `atmosphere/bruneton.slang`, `modes/neural.slang`)
in float32, the precision of the uniforms and the shaders.

Images are top row first, as PNGs are; the renderer's `accum` and `--out-linear`
PFM are bottom row first (`read_pfm` flips them).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# shaders/common/constants.slang
PLANET_RADIUS = np.float32(6371e3)
ATMOSPHERE_RADIUS = np.float32(6371e3 + 110e3)
# shaders/atmosphere/bruneton.slang
SCATTERING_R_SIZE = np.float32(4096.0)
SCATTERING_MU_HALF_SIZE = np.float32(2048.0)
SCATTERING_MU_S_SIZE = np.float32(4096.0)
MU_S_MIN = np.float32(-0.2)
H = np.sqrt(ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - PLANET_RADIUS * PLANET_RADIUS)
# shaders/spectral/cie.slang: the display adds these stops to the config's exposure.
ORIGINAL_ESTIMATOR_STOPS = -7.406


@dataclass(frozen=True)
class Config:
    """The fields of `viewer/Config.zig` that set the image; defaults are its defaults."""

    position: tuple[float, float, float] = (-15e6, 0.0, 15e6)
    forward: tuple[float, float, float] = (np.sqrt(0.5), 0.0, -np.sqrt(0.5))
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_fov_degrees: float = 27.0
    aspect_scale: float = 1.0
    sun_angle_degrees: float = 60.0
    sun_path_rotation_degrees: float = -45.0
    exposure: float = 2.5


_ZON_FIELDS = {
    "position": "position",
    "forward": "forward",
    "up": "up",
    "vertical_fov_degrees": "vertical_fov_degrees",
    "aspect_scale": "aspect_scale",
    "angle_degrees": "sun_angle_degrees",
    "path_rotation_degrees": "sun_path_rotation_degrees",
    "exposure": "exposure",
}


def load_config(path: Path) -> Config:
    """A `--config` file: missing fields keep their defaults, directions renormalize."""
    text = path.read_text()
    values = {}
    for zon_name, field in _ZON_FIELDS.items():
        match = re.search(rf"\.{zon_name}\s*=\s*(\.\{{[^}}]*\}}|[-+0-9.eE]+)", text)
        if match is None:
            continue
        literal = match.group(1)
        if literal.startswith(".{"):
            values[field] = tuple(float(x) for x in literal[2:-1].split(",") if x.strip())
        else:
            values[field] = float(literal)
    for direction in ("forward", "up"):
        if direction in values:
            vector = np.array(values[direction])
            values[direction] = tuple(vector / np.linalg.norm(vector))
    return Config(**values)


def _limb_config() -> Config:
    """From 400 km, looking along the horizon; every pixel's ray starts inside the atmosphere."""
    radius = 6371e3 + 400e3
    dip = np.arccos(6371e3 / radius)
    return Config(position=(0.0, radius, 0.0), forward=(np.cos(dip), -np.sin(dip), 0.0))


VIEWS = {
    "default": Config(),
    "limb": _limb_config(),
    "sunset_limb": load_config(REPO / "presets" / "sunset-hurricane.zon"),
}


@dataclass(frozen=True)
class FrameUniforms:
    """`FrameUniforms` (shared/uniforms.slang) as float32 3-vectors plus the `.w` scalars."""

    camera_position: np.ndarray
    camera_forward: np.ndarray
    half_height: np.float32
    camera_right: np.ndarray
    aspect: np.float32
    camera_up: np.ndarray
    sun_direction: np.ndarray


def frame_uniforms(config: Config, width: int, height: int) -> FrameUniforms:
    """`Camera.view` then `View.writeUniforms`, in float64 and rounded to float32 at the end."""
    forward = np.array(config.forward)
    up_reference = np.array(config.up)
    # Camera.left; looking along `up` falls back to the world axis least aligned with the view.
    if abs(np.dot(up_reference, forward)) <= 0.999:
        left = np.cross(up_reference, forward)
    else:
        axis = np.zeros(3)
        axis[np.argmin(np.abs(forward))] = 1.0
        left = np.cross(axis, forward)
    right = -left / np.linalg.norm(left)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    angle = np.radians(config.sun_angle_degrees)
    rotation = np.radians(config.sun_path_rotation_degrees)
    sun = np.array([-np.sin(angle), np.cos(angle) * -np.sin(rotation), np.cos(angle) * np.cos(rotation)])

    f32 = np.float32
    return FrameUniforms(
        camera_position=np.array(config.position, dtype=f32),
        camera_forward=forward.astype(f32),
        half_height=f32(np.tan(np.radians(config.vertical_fov_degrees) / 2)),
        camera_right=right.astype(f32),
        aspect=f32(width / height * config.aspect_scale),
        camera_up=up.astype(f32),
        sun_direction=sun.astype(f32),
    )


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _safe_sqrt(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(a, np.float32(0.0)))


def _texel_coord(x: np.ndarray, size: np.float32) -> np.ndarray:
    return np.float32(0.5) / size + x * (np.float32(1.0) - np.float32(1.0) / size)


def camera_rays(uniforms: FrameUniforms, width: int, height: int, jitter: np.ndarray | None = None) -> np.ndarray:
    """`PixelSample.camera_ray_direction`, [H, W, 3] top row first.

    `jitter` is the [H, W, 2] sub-pixel offset in the shader's bottom-up pixel
    order; None takes the pixel centre.
    """
    f32 = np.float32
    pixel = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(f32)
    offset = np.full_like(pixel, 0.5) if jitter is None else jitter.astype(f32)
    plane = f32(2.0) * (pixel + offset) / np.array([width, height], dtype=f32) - f32(1.0)
    half_width = uniforms.half_height * uniforms.aspect
    direction = _normalize(
        uniforms.camera_forward
        + (plane[..., 0:1] * half_width) * uniforms.camera_right
        + (plane[..., 1:2] * uniforms.half_height) * uniforms.camera_up
    )
    return direction[::-1]


def enter_atmosphere(pos: np.ndarray, dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rays from outside move onto the top of the atmosphere; returns (pos, enters)."""
    pos = np.broadcast_to(pos, dir.shape)
    inside = np.linalg.norm(pos, axis=-1) <= ATMOSPHERE_RADIUS
    b = _dot(pos, dir)
    discriminant = b * b - _dot(pos, pos) + ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS
    t = np.where(discriminant < 0, np.float32(-1.0), -b - _safe_sqrt(discriminant))
    moved = pos + np.maximum(t, np.float32(0.0))[..., None] * dir
    return np.where(inside[..., None], pos, moved), inside | (t > 0)


def scattering_params_from_ray(pos: np.ndarray, dir: np.ndarray, sun: np.ndarray) -> tuple[np.ndarray, ...]:
    """(r, mu, mu_s, nu)."""
    r = np.linalg.norm(pos, axis=-1)
    return r, _dot(pos, dir) / r, _dot(pos, sun) / r, _dot(dir, sun)


def canonical_ray(r, mu, mu_s, nu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pos, dir, sun) of the canonical frame: origin on +Y, sun in the XY plane."""
    f32 = np.float32
    sin_theta = _safe_sqrt(f32(1.0) - mu * mu)
    sin_theta_s = _safe_sqrt(f32(1.0) - mu_s * mu_s)
    defined = (np.abs(sin_theta) > f32(1e-8)) & (np.abs(sin_theta_s) > f32(1e-8))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = np.clip((nu - mu * mu_s) / (sin_theta * sin_theta_s), f32(-1.0), f32(1.0))
    cos_phi = np.where(defined, cos_phi, f32(0.0))
    sin_phi = _safe_sqrt(f32(1.0) - cos_phi * cos_phi)
    zero = np.zeros_like(r)
    pos = np.stack([zero, r, zero], axis=-1)
    dir = _normalize(np.stack([sin_theta * cos_phi, mu, sin_theta * sin_phi], axis=-1))
    sun = _normalize(np.stack([sin_theta_s, mu_s, zero], axis=-1))
    return pos, dir, sun


def _distance_to_atmosphere_top(r, mu):
    discriminant = r * r * (mu * mu - np.float32(1.0)) + ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS
    return np.maximum(-r * mu + _safe_sqrt(discriminant), np.float32(0.0))


def scattering_uvwz(r, mu, mu_s, nu) -> np.ndarray:
    """(u_nu, u_mu_s, u_mu, u_r), the network's input order."""
    f32 = np.float32
    rho = _safe_sqrt(r * r - PLANET_RADIUS * PLANET_RADIUS)
    u_r = _texel_coord(rho / H, SCATTERING_R_SIZE)

    r_mu = r * mu
    discriminant = r_mu * r_mu - r * r + PLANET_RADIUS * PLANET_RADIUS
    intersects_ground = (mu < 0) & (r * r * (mu * mu - f32(1.0)) + PLANET_RADIUS * PLANET_RADIUS >= 0)
    # Ground branch.
    d = -r_mu - _safe_sqrt(discriminant)
    d_min = r - PLANET_RADIUS
    d_max = rho
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(d_max == d_min, f32(0.0), (d - d_min) / (d_max - d_min))
    u_mu_ground = f32(0.5) - f32(0.5) * _texel_coord(x, SCATTERING_MU_HALF_SIZE)
    # Sky branch.
    d = -r_mu + _safe_sqrt(discriminant + H * H)
    d_min = ATMOSPHERE_RADIUS - r
    d_max = rho + H
    u_mu_sky = f32(0.5) + f32(0.5) * _texel_coord((d - d_min) / (d_max - d_min), SCATTERING_MU_HALF_SIZE)
    u_mu = np.where(intersects_ground, u_mu_ground, u_mu_sky)

    d_min = ATMOSPHERE_RADIUS - PLANET_RADIUS
    d_max = H
    a = (_distance_to_atmosphere_top(PLANET_RADIUS, mu_s) - d_min) / (d_max - d_min)
    A = (_distance_to_atmosphere_top(PLANET_RADIUS, MU_S_MIN) - d_min) / (d_max - d_min)
    u_mu_s = _texel_coord(np.maximum(f32(1.0) - a / A, f32(0.0)) / (f32(1.0) + a), SCATTERING_MU_S_SIZE)

    u_nu = (nu + f32(1.0)) / f32(2.0)
    return np.stack([u_nu, u_mu_s, u_mu, u_r], axis=-1).astype(f32)


def reachable(uvwz: np.ndarray) -> np.ndarray:
    """Whether a camera ray can produce uvwz [..., 4]: scattering_uvwz maps the ground
    and sky halves of mu into texel-centre ranges, leaving out |u_mu - 0.5| < 1/8192,
    whose inverse clamps to unrelated, often dark rays."""
    return np.abs(uvwz[..., 2] - 0.5) >= 0.25 / SCATTERING_MU_HALF_SIZE


def _texel_unit(u: np.ndarray, size: np.float32) -> np.ndarray:
    return (u - np.float32(0.5) / size) / (np.float32(1.0) - np.float32(1.0) / size)


def scattering_params_from_uvwz(uvwz: np.ndarray) -> tuple[np.ndarray, ...]:
    """(r, mu, mu_s, nu) of uvwz [..., 4], nu clamped to the range mu and mu_s allow,
    as `scattering_params_from_uvwz` in bruneton.slang (what datagen evaluates)."""
    f32 = np.float32
    uvwz = uvwz.astype(f32)
    rho = H * _texel_unit(uvwz[..., 3], SCATTERING_R_SIZE)
    r = np.sqrt(rho * rho + PLANET_RADIUS * PLANET_RADIUS)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_min = r - PLANET_RADIUS
        d = d_min + (rho - d_min) * _texel_unit(f32(1.0) - f32(2.0) * uvwz[..., 2], SCATTERING_MU_HALF_SIZE)
        mu_ground = np.where(d == 0, f32(-1.0), np.clip(-(rho * rho + d * d) / (f32(2.0) * r * d), -1.0, 1.0))
        d_min = ATMOSPHERE_RADIUS - r
        d = d_min + (rho + H - d_min) * _texel_unit(f32(2.0) * uvwz[..., 2] - f32(1.0), SCATTERING_MU_HALF_SIZE)
        mu_sky = np.where(d == 0, f32(1.0), np.clip((H * H - rho * rho - d * d) / (f32(2.0) * r * d), -1.0, 1.0))
    mu = np.where(uvwz[..., 2] < 0.5, mu_ground, mu_sky).astype(f32)

    x_mu_s = _texel_unit(uvwz[..., 1], SCATTERING_MU_S_SIZE)
    d_min = ATMOSPHERE_RADIUS - PLANET_RADIUS
    A = (_distance_to_atmosphere_top(PLANET_RADIUS, MU_S_MIN) - d_min) / (H - d_min)
    a = (A - x_mu_s * A) / (f32(1.0) + x_mu_s * A)
    d = d_min + np.minimum(a, A) * (H - d_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_s = np.where(d == 0, f32(1.0), np.clip((H * H - d * d) / (f32(2.0) * PLANET_RADIUS * d), -1.0, 1.0))
    mu_s = mu_s.astype(f32)

    nu = np.clip(uvwz[..., 0] * f32(2.0) - f32(1.0), -1.0, 1.0)
    spread = np.sqrt((f32(1.0) - mu * mu) * (f32(1.0) - mu_s * mu_s))
    return r, mu, mu_s, np.clip(nu, mu * mu_s - spread, mu * mu_s + spread).astype(f32)


def uvwz_from_uniforms(
    uniforms: FrameUniforms, width: int, height: int, jitter: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`mlp_main` up to the network input: (uvwz float32 [H, W, 4], enters bool [H, W])."""
    dir = camera_rays(uniforms, width, height, jitter)
    pos, enters = enter_atmosphere(uniforms.camera_position, dir)
    # The canonical round trip clamps nu to the range mu and mu_s allow.
    world = scattering_params_from_ray(pos, dir, uniforms.sun_direction)
    canonical = canonical_ray(*world)
    return scattering_uvwz(*scattering_params_from_ray(*canonical)), enters


def view_uvwz(view_name: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre network inputs of a named view (`VIEWS`), top row first."""
    return uvwz_from_uniforms(frame_uniforms(VIEWS[view_name], width, height), width, height)


def tonemap(linear_rgb: np.ndarray, exposure: float) -> np.ndarray:
    """Linear Rec.709 to sRGB-encoded [0, 1] with the config's exposure (stops).

    The renderer's display also applies vignette, OpenDRT, and the camera
    response; this keeps only its exposure scale, a clamp, and the sRGB OETF.
    """
    x = np.clip(linear_rgb * np.exp2(exposure + ORIGINAL_ESTIMATOR_STOPS), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def read_pfm(path: Path) -> np.ndarray:
    """An `--out-linear` PFM as float32 [H, W, 3], top row first."""
    data = path.read_bytes()
    header = data.split(b"\n", 3)
    assert header[0] == b"PF", "not an RGB PFM"
    width, height = (int(v) for v in header[1].split())
    scale = float(header[2])
    dtype = "<f4" if scale < 0 else ">f4"
    pixels = np.frombuffer(header[3], dtype=dtype, count=width * height * 3).reshape(height, width, 3)
    return pixels[::-1].astype(np.float32)
