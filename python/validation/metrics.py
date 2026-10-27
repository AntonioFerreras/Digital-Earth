"""The validation suite's metrics (TESTING.md): of images, display and linear, and of
the cloud bench's probe fields, as `earth bake-probes` writes them."""

import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clouds.score import read_pfm  # noqa: E402

# Wang et al. 2004: an 11-tap Gaussian window of 1.5 pixels, constants for unit range.
SSIM_SIGMA = 1.5
SSIM_RADIUS = 5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2

# ProbeCache.zig's file: magic, then Key at format 5.
FIELD_MAGIC = b"EPROBE01"
FIELD_KEY_BYTES = 60
# SH3_COUNT in shaders/common/spherical_harmonics.slang.
SH_COUNT = 16
SKY_ORDER = 3
SUN_BIN_COUNT = 64
# The order of `fit_probes`' fit of the sun's log fluence.
SUN_ORDER = 5
# SUN_FLOOR in shaders/modes/cloud_bench_probes.slang.
SUN_FLOOR = 1e-5
SUN_DIRECTION_COUNT = 32


def read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0


def gaussian_filter(image: np.ndarray) -> np.ndarray:
    """SSIM's window over the first two axes, edges reflected."""
    taps = np.exp(-0.5 * (np.arange(-SSIM_RADIUS, SSIM_RADIUS + 1) / SSIM_SIGMA) ** 2).astype(np.float32)
    taps /= taps.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * image.ndim
        pad[axis] = (SSIM_RADIUS, SSIM_RADIUS)
        padded = np.pad(image, pad, mode="reflect")
        size = image.shape[axis]
        image = sum(w * np.take(padded, range(k, k + size), axis) for k, w in enumerate(taps))
    return image


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """The mean structural similarity of two images in [0, 1], over pixels and channels."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    mean_a, mean_b = gaussian_filter(a), gaussian_filter(b)
    var_a = gaussian_filter(a * a) - mean_a**2
    var_b = gaussian_filter(b * b) - mean_b**2
    covariance = gaussian_filter(a * b) - mean_a * mean_b
    similarity = ((2 * mean_a * mean_b + SSIM_C1) * (2 * covariance + SSIM_C2)) / (
        (mean_a**2 + mean_b**2 + SSIM_C1) * (var_a + var_b + SSIM_C2))
    return float(similarity.mean())


def image_metrics(test: Path, reference: Path) -> dict:
    """Of the display image `test`.png against `reference`.png."""
    shown, truth = read_png(test.with_suffix(".png")), read_png(reference.with_suffix(".png"))
    mse = float(np.mean((shown - truth) ** 2))
    return {"psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)), "ssim": ssim(shown, truth)}


class Field:
    """A probe field's sums per walk: the sky's transfer (probes, 16, 4) and the sun's
    bins (probes, 64, 4), x: phi, yzw: F."""

    def __init__(self, path: Path):
        raw = path.read_bytes()
        if raw[:8] != FIELD_MAGIC or struct.unpack("<I", raw[8:12])[0] != 4:
            raise SystemExit(f"{path}: not a probe field of format 5")
        key = raw[8 : 8 + FIELD_KEY_BYTES]
        self.layout = key[4:52]
        self.photons = bool(struct.unpack("<I", key[56:60])[0] & 2)
        self.walks = struct.unpack("<I", raw[8 + FIELD_KEY_BYTES : 12 + FIELD_KEY_BYTES])[0]
        body = np.frombuffer(raw, "<f4", offset=16 + FIELD_KEY_BYTES)
        count = body.size // ((SH_COUNT + SUN_BIN_COUNT) * 4)
        self.sky = body[: count * SH_COUNT * 4].reshape(count, SH_COUNT, 4).astype(np.float64) / self.walks
        self.bins = body[count * SH_COUNT * 4 :].reshape(count, SUN_BIN_COUNT, 4).astype(np.float64) / self.walks

    def second_half(self, half: "Field") -> "Field":
        """The walks this field adds to `half`, which it went on from."""
        rest = object.__new__(Field)
        rest.layout, rest.photons, rest.walks = self.layout, self.photons, self.walks - half.walks
        rest.sky = (self.sky * self.walks - half.sky * half.walks) / rest.walks
        rest.bins = (self.bins * self.walks - half.bins * half.walks) / rest.walks
        return rest

    def sun_fluence(self, directions: np.ndarray) -> np.ndarray:
        """phi(s) by `fit_probes`' fit before its quantisation: log-SH through order
        SUN_ORDER of the bins."""
        phi = self.bins[..., 0] * SUN_BIN_COUNT
        brightest = phi.max(-1, keepdims=True)
        log_phi = np.log(np.maximum(phi, SUN_FLOOR * np.maximum(brightest, 1e-30)))
        coefficients = log_phi @ sh_basis(BIN_CENTRES, SUN_ORDER) * (4.0 * math.pi / SUN_BIN_COUNT)
        return np.where(brightest > 0, np.exp(coefficients @ sh_basis(directions, SUN_ORDER).T), 0.0)

    def sky_fluence(self, sky: np.ndarray) -> np.ndarray:
        """phi under a sky of harmonics `sky`, as `fit_probes` contracts it before its
        quantisation: photon fields hold its scattered light in the bins."""
        phi = self.sky[..., 0] @ sky
        if self.photons:
            radiance = sh_basis(BIN_CENTRES, SKY_ORDER) @ sky
            phi = phi + 4.0 * math.pi * self.bins[..., 0] @ radiance
        return phi


def sh_basis(d: np.ndarray, order: int) -> np.ndarray:
    """shaders/common/spherical_harmonics.slang's real harmonics through `order`, at
    index l (l + 1) + m, by the same recurrence as its `sh5_basis`."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    basis = [None] * (order + 1) ** 2
    re, im, diagonal = np.ones_like(x), np.zeros_like(x), 1.0
    for m in range(order + 1):
        previous, current = np.zeros_like(x), np.zeros_like(x)
        for l in range(m, order + 1):
            p = diagonal * np.ones_like(x) if l == m else ((2 * l - 1) * z * current - (l + m - 1) * previous) / (l - m)
            previous, current = current, p
            scale = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) * (math.sqrt(2) if m else 1)
            basis[l * (l + 1) + m] = scale * p * re
            if m:
                basis[l * (l + 1) - m] = scale * p * im
        re, im = re * x - im * y, re * y + im * x
        diagonal *= 2 * m + 1
    return np.stack(basis, -1)


def equal_area_square_to_sphere(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """shaders/common/octahedral.slang's map (Clarberg)."""
    u, v = 2 * u - 1, 2 * v - 1
    d = 1 - (np.abs(u) + np.abs(v))
    r = 1 - np.abs(d)
    phi = np.where(r == 0, 1, (np.abs(v) - np.abs(u)) / np.where(r == 0, 1, r) + 1) * math.pi / 4
    z = np.sign(d) * (1 - r * r)
    s = np.sqrt(np.maximum(2 - r * r, 0))
    return np.stack([np.sign(u) * np.cos(phi) * r * s, np.sign(v) * np.sin(phi) * r * s, z], -1)


_centres = (np.arange(8) + 0.5) / 8
BIN_CENTRES = equal_area_square_to_sphere(*[a.ravel() for a in np.meshgrid(_centres, _centres)])


def fibonacci_sphere(count: int) -> np.ndarray:
    i = np.arange(count) + 0.5
    y = 1 - 2 * i / count
    radius = np.sqrt(1 - y * y)
    golden = math.pi * (3 - math.sqrt(5))
    return np.stack([radius * np.cos(golden * i), y, radius * np.sin(golden * i)], -1)


SUN_DIRECTIONS = fibonacci_sphere(SUN_DIRECTION_COUNT)
# A sky brighter overhead (+y), in harmonics through order 2.
TEST_SKY = np.array([1.0, 0.5] + [0.0] * (SH_COUNT - 2))


def relative_errors(test: np.ndarray, truth: np.ndarray) -> np.ndarray:
    lit = truth > 0
    return np.abs(test[lit] - truth[lit]) / truth[lit]


def field_metrics(test: Field, truth: Field) -> dict:
    """`test` against the converged `truth` of the same layout: relative errors over
    probes (and sun directions), and the bias by deciles of the truth's brightness."""
    sun_test, sun_truth = test.sun_fluence(SUN_DIRECTIONS), truth.sun_fluence(SUN_DIRECTIONS)
    sun = relative_errors(sun_test, sun_truth)
    sky = relative_errors(test.sky_fluence(TEST_SKY), truth.sky_fluence(TEST_SKY))
    total_truth = truth.bins[..., 0].sum(-1)
    flux_error = np.linalg.norm((test.bins[..., 1:] - truth.bins[..., 1:]).sum(1), axis=-1)
    flux = flux_error[total_truth > 0] / total_truth[total_truth > 0]
    order = np.argsort(total_truth)
    order = order[total_truth[order] > 0]
    deciles = [float(sun_test[group].mean() / sun_truth[group].mean()) for group in np.array_split(order, 10)]
    return {
        "sun_error_median": float(np.median(sun)),
        "sun_error_p90": float(np.percentile(sun, 90)),
        "sky_error_median": float(np.median(sky)),
        "sky_error_p90": float(np.percentile(sky, 90)),
        "flux_error_median": float(np.median(flux)),
        "sun_ratio_by_decile": deciles,
    }


def convergence(half: Field, full: Field) -> dict:
    """The relative standard deviation of the full field's sun and sky fluence, from
    the disagreement of its two halves: robust, as 1.4826 times the median absolute
    relative difference, halved in variance once for each half and once for the
    full field's twice as many walks."""
    rest = full.second_half(half)

    def sigma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = 0.5 * (a + b)
        lit = mean > 0
        return 1.4826 * np.abs(a[lit] - b[lit]) / mean[lit] / 2.0

    sun = sigma(half.sun_fluence(SUN_DIRECTIONS), rest.sun_fluence(SUN_DIRECTIONS))
    sky = sigma(half.sky_fluence(TEST_SKY), rest.sky_fluence(TEST_SKY))
    return {
        "sun_sigma_median": float(np.median(sun)),
        "sun_sigma_p90": float(np.percentile(sun, 90)),
        "sky_sigma_median": float(np.median(sky)),
    }
