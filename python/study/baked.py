"""Candidate F: the LUT that `earth datagen --sampling grid` bakes, read directly.

The bake follows atmosphere/lut4d.slang: vertex k of an axis at fraction
k / (n - 1) of the uvwz range scattering_uvwz reaches, mu as two halves (ground
rays, then sky rays) that interpolation never crosses, records u fastest.
`lut4d_sample` is a float64 port of the shader's lookup; the texture unit
rounds the mu_s, mu, and r weights to 8 fractional bits, which `weight_bits`
reproduces.
"""

from pathlib import Path

import numpy as np

import datagen_format
from study import encodings, metrics

LUT_FILE = "lut_bruneton_32x32x128x32_64k.edgn"
LUT_SHAPE = (32, 32, 128, 32)  # vertices along u (nu), v (mu_s), w (mu), z (r)
REFERENCE_FILE = "ref_clean.edgn"
TEXTURE_WEIGHT_BITS = 8


def lut4d_positions(uvwz: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Lattice positions [B, 4] and the largest base vertex per point [B, 4], as lut4d_sample
    computes them; mu's base stays inside the point's half."""
    def fraction(u, low, high):
        return np.clip((u - low) / (high - low), 0.0, 1.0)

    n = np.array(shape)
    half = shape[2] // 2
    w = uvwz[:, 2]
    sky = w >= 0.5
    ground_low = encodings.W_GROUND_LOW
    mu = np.where(sky, half + fraction(w, 0.5 + ground_low, 1.0 - ground_low) * (half - 1),
                  fraction(w, ground_low, 0.5 - ground_low) * (half - 1))
    position = np.stack([fraction(uvwz[:, 0], 0.0, 1.0) * (n[0] - 1),
                         fraction(uvwz[:, 1], encodings.V_LOW, 1.0 - encodings.V_LOW) * (n[1] - 1),
                         mu,
                         fraction(uvwz[:, 3], encodings.Z_LOW, 1.0 - encodings.Z_LOW) * (n[3] - 1)], axis=1)
    base_max = np.broadcast_to(n - 2, position.shape).copy()
    base_max[:, 2] = np.where(sky, 2 * half - 2, half - 2)
    return position, base_max


def lut4d_sample(table: np.ndarray, uvwz: np.ndarray, weight_bits: int | None) -> np.ndarray:
    """Quadrilinear lookup of table [n_z, n_w, n_v, n_u, 3] at uvwz [B, 4]; with
    `weight_bits`, the mu_s, mu, and r weights round as the texture unit's do."""
    shape = table.shape[:4][::-1]
    position, base_max = lut4d_positions(uvwz.astype(np.float64), shape)
    base = np.minimum(np.floor(position), base_max).astype(np.int64)
    fraction = position - base
    if weight_bits is not None:
        scale = 2.0**weight_bits
        fraction[:, 1:] = np.round(fraction[:, 1:] * scale) / scale
    total = np.zeros((len(uvwz), 3))
    for corner in range(16):
        offset = np.array([(corner >> a) & 1 for a in range(4)])
        vertex = base + offset
        weight = np.prod(np.where(offset != 0, fraction, 1.0 - fraction), axis=1)
        total += weight[:, None] * table[vertex[:, 3], vertex[:, 2], vertex[:, 1], vertex[:, 0]]
    return total


def load_table(data_dir: Path) -> np.ndarray:
    """The bake as [n_z, n_w, n_v, n_u, 3], checked to sit on lut4d's vertices."""
    records = datagen_format.read_records(data_dir / LUT_FILE)
    n = LUT_SHAPE
    assert len(records) == np.prod(n)
    position, _ = lut4d_positions(records[:, :4].astype(np.float64), n)
    vertex = np.stack(np.meshgrid(*[np.arange(size) for size in n], indexing="ij"), axis=-1)
    expected = vertex.transpose(3, 2, 1, 0, 4).reshape(-1, 4)
    assert np.abs(position - expected).max() < 1e-3, "the bake is not on lut4d's vertices"
    return records[:, 4:].reshape(n[3], n[2], n[1], n[0], 3).astype(np.float64)


def evaluate(data_dir: Path) -> tuple[dict, np.ndarray]:
    """The result record and the exact-weight prediction at the reference points."""
    table = load_table(data_dir)
    reference = datagen_format.read_records(data_dir / REFERENCE_FILE)
    exact = lut4d_sample(table, reference[:, :4], None)
    # Shipped as rgba16f, filtered by the texture unit.
    texture = lut4d_sample(table.astype(np.float16).astype(np.float64), reference[:, :4], TEXTURE_WEIGHT_BITS)
    result = {
        "spec": {"name": "baked_lut_32x32x128x32", "candidate": "F",
                 "encoding": {"kind": "bruneton", "vertices": list(LUT_SHAPE)}},
        "parameters": int(np.prod(LUT_SHAPE)) * 3,
        "samples_seen": 0,
        "metrics": metrics.evaluate(exact, reference),
        "metrics_shipped_texture": metrics.evaluate(texture, reference),
    }
    return result, exact.astype(np.float32)
