"""Predictions at the precision a model ships with, per the inference benchmark's
fastest storage: half-precision tables and planes, and for hardware-filtered
textures (K-Planes planes, LUTs) interpolation weights rounded to the texture
unit's 8 fractional bits. Heads stay float32, as their constant buffers are.

The rounding-to-1/256 weight model approximates the texture unit; the
benchmark measured that real filtering differs from it by up to about 1 % of
the output RMS.
"""

import numpy as np
import torch

from study import baked, encodings


def half(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().half().float()


def kplanes_texture(planes: encodings.KPlanes, uvwz: torch.Tensor, weight_bits: int | None) -> torch.Tensor:
    """K-Planes features from half-precision planes with bilinear weights rounded to
    `weight_bits` fractional bits (None: exact), as kplanes.slang's TexturePlanes reads them."""
    x = uvwz.clamp(0.0, 1.0)
    scales = []
    for s, vertices in enumerate(planes.vertices):
        product = None
        for p, (a, b) in enumerate(encodings.PLANE_AXES):
            plane = half(planes.planes[s * len(encodings.PLANE_AXES) + p])[0]  # [F, n_b, n_a]
            n = torch.tensor([vertices[a], vertices[b]], device=x.device)
            t = x[:, [a, b]] * (n - 1)
            base = torch.minimum(t.floor(), (n - 2).to(t.dtype))
            weight = t - base
            if weight_bits is not None:
                weight = torch.round(weight * 2.0**weight_bits) / 2.0**weight_bits
            base = base.long()
            sampled = 0.0
            for corner in range(4):
                i, j = corner & 1, corner >> 1
                corner_weight = (weight[:, 0] if i else 1.0 - weight[:, 0]) * (weight[:, 1] if j else 1.0 - weight[:, 1])
                sampled = sampled + corner_weight[None, :] * plane[:, base[:, 1] + j, base[:, 0] + i]
            product = sampled if product is None else product * sampled
        scales.append(product.T)
    return torch.cat(scales, dim=1)


@torch.no_grad()
def predict(model: encodings.Model, spec: dict, uvwz: np.ndarray, storage: str) -> np.ndarray:
    """The model's prediction at uvwz [N, 4] as `storage` ships it: "texture" (half,
    filtered weights) or "buffer" (half, exact weights)."""
    assert storage in ("texture", "buffer")
    weight_bits = baked.TEXTURE_WEIGHT_BITS if storage == "texture" else None
    kind = spec["encoding"]["kind"]
    if kind == "bruneton":
        table = model.encoding.table.detach().cpu().double().numpy()
        n = model.encoding.vertices
        table = table.astype(np.float16).astype(np.float64).reshape(n[3], n[2], n[1], n[0], 3)
        return baked.lut4d_sample(table, uvwz, weight_bits)
    x = torch.from_numpy(uvwz.astype(np.float32)).to(next(model.parameters()).device)
    outputs = []
    for start in range(0, len(x), 2**16):
        chunk = model.encoding_input(x[start:start + 2**16])
        if kind == "kplanes":
            features = kplanes_texture(model.encoding, chunk, weight_bits)
        elif kind in ("hash", "dense"):
            table = model.encoding.table.data
            exact = table.clone()
            table.copy_(half(table))
            features = model.encoding(chunk)
            table.copy_(exact)
        else:
            assert kind in ("raw", "fourier"), kind
            features = model.encoding(chunk)
        rgb = features if model.head is None else model.head(features)
        outputs.append(torch.nn.functional.softplus(rgb) if model.softplus else rgb)
    return torch.cat(outputs).cpu().double().numpy()
