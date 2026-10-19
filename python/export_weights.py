"""Export a trained network to the renderer's EMLP file, version 3.

The layout is specified in src/assets/Network.zig, which reads it. Two sources:

    --checkpoint <file.pth>   the plain MLP of python/model.py on raw uvwz, without
                              u_mu rotation; --training-data sets the output scale
    --run <dir>               an encoding-study run (python/study): result.json's spec
                              and model.pt, any encoding; learned on `earth datagen` data

Every export is read back and evaluated by `reference`, a float64 numpy forward
pass over the file's own tensors with the shaders' float32 lattice coordinates,
and compared with torch on random uvwz: max |reference - torch| / RMS(torch) must
stay below `VERIFY_TOLERANCE`. Hash grids and K-Planes also report what shipping
changes: half-precision tables and planes, and for K-Planes the texture unit's
8-bit filter weights, emulated.

The renderer multiplies the network's output by the output scale: networks
trained on `earth datagen` output need none, while the text data of the original
Taichi data_prep.py is `TAICHI_OUTPUT_SCALE` times too dark.

Usage:
    python python/export_weights.py --checkpoint model_weights/mlp_e120.pth --training-data taichi \\
        --out cache/mlp/mlp_e120.emlp
    python python/export_weights.py --run ../zs-study/cache/study/runs/kplanes8_w16x1_s16 \\
        --out cache/neural/kplanes8_w16x1_s16.emlp
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = b"EMLP"
VERSION = 3
# 2^-ORIGINAL_ESTIMATOR_STOPS in shaders/spectral/cie.slang: the original estimator
# divided by an unnormalized wavelength pdf.
TAICHI_OUTPUT_SCALE = 2.0 ** 7.406
OUTPUT_SCALES = {"datagen": 1.0, "taichi": TAICHI_OUTPUT_SCALE}
ACTIVATIONS = {"identity": 0, "gelu": 1, "relu": 2}
ENCODINGS = {"raw": 0, "fourier": 1, "hash_grid": 2, "k_planes": 3}
FLAG_MU_ROTATION = 1
HASH_PRIMES = np.array([1, 2654435761, 805459861, 3674653429], dtype=np.uint64)
PLANE_AXES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
K_PLANES_SCALES = 3
# The texture unit's bilinear weights carry 8 fractional bits.
FILTER_WEIGHT_BITS = 8
VERIFY_TOLERANCE = 1e-6
VERIFY_SAMPLES = 8192
VERIFY_GRID = 4096

# A network is a dict: output_scale, mu_rotation, encoding (its kind and tensors),
# and layers [(weight [out, in], bias [out], activation name)].


def mlp_checkpoint_network(checkpoint: Path, output_scale: float) -> tuple[dict, nn.Module]:
    """The python/model.py MLP of `checkpoint` and its torch module."""
    from model import MLP

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state_dict.items()})
    model.eval()
    layers = []
    for module in model.linear_layers:
        if isinstance(module, nn.Linear):
            layers.append([module, "identity"])
        else:
            assert isinstance(module, nn.GELU) and module.approximate == "none", "the shader implements exact GELU only"
            assert layers[-1][1] == "identity", "two activations in a row"
            layers[-1][1] = "gelu"
    # MLP.forward applies ReLU to the output layer in code, not as a module.
    layers.append([model.output_layer, "relu"])
    network = {"output_scale": output_scale, "mu_rotation": False, "encoding": {"kind": "raw"},
               "layers": [linear_arrays(linear, activation) for linear, activation in layers]}
    return network, model


def study_network(run_dir: Path) -> tuple[dict, nn.Module]:
    """The encoding-study model of `run_dir` and its torch module, on the CPU."""
    from study import encodings

    spec = json.loads((run_dir / "result.json").read_text())["spec"]
    model = encodings.build(spec)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return study_model_network(model, spec), model


def study_model_network(model: nn.Module, spec: dict) -> dict:
    """An encoding-study model (python/study/encodings.py) built from `spec`."""
    assert spec["output"] == "identity", "the shader's output layer is ReLU; softplus does not export"
    encoding = model.encoding
    kind = spec["encoding"]["kind"]
    if kind == "raw":
        encoding_dict = {"kind": "raw"}
    elif kind == "fourier":
        encoding_dict = {"kind": "fourier", "octaves": encoding.octaves,
                         "frequency": np.array(encoding.frequency_list, np.float32)}
    elif kind == "hash":
        levels = [list(cells) + [int(encoding.first[l]), encoding.table_size - 1, int(encoding.hashed[l]), 0]
                  for l, cells in enumerate(encoding.cells_list)]
        encoding_dict = {"kind": "hash_grid", "levels": np.array(levels, np.uint32),
                         "table": encoding.table.detach().cpu().numpy().astype(np.float32).reshape(-1)}
    else:
        assert kind == "kplanes", f"the neural mode has no {kind} encoding"
        # Each plane is [1, F, n_b, n_a]: rows along the second axis, features last.
        planes = [plane.detach().cpu()[0].permute(1, 2, 0).reshape(-1).numpy() for plane in encoding.planes]
        encoding_dict = {"kind": "k_planes", "features": encoding.features,
                         "vertices": np.array(encoding.vertices, np.uint32),
                         "planes": np.concatenate(planes).astype(np.float32)}
    modules = list(model.head)
    layers = []
    for index, module in enumerate(modules):
        if not isinstance(module, nn.Linear):
            continue
        following = modules[index + 1] if index + 1 < len(modules) else None
        if following is None:
            activation = "relu"  # Trained as identity; the shader clamps radiance at zero.
        elif isinstance(following, nn.GELU):
            assert following.approximate == "none", "the shader implements exact GELU only"
            activation = "gelu"
        else:
            assert isinstance(following, nn.ReLU), following
            activation = "relu"
        layers.append(linear_arrays(module, activation))
    return {"output_scale": 1.0, "mu_rotation": bool(spec.get("mu_rotation", False)), "encoding": encoding_dict,
            "layers": layers}


def linear_arrays(linear: nn.Linear, activation: str) -> tuple[np.ndarray, np.ndarray, str]:
    return (linear.weight.detach().cpu().numpy().astype(np.float32),
            linear.bias.detach().cpu().numpy().astype(np.float32), activation)


def encoding_words(encoding: dict) -> np.ndarray:
    kind = encoding["kind"]
    if kind == "raw":
        return np.zeros(0, "<u4")
    if kind == "fourier":
        return np.concatenate([np.array([encoding["octaves"]], "<u4"), encoding["frequency"].astype("<f4").view("<u4")])
    if kind == "hash_grid":
        levels = encoding["levels"]
        return np.concatenate([np.array([len(levels)], "<u4"), levels.astype("<u4").reshape(-1),
                               encoding["table"].astype("<f4").view("<u4")])
    assert kind == "k_planes", kind
    return np.concatenate([np.array([encoding["features"]], "<u4"), encoding["vertices"].astype("<u4").reshape(-1),
                           encoding["planes"].astype("<f4").view("<u4")])


def encode(network: dict) -> bytes:
    output_scale = network["output_scale"]
    assert output_scale > 0 and math.isfinite(output_scale)
    encoding = encoding_words(network["encoding"])
    layers = network["layers"]
    header = np.array([VERSION, len(layers), np.float32(output_scale).view("<u4"),
                       FLAG_MU_ROTATION if network["mu_rotation"] else 0, ENCODINGS[network["encoding"]["kind"]],
                       encoding.size], "<u4")
    table = np.array([[weight.shape[1], weight.shape[0], ACTIVATIONS[activation]] for weight, _, activation in layers],
                     "<u4")
    parts = [MAGIC, header.tobytes(), encoding.tobytes(), table.tobytes()]
    for weight, bias, _ in layers:
        parts += [weight.astype("<f4").tobytes(), bias.astype("<f4").tobytes()]
    return b"".join(parts)


def decode(blob: bytes) -> dict:
    """A network from an EMLP file of any version."""
    assert blob[:4] == MAGIC, "not an EMLP file"
    words = np.frombuffer(blob, "<u4")
    version, layer_count = int(words[1]), int(words[2])
    assert version in (1, 2, VERSION), f"unsupported EMLP version {version}"
    position = 3
    output_scale = TAICHI_OUTPUT_SCALE
    if version >= 2:
        output_scale = float(words[3:4].view("<f4")[0])
        position = 4
    mu_rotation, encoding = False, {"kind": "raw"}
    if version == VERSION:
        flags, kind_code, size = (int(w) for w in words[position:position + 3])
        position += 3
        mu_rotation = bool(flags & FLAG_MU_ROTATION)
        kind = {code: name for name, code in ENCODINGS.items()}[kind_code]
        encoding = decode_encoding(kind, words[position:position + size])
        position += size
    table = words[position:position + 3 * layer_count].reshape(layer_count, 3)
    position += 3 * layer_count
    floats = words.view("<f4")
    names = {code: name for name, code in ACTIVATIONS.items()}
    layers = []
    for in_dim, out_dim, activation in table.tolist():
        weight = floats[position:position + out_dim * in_dim].reshape(out_dim, in_dim)
        position += out_dim * in_dim
        bias = floats[position:position + out_dim]
        position += out_dim
        layers.append((weight, bias, names[activation]))
    assert position == words.size, "trailing bytes after the last layer"
    return {"output_scale": output_scale, "mu_rotation": mu_rotation, "encoding": encoding, "layers": layers}


def decode_encoding(kind: str, words: np.ndarray) -> dict:
    if kind == "raw":
        return {"kind": kind}
    if kind == "fourier":
        return {"kind": kind, "octaves": int(words[0]), "frequency": words[1:5].view("<f4")}
    if kind == "hash_grid":
        count = int(words[0])
        return {"kind": kind, "levels": words[1:1 + 8 * count].reshape(count, 8),
                "table": words[1 + 8 * count:].view("<f4")}
    return {"kind": kind, "features": int(words[0]), "vertices": words[1:1 + 4 * K_PLANES_SCALES].reshape(-1, 4),
            "planes": words[1 + 4 * K_PLANES_SCALES:].view("<f4")}


# The reference forward pass ---------------------------------------------------------


def fp32_product(x: np.ndarray, scale) -> np.ndarray:
    """x * scale rounded to float32, as the shaders compute lattice coordinates: at 1024
    cells the rounding moves interpolation weights by up to 6e-5."""
    return (x.astype(np.float32) * np.asarray(scale, np.float32)).astype(np.float64)


def fourier(x: np.ndarray, octaves: int, frequency: np.ndarray) -> np.ndarray:
    cycles = fp32_product(x, frequency)
    features = [x]
    for k in range(octaves):
        phase = 2.0 * math.pi * np.mod(cycles * 2.0**k, 1.0)
        features.append(np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(len(x), 8))
    return np.concatenate(features, axis=1)


def hash_grid(x: np.ndarray, levels: np.ndarray, table: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    table = table.astype(np.float64).reshape(-1, 2)
    out = []
    for level in levels:
        cells = level[:4].astype(np.int64)
        first, mask, hashed = int(level[4]), int(level[5]), bool(level[6])
        position = fp32_product(x, cells)
        base = np.minimum(np.floor(position).astype(np.int64), cells - 1)
        w = position - base
        stride = np.array([1, cells[0] + 1, (cells[0] + 1) * (cells[1] + 1),
                           (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1)])
        total = np.zeros((len(x), 2))
        for corner in range(16):
            offset = np.array([(corner >> a) & 1 for a in range(4)])
            vertex = base + offset
            weight = np.prod(np.where(offset != 0, w, 1.0 - w), axis=1)
            if hashed:
                products = (vertex.astype(np.uint64) * HASH_PRIMES) & np.uint64(0xFFFFFFFF)
                index = (products[:, 0] ^ products[:, 1] ^ products[:, 2] ^ products[:, 3]) & np.uint64(mask)
            else:
                index = vertex @ stride
            total += weight[:, None] * table[first + index.astype(np.int64)]
        out.append(total)
    return np.concatenate(out, axis=1)


def k_planes(x: np.ndarray, features: int, vertices: np.ndarray, planes: np.ndarray,
             weight_bits: int | None) -> np.ndarray:
    """Bilinear with aligned corners; `weight_bits` rounds the weights as a texture unit does."""
    x = np.clip(x, 0.0, 1.0)
    planes = planes.astype(np.float64)
    out = []
    source = 0
    for n in vertices.astype(np.int64):
        t = fp32_product(x, n - 1)
        product = np.ones((len(x), features))
        for a, b in PLANE_AXES:
            size = n[a] * n[b] * features
            plane = planes[source:source + size].reshape(n[b], n[a], features)
            source += size
            base = np.minimum(np.floor(t[:, [a, b]]).astype(np.int64), np.array([n[a], n[b]]) - 2)
            w = t[:, [a, b]] - base
            if weight_bits is not None:
                w = np.round(w * 2.0**weight_bits) / 2.0**weight_bits
            sampled = np.zeros((len(x), features))
            for corner in range(4):
                offset = np.array([corner & 1, (corner >> 1) & 1])
                vertex = base + offset
                weight = np.prod(np.where(offset != 0, w, 1.0 - w), axis=1)
                sampled += weight[:, None] * plane[vertex[:, 1], vertex[:, 0]]
            product *= sampled
        out.append(product)
    assert source == len(planes)
    return np.concatenate(out, axis=1)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def half(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


def reference(network: dict, uvwz: np.ndarray, half_tables: bool = False, filter_weights: bool = False) -> np.ndarray:
    """The network's radiance at uvwz [N, 4] in float64. `half_tables` reads hash tables
    and planes at the half precision the GPU stores; `filter_weights` rounds K-Planes'
    bilinear weights as the texture unit does."""
    x = uvwz.astype(np.float32)
    if network["mu_rotation"]:
        x[:, 2] = np.mod(x[:, 2] + np.float32(0.5), np.float32(1.0))
    x = x.astype(np.float64)
    encoding = network["encoding"]
    kind = encoding["kind"]
    if kind == "raw":
        h = x
    elif kind == "fourier":
        h = fourier(x, encoding["octaves"], encoding["frequency"])
    elif kind == "hash_grid":
        h = hash_grid(x, encoding["levels"], half(encoding["table"]) if half_tables else encoding["table"])
    else:
        h = k_planes(x, encoding["features"], encoding["vertices"],
                     half(encoding["planes"]) if half_tables else encoding["planes"],
                     FILTER_WEIGHT_BITS if filter_weights else None)
    for weight, bias, activation in network["layers"]:
        h = h @ weight.astype(np.float64).T + bias.astype(np.float64)
        if activation == "gelu":
            h = gelu(h)
        elif activation == "relu":
            h = np.maximum(h, 0.0)
    return network["output_scale"] * h


def torch_radiance(model: nn.Module, uvwz: np.ndarray, output_scale: float) -> np.ndarray:
    """The torch module's radiance in float64, clamped at zero as the shipped output layer is.
    In float32 torch's own rounding (Fourier phases, 64-wide sums) would dominate."""
    with torch.no_grad():
        prediction = model.double()(torch.from_numpy(uvwz)).numpy()
    model.float()
    return output_scale * np.maximum(prediction, 0.0)


def verify(model: nn.Module, blob: bytes) -> dict:
    """The file read back against torch on random uvwz, errors relative to the output RMS;
    for tables and planes also what shipping them changes."""
    # On a 2^-12 grid every float32 step of the shaders, the u_mu rotation and the
    # lattice coordinates, is exact, so float64 torch and the reference agree to
    # rounding unless the file is laid out wrong.
    uvwz = np.random.default_rng(0).integers(0, VERIFY_GRID + 1, (VERIFY_SAMPLES, 4)) / VERIFY_GRID
    network = decode(blob)
    expected = torch_radiance(model, uvwz, network["output_scale"])
    rms = float(np.sqrt(np.mean(expected**2)))
    exact = reference(network, uvwz)
    report = {"max_error_over_rms": float(np.abs(exact - expected).max() / rms)}
    if network["encoding"]["kind"] in ("hash_grid", "k_planes"):
        shipped = reference(network, uvwz, half_tables=True, filter_weights=True)
        lit = expected > 0.01 * rms
        relative = np.abs(shipped - exact)[lit] / expected[lit]
        report["shipped_max_over_rms"] = float(np.abs(shipped - exact).max() / rms)
        report["shipped_median_rel"] = float(np.median(relative))
        report["shipped_p99_rel"] = float(np.percentile(relative, 99))
    return report


def export(network: dict, model: nn.Module, out: Path) -> dict:
    """Writes `network` to `out` once its read-back passes `verify`; returns the report."""
    blob = encode(network)
    report = verify(model, blob)
    if not report["max_error_over_rms"] <= VERIFY_TOLERANCE:
        raise SystemExit(f"{out}: the file's forward pass differs from torch by "
                         f"{report['max_error_over_rms']:.2e} of the output RMS")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    report["bytes"] = len(blob)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="python/model.py MLP state dict (.pth)")
    source.add_argument("--run", type=Path, help="encoding-study run directory (result.json, model.pt)")
    parser.add_argument("--training-data", choices=OUTPUT_SCALES,
                        help="with --checkpoint: what the network learned, `earth datagen` output or the original "
                             "Taichi text data")
    parser.add_argument("--out", type=Path, required=True, help="output .emlp path")
    args = parser.parse_args()

    if args.checkpoint:
        if args.training_data is None:
            parser.error("--checkpoint needs --training-data")
        network, model = mlp_checkpoint_network(args.checkpoint, OUTPUT_SCALES[args.training_data])
    else:
        network, model = study_network(args.run)
    report = export(network, model, args.out)
    print(f"wrote {args.out}: {json.dumps(report)}")


if __name__ == "__main__":
    main()
