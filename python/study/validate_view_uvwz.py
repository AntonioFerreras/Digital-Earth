"""Checks view_uvwz against the renderer: `earth render --mode mlp` versus the EMLP
forward pass on the uvwz of the same jittered rays, averaged over the same frames.

Usage (GPU renders go through `--earth`):
    python python/study/validate_view_uvwz.py --earth <earth binary> --weights <file.emlp> \
        --width 320 --height 180 --spp 16 --out-dir cache/study/validate
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_weights import ACTIVATIONS, decode  # noqa: E402

import view_uvwz  # noqa: E402


def pcg4d(v: np.ndarray) -> np.ndarray:
    """shaders/common/rng.slang pcg4d on uint32 [..., 4]."""
    v = v.astype(np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    x, y, z, w = (v[..., i].copy() for i in range(4))
    for round_index in range(2):
        x += y * w
        y += z * x
        z += x * y
        w += y * z
        if round_index == 0:
            x ^= x >> np.uint32(16)
            y ^= y >> np.uint32(16)
            z ^= z >> np.uint32(16)
            w ^= w >> np.uint32(16)
    return np.stack([x, y, z, w], axis=-1)


def pixel_jitter(frame: int, width: int, height: int) -> np.ndarray:
    """`rng.next4().xy` of dimension 0, [H, W, 2] bottom row first."""
    px, py = np.meshgrid(np.arange(width, dtype=np.uint32), np.arange(height, dtype=np.uint32))
    keys = np.stack([px, py, np.full_like(px, frame), np.zeros_like(px)], axis=-1)
    bits = pcg4d(keys)[..., :2] >> np.uint32(8)
    return bits.astype(np.float32) * np.float32(1.0 / 16777216.0)


def forward(layers, uvwz: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(uvwz.reshape(-1, 4).astype(np.float64))
    with torch.no_grad():
        for weight, bias, activation in layers:
            x = x @ torch.from_numpy(weight.astype(np.float64)).T + torch.from_numpy(bias.astype(np.float64))
            if activation == ACTIVATIONS["gelu"]:
                x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
            elif activation == ACTIVATIONS["relu"]:
                x = torch.clamp(x, min=0.0)
    return x.numpy().reshape(*uvwz.shape[:-1], 3)


def write_config(config: view_uvwz.Config, path: Path) -> None:
    def vec(v):
        return ".{ " + ", ".join(repr(float(c)) for c in v) + " }"

    path.write_text(
        ".{\n"
        f"    .camera = .{{ .position = {vec(config.position)}, .forward = {vec(config.forward)}, "
        f".up = {vec(config.up)}, .vertical_fov_degrees = {config.vertical_fov_degrees!r}, "
        f".aspect_scale = {config.aspect_scale!r} }},\n"
        f"    .sun = .{{ .angle_degrees = {config.sun_angle_degrees!r}, "
        f".path_rotation_degrees = {config.sun_path_rotation_degrees!r} }},\n"
        f"    .exposure = {config.exposure!r},\n"
        "}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--earth", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--lock", type=Path, help="flock file serializing GPU use")
    parser.add_argument("--views", nargs="+", default=list(view_uvwz.VIEWS))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("cache/study/validate"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_scale, layers = decode(args.weights.read_bytes())
    for name in args.views:
        config = view_uvwz.VIEWS[name]
        # The default view renders without --config, the preset view from its own file.
        config_path = {"default": None, "sunset_limb": view_uvwz.REPO / "presets" / "sunset-hurricane.zon"}.get(
            name, args.out_dir / f"{name}.zon"
        )
        if config_path == args.out_dir / f"{name}.zon":
            write_config(config, config_path)
        linear_path = args.out_dir / f"{name}_mlp.pfm"
        command = [
            str(args.earth), "render", "--mode", "mlp", "--preset", "symmetric",
            "--weights", str(args.weights), *(["--config", str(config_path)] if config_path else []),
            "--width", str(args.width), "--height", str(args.height), "--spp", str(args.spp),
            "--out", str(args.out_dir / f"{name}_mlp.png"), "--out-linear", str(linear_path),
        ]
        if args.lock:
            command = ["flock", str(args.lock), *command]
        subprocess.run(command, check=True)
        rendered = view_uvwz.read_pfm(linear_path)

        uniforms = view_uvwz.frame_uniforms(config, args.width, args.height)
        expected = np.zeros_like(rendered, dtype=np.float64)
        for frame in range(args.spp):
            jitter = pixel_jitter(frame, args.width, args.height)
            uvwz, enters = view_uvwz.uvwz_from_uniforms(uniforms, args.width, args.height, jitter)
            expected += np.where(enters[..., None], output_scale * forward(layers, uvwz), 0.0)
        expected /= args.spp

        uvwz_centre, enters_centre = view_uvwz.view_uvwz(name, args.width, args.height)
        centre = np.where(enters_centre[..., None], output_scale * forward(layers, uvwz_centre), 0.0)

        def report(label, reference, mask):
            difference = np.abs(rendered - reference)[mask]
            relative = difference / np.maximum(np.abs(reference[mask]), 1e-3 * np.abs(reference[mask]).max())
            print(f"{name:12s} {label:24s} pixels {mask.sum():7d}  abs max {difference.max():.3e} "
                  f"mean {difference.mean():.3e}  rel max {relative.max():.3e} mean {relative.mean():.3e}  "
                  f"radiance mean {np.abs(reference[mask]).mean():.3e}")

        all_pixels = np.ones(rendered.shape[:2], dtype=bool)
        report("same-jitter average", expected, all_pixels)
        # Pixels whose every jittered ray agrees with the centre on entering the atmosphere.
        interior = enters_centre & (expected[..., 0] > 0) & np.all(rendered > 0, axis=-1)
        report("pixel centre, interior", centre, interior)
        print(f"{name:12s} enters (centre) {enters_centre.mean():.4f}  rendered nonzero "
              f"{np.any(rendered != 0, axis=-1).mean():.4f}")


if __name__ == "__main__":
    main()
