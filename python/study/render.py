"""Python-side renders of trained models next to the path tracer: each pixel's
uvwz (view_uvwz) through the model, tonemapped as view_uvwz.tonemap does."""

import json
from pathlib import Path

import numpy as np
import torch

from study import baked, encodings, metrics, view_uvwz

SIZE = (640, 360)


def load_model(run_dir: Path) -> encodings.Model:
    spec = json.loads((run_dir / "result.json").read_text())["spec"]
    model = encodings.build(spec)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.cuda().eval()


def radiance(run_dir: Path, uvwz: np.ndarray, data_dir: Path) -> np.ndarray:
    """The run's prediction at uvwz [N, 4], clamped at zero as the shipped ReLU output does."""
    if run_dir.name.startswith("baked_lut"):
        return np.maximum(baked.lut4d_sample(baked.load_table(data_dir), uvwz, None), 0.0)
    model = load_model(run_dir)
    with torch.no_grad():
        x = torch.from_numpy(uvwz).cuda()
        prediction = torch.cat([model(x[start:start + 2**16]) for start in range(0, len(x), 2**16)])
    return np.maximum(prediction.cpu().numpy().astype(np.float64), 0.0)


def compare(run_dirs: list[Path], views: list[str], renders_dir: Path, data_dir: Path, out_png: Path) -> dict:
    """Saves a grid (rows: views; columns: path tracer, then each run) and returns each
    run's mean relative error against the path tracer over lit atmosphere pixels."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    width, height = SIZE
    errors: dict[str, dict[str, float]] = {}
    figure, axes = pyplot.subplots(len(views), len(run_dirs) + 1, figsize=(3.2 * (len(run_dirs) + 1), 1.9 * len(views)),
                                   squeeze=False)
    for row, view in enumerate(views):
        uvwz = np.load(renders_dir / f"{view}_uvwz_{width}x{height}.npy")
        enters = np.load(renders_dir / f"{view}_enters_{width}x{height}.npy")
        reference = np.load(renders_dir / f"{view}_pt_65536_{width}x{height}.npy").astype(np.float64)
        exposure = view_uvwz.VIEWS[view].exposure
        lit = enters & (reference.mean(axis=2) > metrics.LIT_MIN)
        images = [("path tracer", reference)]
        for run_dir in run_dirs:
            image = np.zeros((height, width, 3))
            image[enters] = radiance(run_dir, uvwz[enters], data_dir)
            error = np.abs(image - reference)[lit] / np.abs(reference[lit]).clip(1e-12)
            errors.setdefault(run_dir.name, {})[view] = float(error.mean())
            images.append((f"{run_dir.name}\n{100 * error.mean():.1f}% mean rel. vs PT", image))
        for column, (title, image) in enumerate(images):
            axis = axes[row, column]
            axis.imshow(view_uvwz.tonemap(image, exposure))
            axis.set_title(title, fontsize=7)
            axis.set_axis_off()
    figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=110)
    pyplot.close(figure)
    return errors
