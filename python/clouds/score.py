"""Scores the cloud bench's approximation against its reference.

Takes three `earth render --mode cloud_bench --out-linear` images of one view,
rendered with `--bench-view reference`, `approximation`, and `coverage`. Cloud
pixels are those the cloud covers; they are split into thirds by the reference's
luminance, so that dark undersides and bright sunlit tops are scored apart. Ratios
are of mean luminance, approximation over reference.

Run: pixi run -e py python python/clouds/score.py reference.pfm approximation.pfm coverage.pfm
"""

import argparse
from pathlib import Path

import numpy as np

REC709_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
# Opacity below which a pixel counts as sky: the sky seen through thin cloud would
# dominate it.
COVERAGE_MIN = 0.5


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        if f.readline().strip() != b"PF":
            raise SystemExit(f"{path}: not an RGB PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(height, width, 3)
    return data[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("approximation", type=Path)
    parser.add_argument("coverage", type=Path)
    args = parser.parse_args()

    reference = read_pfm(args.reference) @ REC709_LUMINANCE
    approximation = read_pfm(args.approximation) @ REC709_LUMINANCE
    if reference.shape != approximation.shape:
        raise SystemExit("the images differ in size")
    cloud = read_pfm(args.coverage)[..., 0] > COVERAGE_MIN
    if not cloud.any():
        raise SystemExit("no cloud pixels")
    ref, approx = reference[cloud], approximation[cloud]
    edges = np.quantile(ref, [1 / 3, 2 / 3])
    thirds = np.digitize(ref, edges)
    ratios = [approx[thirds == k].mean() / ref[thirds == k].mean() for k in range(3)]
    log_rms = np.sqrt(np.mean(np.log((approx + 1e-4) / (ref + 1e-4)) ** 2))
    print(f"cloud pixels {cloud.sum()}")
    print(f"ratio mean {approx.mean() / ref.mean():.3f}  dark {ratios[0]:.3f}  mid {ratios[1]:.3f}  bright {ratios[2]:.3f}")
    print(f"log rms {log_rms:.3f}")


if __name__ == "__main__":
    main()
