"""Converts an OpenVDB density grid into the cloud bench's dense volume (.ecld).

The `cloud_bench` render mode (src/render/CloudBench.zig) reads the result; the
format is src/assets/CloudVolume.zig's. The grid keeps the VDB's world frame, in
which the Walt Disney Animation Studios cloud's Mitsuba scene is set, so cameras
from that scene carry over unchanged.

Next to the density goes a majorant per cell of CELL^3 voxels: the maximum over
the cell grown by one voxel, which bounds the trilinear filter anywhere in it. Last
comes the distance from inside the cloud to its edge, on a grid DISTANCE_DOWNSAMPLE
times coarser, for the dimensional profile of Nubis Cubed (Schneider 2023): the
cloud is where the coarse grid's mean density exceeds --surface-density, and
outside it the distance is zero.

Run: pixi run -e py python python/clouds/convert_vdb.py \
         sources/wdas_cloud/wdas_cloud/wdas_cloud_half.vdb cache/clouds/wdas_cloud_half.ecld
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import openvdb
from scipy import ndimage

MAGIC = b"ECLOUD02"
HEADER_SIZE = 64
CELL = 8
DISTANCE_DOWNSAMPLE = 2


def read_density(path: Path, grid_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """The active bounding box as a dense (z, y, x) array, its first index, and the voxel size."""
    grid = openvdb.read(str(path), grid_name)
    voxel = grid.transform.voxelSize()
    if not (voxel[0] == voxel[1] == voxel[2]):
        raise SystemExit(f"{path}: voxels are not cubes: {voxel}")
    lo, hi = grid.evalActiveVoxelBoundingBox()
    shape = tuple(h - l + 1 for l, h in zip(lo, hi))
    dense = np.zeros(shape, np.float32)
    grid.copyToArray(dense, ijk=lo)
    return np.ascontiguousarray(dense.transpose(2, 1, 0)), np.array(lo, np.float64), voxel[0]


def majorants(density: np.ndarray) -> np.ndarray:
    grown = density.copy()
    for axis in range(3):
        before = np.zeros_like(grown)
        after = np.zeros_like(grown)
        lead = [slice(None)] * 3
        lag = [slice(None)] * 3
        lead[axis], lag[axis] = slice(1, None), slice(None, -1)
        before[tuple(lead)] = grown[tuple(lag)]
        after[tuple(lag)] = grown[tuple(lead)]
        grown = np.maximum(grown, np.maximum(before, after))
    cells = [-(-n // CELL) for n in density.shape]
    padded = np.zeros([n * CELL for n in cells], np.float32)
    padded[: density.shape[0], : density.shape[1], : density.shape[2]] = grown
    blocks = padded.reshape(cells[0], CELL, cells[1], CELL, cells[2], CELL)
    return blocks.max(axis=(1, 3, 5))


def edge_distance(density: np.ndarray, voxel: float, surface_density: float) -> np.ndarray:
    """Distance to the cloud's edge, in the grid's units, per coarse cell (z, y, x)."""
    k = DISTANCE_DOWNSAMPLE
    coarse_shape = [-(-n // k) for n in density.shape]
    padded = np.zeros([n * k for n in coarse_shape], np.float32)
    padded[: density.shape[0], : density.shape[1], : density.shape[2]] = density
    coarse = padded.reshape(coarse_shape[0], k, coarse_shape[1], k, coarse_shape[2], k).mean(axis=(1, 3, 5))
    # A border of outside cells: the grid's faces bound the cloud.
    inside = np.pad(coarse > surface_density, 1)
    distance = ndimage.distance_transform_edt(inside, sampling=k * voxel)
    return distance[1:-1, 1:-1, 1:-1].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vdb", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--grid", default="density")
    parser.add_argument("--surface-density", type=float, default=0.1)
    args = parser.parse_args()

    density, first_index, voxel = read_density(args.vdb, args.grid)
    if density.min() < 0.0 or not np.isfinite(density).all():
        raise SystemExit(f"{args.vdb}: density must be finite and non-negative")
    cell_max = majorants(density)
    distance = edge_distance(density, voxel, args.surface_density)
    # VDB voxel centres sit at index * voxel, so the grid starts half a voxel before its first.
    min_corner = (first_index - 0.5) * voxel
    size_zyx, cells_zyx, distance_zyx = density.shape, cell_max.shape, distance.shape
    header = MAGIC + struct.pack(
        "<3I I 3I f 3f 3I",
        *size_zyx[::-1],
        CELL,
        *cells_zyx[::-1],
        voxel,
        *min_corner,
        *distance_zyx[::-1],
    )
    header += bytes(HEADER_SIZE - len(header))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as out:
        out.write(header)
        out.write(density.astype("<f2").tobytes())
        out.write(cell_max.astype("<f2").tobytes())
        out.write(distance.astype("<f2").tobytes())
    print(
        f"{args.out}: {size_zyx[::-1]} voxels of {voxel:.4f}, {cells_zyx[::-1]} cells, "
        f"min corner {min_corner}, mean density {density.mean():.4f}, deepest point {distance.max():.1f} units in"
    )


if __name__ == "__main__":
    main()
