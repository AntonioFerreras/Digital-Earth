"""Converts an OpenVDB density grid into the cloud bench's dense volume (.ecld).

The `cloud_bench` render mode (src/render/CloudBench.zig) reads the result; the
format is src/assets/CloudVolume.zig's. The grid keeps the VDB's world frame, in
which the Walt Disney Animation Studios cloud's Mitsuba scene is set, so cameras
from that scene carry over unchanged; other clouds take --voxel for their size in
that frame, --centre to sit on its vertical axis as the WDAS cloud does, --z-up
when their file's up axis is z, and --density, --match-column or --match-density
to bring their optical depth, or their material, to the WDAS cloud's, since the
bench scales every cloud's density alike. Negative rounding noise in the density is
clamped.

A cloud is brought to the resolution the bench can hold with --resample, a cubic
resampling by that factor per axis, below one a downsampling, and a coarser one
gains detail with --erode, which carves the resampled density with a tiled Worley
noise the way game clouds do (Schneider 2015): with d the density over its
reference, the brightest active voxels', and n the noise, d becomes
(d - s n) / (1 - s n), so that the core keeps its shape while the thin edges break
into wisps and holes at the new voxels' scale.

Next to the density goes a majorant per cell of CELL^3 voxels: the maximum over
the cell grown by one voxel, which bounds the trilinear filter anywhere in it. Last
comes the distance from inside the cloud to its edge, on a grid DISTANCE_DOWNSAMPLE
times coarser, for the dimensional profile of Nubis Cubed (Schneider 2023): the
cloud is where the coarse grid's mean density exceeds --surface-density, and
outside it the distance is zero. The volume is built a slab of voxels at a time,
read from the file as needed, as a field of clouds outgrows memory as a float array.

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
# Output voxels per slab, and the source voxels either side that the cubic filter reaches.
SLAB = 32
RESAMPLE_MARGIN = 2
# At most this many active voxels, drawn evenly, stand for the whole in the erosion's
# reference percentile.
REFERENCE_SAMPLE = 1 << 24
NOISE_TILE = 128
NOISE_OCTAVES = 3
# Of the density, the percentile over the active voxels that the erosion keeps whole.
REFERENCE_PERCENTILE = 99.0
# Voxels with less are outside the footprint that the column statistic is taken over.
FOOTPRINT_DENSITY = 1e-3


class Source:
    """A VDB density grid's active bounding box, read a slab of z at a time as dense
    (z, y, x) arrays with y up; `shape` in that order, `first` the box's first index
    as (x, y, z) in that frame, and `voxel` the file's voxel size."""

    def __init__(self, path: Path, grid_name: str, z_up: bool):
        self.grid = openvdb.read(str(path), grid_name)
        voxel = self.grid.transform.voxelSize()
        if not np.allclose(voxel, voxel[0], rtol=1e-6):
            raise SystemExit(f"{path}: voxels are not cubes: {voxel}")
        self.voxel = voxel[0]
        self.path = path
        self.lo, hi = self.grid.evalActiveVoxelBoundingBox()
        self.file_shape = [h - l + 1 for l, h in zip(self.lo, hi)]
        self.axes = (1, 2, 0) if z_up else (2, 1, 0)
        self.shape = tuple(self.file_shape[a] for a in self.axes)
        self.first = np.array((self.lo[0], self.lo[2], self.lo[1]) if z_up else self.lo, np.float64)

    def slab(self, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) of z as float32, negative rounding noise clamped."""
        z_axis = self.axes[0]
        ijk = list(self.lo)
        ijk[z_axis] += start
        shape = list(self.file_shape)
        shape[z_axis] = stop - start
        dense = np.zeros(shape, np.float32)
        self.grid.copyToArray(dense, ijk=tuple(ijk))
        if not np.isfinite(dense).all():
            raise SystemExit(f"{self.path}: density must be finite")
        return np.ascontiguousarray(np.maximum(dense.transpose(self.axes), 0.0))


def worley_tile(period: int, seed: int) -> np.ndarray:
    """Inverted Worley noise in [0, 1] over a NOISE_TILE cube, periodic, with feature
    points `period` voxels apart, summed over NOISE_OCTAVES halving octaves."""
    rng = np.random.default_rng(seed)
    axis = np.arange(NOISE_TILE, dtype=np.float32) + 0.5
    position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    total = np.zeros((NOISE_TILE,) * 3, np.float32)
    weight = 0.0
    for octave in range(NOISE_OCTAVES):
        cells = NOISE_TILE // (period >> octave)
        points = rng.random((cells, cells, cells, 3), np.float32)
        cell = position / (NOISE_TILE / cells)
        base = np.floor(cell).astype(np.int64)
        nearest = np.full(total.shape, np.inf, np.float32)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = base + np.array([dz, dy, dx])
                    wrapped = neighbour % cells
                    feature = points[wrapped[..., 0], wrapped[..., 1], wrapped[..., 2]] + neighbour
                    np.minimum(nearest, np.linalg.norm(cell - feature, axis=-1), out=nearest)
        octave_weight = 0.5**octave
        total += octave_weight * (1.0 - np.minimum(nearest, 1.0))
        weight += octave_weight
    return total / weight


def erosion_reference(source: Source, density_factor: float) -> float:
    """The REFERENCE_PERCENTILE of the active density, over an even sample of it."""
    values = []
    stride = max(source.shape[0] * source.shape[1] * source.shape[2] // REFERENCE_SAMPLE, 1)
    for start in range(0, source.shape[0], SLAB):
        slab = source.slab(start, min(start + SLAB, source.shape[0])).ravel()[::stride]
        values.append(slab[slab > FOOTPRINT_DENSITY])
    active = np.concatenate(values)
    return float(np.percentile(active, REFERENCE_PERCENTILE)) * density_factor if active.size else 1.0


def resample_and_erode(source: Source, density_factor: float, factor: float, noise: np.ndarray | None, strength: float) -> np.ndarray:
    """The density times `density_factor`, `factor` times finer per axis, as f16
    (z, y, x), eroded by `noise`. Output voxel i covers source coordinate
    (i + 0.5) / factor - 0.5, so that the two grids share their box."""
    shape = tuple(round(n * factor) for n in source.shape)
    out = np.empty(shape, np.float16)
    reference = erosion_reference(source, density_factor) if noise is not None else 1.0
    order = 3 if factor > 1.0 else 1
    coordinates = [(np.arange(n, dtype=np.float64) + 0.5) / factor - 0.5 for n in shape]
    y_index = np.arange(shape[1]) % NOISE_TILE
    x_index = np.arange(shape[2]) % NOISE_TILE
    for start in range(0, shape[0], SLAB):
        stop = min(start + SLAB, shape[0])
        if factor == 1.0:
            slab = source.slab(start, stop) * density_factor
        else:
            z = coordinates[0][start:stop]
            source_start = max(int(np.floor(z[0])) - RESAMPLE_MARGIN, 0)
            source_stop = min(int(np.ceil(z[-1])) + RESAMPLE_MARGIN + 1, source.shape[0])
            grid = np.meshgrid(z - source_start, coordinates[1], coordinates[2], indexing="ij")
            slab = ndimage.map_coordinates(source.slab(source_start, source_stop), grid, order=order, mode="nearest")
            slab = np.maximum(slab, 0.0) * density_factor
        if noise is not None:
            carve = strength * noise[np.ix_(np.arange(start, stop) % NOISE_TILE, y_index, x_index)]
            slab = np.maximum((slab / reference - carve) / (1.0 - carve), 0.0) * reference
        out[start:stop] = slab
    return out


def median_active(density: np.ndarray) -> float:
    """The median density over the voxels that hold cloud."""
    active = [density[start : start + SLAB][density[start : start + SLAB] > FOOTPRINT_DENSITY] for start in range(0, density.shape[0], SLAB)]
    active = np.concatenate(active).astype(np.float32)
    return float(np.median(active)) if active.size else 0.0


def median_column(density: np.ndarray, voxel: float) -> float:
    """The median vertical column over the cloud's footprint, in density times units."""
    columns = np.zeros((density.shape[0], density.shape[2]), np.float32)
    for start in range(0, density.shape[0], SLAB):
        columns[start : start + SLAB] = density[start : start + SLAB].astype(np.float32).sum(axis=1)
    footprint = columns[columns > FOOTPRINT_DENSITY]
    return float(np.median(footprint)) * voxel if footprint.size else 0.0


def scale(density: np.ndarray, factor: float) -> None:
    for start in range(0, density.shape[0], SLAB):
        density[start : start + SLAB] = density[start : start + SLAB].astype(np.float32) * factor


def majorants(density: np.ndarray) -> np.ndarray:
    cells = [-(-n // CELL) for n in density.shape]
    result = np.zeros(cells, np.float32)
    for c in range(cells[0]):
        start, stop = c * CELL, (c + 1) * CELL
        source_start = max(start - 1, 0)
        source = density[source_start : min(stop + 1, density.shape[0])].astype(np.float32)
        slab = np.zeros((CELL + 2, cells[1] * CELL + 2, cells[2] * CELL + 2), np.float32)
        z = 1 - (start - source_start)
        slab[z : z + source.shape[0], 1 : 1 + source.shape[1], 1 : 1 + source.shape[2]] = source
        grown = ndimage.maximum_filter(slab, size=3, mode="constant")[1:-1, 1:-1, 1:-1]
        result[c] = grown.reshape(CELL, cells[1], CELL, cells[2], CELL).max(axis=(0, 2, 4))
    return result


def edge_distance(density: np.ndarray, voxel: float, surface_density: float) -> np.ndarray:
    """Distance to the cloud's edge, in the grid's units, per coarse cell (z, y, x)."""
    k = DISTANCE_DOWNSAMPLE
    coarse_shape = [-(-n // k) for n in density.shape]
    inside = np.zeros(coarse_shape, bool)
    for c in range(coarse_shape[0]):
        source = density[c * k : (c + 1) * k].astype(np.float32)
        slab = np.zeros((k, coarse_shape[1] * k, coarse_shape[2] * k), np.float32)
        slab[: source.shape[0], : source.shape[1], : source.shape[2]] = source
        inside[c] = slab.reshape(k, coarse_shape[1], k, coarse_shape[2], k).mean(axis=(0, 2, 4)) > surface_density
    # A border of outside cells: the grid's faces bound the cloud.
    distance = ndimage.distance_transform_edt(np.pad(inside, 1), sampling=k * voxel)
    return distance[1:-1, 1:-1, 1:-1].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vdb", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--grid", default="density")
    parser.add_argument("--surface-density", type=float, default=0.1)
    parser.add_argument("--voxel", type=float, help="voxel size in the bench's units, instead of the file's")
    parser.add_argument("--density", type=float, default=1.0, help="factor on the file's density")
    parser.add_argument("--match-column", type=float, help="scale the density to this median column, in density times units")
    parser.add_argument("--match-density", type=float, help="scale the density to this median over the voxels holding cloud")
    parser.add_argument("--centre", action="store_true", help="put the grid's centre on x = z = 0")
    parser.add_argument("--z-up", action="store_true", help="the file's z axis is up")
    parser.add_argument("--resample", type=float, default=1.0, help="resample this many times finer per axis, below 1 coarser")
    parser.add_argument("--erode", type=float, default=0.0, help="strength of the noise carving the edges, below 1")
    parser.add_argument("--erode-period", type=int, default=32, help="the noise's largest feature, in output voxels")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.erode < 1.0:
        raise SystemExit("--erode must lie in [0, 1)")
    if NOISE_TILE % args.erode_period or args.erode_period >> (NOISE_OCTAVES - 1) < 1:
        raise SystemExit(f"--erode-period must divide {NOISE_TILE} and leave {NOISE_OCTAVES} octaves")

    if not args.resample > 0.0:
        raise SystemExit("--resample must be positive")
    source = Source(args.vdb, args.grid, args.z_up)
    voxel = args.voxel if args.voxel is not None else source.voxel / args.resample
    noise = worley_tile(args.erode_period, args.seed) if args.erode > 0.0 else None
    fine = resample_and_erode(source, args.density, args.resample, noise, args.erode)
    # The resampled grid shares the source's box, whose corner lies half a source voxel
    # before the source's first centre and half an output voxel before the output's.
    first_index = (source.first - 0.5) * args.resample + 0.5
    del source
    if args.centre:
        first_index[0], first_index[2] = -(fine.shape[2] // 2), -(fine.shape[0] // 2)
    if args.match_column is not None and args.match_density is not None:
        raise SystemExit("--match-column and --match-density exclude each other")
    column = median_column(fine, voxel)
    if args.match_column is not None and column > 0.0:
        scale(fine, args.match_column / column)
        column = args.match_column
    if args.match_density is not None:
        active = median_active(fine)
        if active > 0.0:
            scale(fine, args.match_density / active)
            column *= args.match_density / active
    cell_max = majorants(fine)
    distance = edge_distance(fine, voxel, args.surface_density)
    # VDB voxel centres sit at index * voxel, so the grid starts half a voxel before its first.
    min_corner = (first_index - 0.5) * voxel
    size_zyx, cells_zyx, distance_zyx = fine.shape, cell_max.shape, distance.shape
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
        out.write(fine.astype("<f2").tobytes())
        out.write(cell_max.astype("<f2").tobytes())
        out.write(distance.astype("<f2").tobytes())
    print(
        f"{args.out}: {size_zyx[::-1]} voxels of {voxel:.4f}, {cells_zyx[::-1]} cells, "
        f"min corner {min_corner}, median column {column:.1f}, deepest point {distance.max():.1f} units in"
    )


if __name__ == "__main__":
    main()
