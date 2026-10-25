"""Writes the validation suite's views (TESTING.md) as viewer configs.

Cameras sit on an equal-area spiral of directions around the cloud's centre, cut off
below VIEW_ELEVATION_MIN so that the lowest stay above the ground. Each wide view turns
to the middle of the cloud's occupied majorant cells as it sees them, and opens its
field of view to take them all in with FRAME_PADDING to spare; every
CLOSE_UP_EVERY-th view is a close-up of CLOSE_UP_FOV_DEGREES on the cloud's centre.
Each direction is written once per sun of SUN_COS_ZENITHS, at SUN_AZIMUTH_DEGREES, with
the auto exposure of that sun (Post.autoExposure in src/render/Post.zig); every other
setting stays at its default, so that the default config is what the views render.

--preview renders the approximation of every view, with a field baked by the default
config, at PREVIEW_OFFSETS stops from its exposure, for judging the exposures.

Run: pixi run -e py python python/validation/views.py [--preview <dir>] [--volume <file.ecld>]
"""

import argparse
import math
import re
import tempfile
from pathlib import Path

import numpy as np

import validate

DIRECTION_COUNT = 6
CLOSE_UP_EVERY = 3
CLOSE_UP_FOV_DEGREES = 10.0
# Lower directions put the farther cameras below the ground.
VIEW_ELEVATION_MIN = math.radians(-10.0)
# Camera distance in radii of the cloud's bounding box.
DISTANCE_RADII = 3.0
# Of the wide views' half-angle, beyond the outermost occupied cell.
FRAME_PADDING = 0.02
# Evenly spaced from the lowest sun auto exposure knows (Post.auto_exposure_table) to
# overhead, all at one azimuth (Config.BenchSun).
SUN_COS_ZENITHS = tuple(float(c) for c in np.linspace(-0.07, 1.0, 4))
SUN_AZIMUTH_DEGREES = 65.0
PREVIEW_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PREVIEW_SIZE = 512
PREVIEW_SPP = 64
# scene.CloudBench.origin in src/render/scene.zig.
METRES_PER_UNIT = 10.0
BASE_ALTITUDE_M = 1000.0
PLANET_RADIUS_M = 6371e3


def occupied_cells(path: Path) -> tuple[np.ndarray, float]:
    """The centres of the occupied majorant cells, planet-centred metres, and their
    half diagonal."""
    header = np.fromfile(path, np.uint8, 64)
    size = header[8:20].view("<u4")
    cell = int(header[20:24].view("<u4")[0])
    cells = header[24:36].view("<u4")
    voxel = float(header[36:40].view("<f4")[0])
    min_corner = header[40:52].view("<f4").astype(np.float64)
    density_count = int(np.prod(size.astype(np.int64)))
    majorant = np.memmap(path, "<f2", "r", 64 + 2 * density_count, tuple(int(c) for c in cells[::-1]))
    z, y, x = np.nonzero(majorant > 0)
    edge = cell * voxel
    local = min_corner + (np.stack([x, y, z], -1) + 0.5) * edge
    origin = np.array([0.0, PLANET_RADIUS_M + BASE_ALTITUDE_M - min_corner[1] * METRES_PER_UNIT, 0.0])
    return origin + METRES_PER_UNIT * local, METRES_PER_UNIT * edge * math.sqrt(3) / 2


def spiral(count: int) -> np.ndarray:
    """Unit directions, equal-area over the sphere above VIEW_ELEVATION_MIN, +Y up."""
    low = math.sin(VIEW_ELEVATION_MIN)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(count) + 0.5
    y = low + (1.0 - low) * i / count
    radius = np.sqrt(1.0 - y * y)
    return np.stack([radius * np.cos(golden * i), y, radius * np.sin(golden * i)], -1)


def basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera.basis with up +Y: right, then up."""
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return right, np.cross(right, forward)


def frame(position: np.ndarray, forward: np.ndarray, points: np.ndarray, half_diagonal: float) -> tuple[np.ndarray, float]:
    """The forward and vertical field of view, in degrees, of a square view from
    `position` that holds every point, each grown by `half_diagonal`, turned twice to
    the middle of what it sees."""
    for _ in range(2):
        right, up = basis(forward)
        offset = points - position
        depth = offset @ forward
        pad = (half_diagonal / depth).max()
        across = [(offset @ axis) / depth for axis in (right, up)]
        low = [a.min() - pad for a in across]
        high = [a.max() + pad for a in across]
        forward = forward + 0.5 * (low[0] + high[0]) * right + 0.5 * (low[1] + high[1]) * up
        forward /= np.linalg.norm(forward)
    half = max(0.5 * (hi - lo) for lo, hi in zip(low, high)) * (1.0 + FRAME_PADDING)
    return forward, math.degrees(2.0 * math.atan(half))


def vector(v: np.ndarray) -> str:
    return ".{ " + ", ".join(f"{c:.9g}" for c in v) + " }"


def config_text(position: np.ndarray, forward: np.ndarray, fov: float, cos_zenith: float) -> str:
    return (
        ".{\n"
        "    .camera = .{\n"
        f"        .position = {vector(position)},\n"
        f"        .forward = {vector(forward)},\n"
        "        .up = .{ 0, 1, 0 },\n"
        f"        .vertical_fov_degrees = {fov:.6g},\n"
        "        .aspect_scale = 1,\n"
        "    },\n"
        f"    .bench_sun = .{{ .cos_zenith = {cos_zenith:.6g}, .azimuth_degrees = {SUN_AZIMUTH_DEGREES:g} }},\n"
        "    .mode = .cloud_bench,\n"
        "    .preset = .symmetric,\n"
        "}\n"
    )


def auto_exposures(configs: list[Path], scratch: Path) -> dict[str, float]:
    """Each config's auto exposure, by stem, as `earth render` reports it."""
    output = validate.render([(config, scratch / config.stem, "reference", 1) for config in configs], 16, None,
                             scratch / "meter.zon", log=True)
    return {Path(out).stem: float(e) for e, out in re.findall(r"auto exposure ([-\d.]+) -> (\S+)", output)}


def with_exposure(text: str, stops: float) -> str:
    text = re.sub(r"    \.exposure = [^\n]*\n", "", text)
    return text.replace("    .preset = .symmetric,\n", f"    .preset = .symmetric,\n    .exposure = {stops:.4f},\n")


def preview(paths: list[Path], out: Path) -> None:
    """The approximation of every view, with a field baked by the default config, at
    PREVIEW_OFFSETS stops from its exposure."""
    out.mkdir(parents=True, exist_ok=True)
    field = out / "field.eprb"
    print(f"field: {validate.bake(paths[0], field)}", flush=True)
    jobs = []
    for path in paths:
        stops = float(re.search(r"\.exposure = ([-\d.]+),", path.read_text()).group(1))
        for offset in PREVIEW_OFFSETS:
            # No dots in the stem: the outputs take theirs from the extension.
            exposed = out / (f"{path.stem}_{offset:+.2f}".replace(".", "p") + ".zon")
            exposed.write_text(with_exposure(path.read_text(), stops + offset))
            jobs.append((exposed, out / exposed.stem, "approximation", PREVIEW_SPP))
    validate.render(jobs, PREVIEW_SIZE, field, out / "jobs.zon")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", type=Path, default=Path("cache/clouds/wdas_cloud_half.ecld"))
    parser.add_argument("--out", type=Path, default=Path("tests/validation/views"))
    parser.add_argument("--preview", type=Path, help="render the views' exposure brackets here")
    args = parser.parse_args()

    points, half_diagonal = occupied_cells(args.volume)
    low, high = points.min(0), points.max(0)
    centre = 0.5 * (low + high)
    distance = DISTANCE_RADII * 0.5 * float(np.linalg.norm(high - low))
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("view_*.zon"):
        stale.unlink()
    paths = []
    for index, direction in enumerate(spiral(DIRECTION_COUNT)):
        position = centre + distance * direction
        assert np.linalg.norm(position) > PLANET_RADIUS_M, "a camera lies below the ground"
        close_up = index % CLOSE_UP_EVERY == CLOSE_UP_EVERY - 1
        forward, fov = (-direction, CLOSE_UP_FOV_DEGREES) if close_up else frame(position, -direction, points, half_diagonal)
        elevation = math.degrees(math.asin(direction[1]))
        print(f"view {index}: elevation {elevation:6.1f}, {'close-up' if close_up else 'wide'}, fov {fov:.2f}")
        for sun_index, cos_zenith in enumerate(SUN_COS_ZENITHS):
            path = args.out / f"view_{index:02d}_sun{sun_index}.zon"
            path.write_text(config_text(position, forward, fov, cos_zenith))
            paths.append(path)
    with tempfile.TemporaryDirectory() as scratch:
        exposures = auto_exposures(paths, Path(scratch))
    for path in paths:
        path.write_text(with_exposure(path.read_text(), exposures[path.stem]))
        print(f"{path}: exposure {exposures[path.stem]:.4f}")
    if args.preview is not None:
        preview(paths, args.preview)


if __name__ == "__main__":
    main()
