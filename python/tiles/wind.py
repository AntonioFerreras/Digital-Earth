"""Bakes the 10 m wind of one ERA5 hour into <data>/tiles/wind.raw.

The wind roughens the water: the renderer reads its speed for the sea's slope
variance and its direction for the streaks the roughness forms along it. The
source is ERA5's hourly 10 m wind on its 0.25 degree grid, from the public
ARCO-ERA5 copy (Copernicus Climate Change Service; contains modified Copernicus
Climate Change Service information), fetched into <data>/sources/era5 on first
use. One hour rather than a monthly mean keeps the weather's calms and gusts,
which a mean averages away.

wind.raw: magic "EWND" | version u32 | width u32 | height u32, then RGBA8
cells, south row first, columns west to east from 180 degrees west, each cell
centred between four ERA5 points: R speed / SPEED_MAX, G and B the eastward and
northward components as (component / SPEED_MAX + 1) / 2, A 255.

Run: pixi run -e py python python/tiles/wind.py --data <root> [--date 2023-07-15 --hour 12]
"""

import argparse
import struct
import urllib.request
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/raw/date-variable-single_level"
COMPONENTS = {"u": "10m_u_component_of_wind", "v": "10m_v_component_of_wind"}
# Above any 10 m wind ERA5 resolves over the sea; faster cells clamp.
SPEED_MAX = 32.0
WIDTH, HEIGHT = 1440, 720
HEADER = struct.Struct("<4s3I")
VERSION = 1


def fetch(sources, date, variable):
    """The day's NetCDF of `variable`, downloaded first if missing."""
    path = sources / f"{variable}.{date}.nc"
    if not path.exists():
        year, month, day = date.split("-")
        print(f"fetching {path.name}", flush=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{ARCO}/{year}/{month}/{day}/{variable}/surface.nc", part)
        part.rename(path)
    return path


def read_hour(path, hour):
    """The hour's field in m/s, rows from 90 degrees north, columns east from 0 degrees."""
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(hour + 1)
    return band.ReadAsArray().astype(np.float64) * band.GetScale() + band.GetOffset()


def to_cells(points):
    """Averages ERA5's 721 x 1440 points into the 720 x 1440 cells between them,
    south row first and columns from 180 degrees west."""
    south_first = points[::-1]
    rows = 0.5 * (south_first[:-1] + south_first[1:])
    west_first = np.roll(rows, WIDTH // 2, axis=1)
    return 0.5 * (west_first + np.roll(west_first, -1, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="data root holding sources/ and tiles/")
    parser.add_argument("--date", default="2023-07-15", help="UTC day, YYYY-MM-DD (default: 2023-07-15)")
    parser.add_argument("--hour", type=int, default=12, choices=range(24), help="UTC hour (default: 12)")
    args = parser.parse_args()

    sources = args.data / "sources" / "era5"
    sources.mkdir(parents=True, exist_ok=True)
    u, v = (read_hour(fetch(sources, args.date, variable), args.hour) for variable in COMPONENTS.values())
    speed = to_cells(np.hypot(u, v))
    u, v = to_cells(u), to_cells(v)
    assert speed.shape == (HEIGHT, WIDTH)

    cells = np.empty((HEIGHT, WIDTH, 4), np.uint8)
    cells[..., 0] = np.round(np.clip(speed / SPEED_MAX, 0, 1) * 255)
    cells[..., 1] = np.round(np.clip((u / SPEED_MAX + 1) / 2, 0, 1) * 255)
    cells[..., 2] = np.round(np.clip((v / SPEED_MAX + 1) / 2, 0, 1) * 255)
    cells[..., 3] = 255
    out = args.data / "tiles" / "wind.raw"
    out.write_bytes(HEADER.pack(b"EWND", VERSION, WIDTH, HEIGHT) + cells.tobytes())
    print(f"{out}: {args.date} {args.hour:02d}:00 UTC, speed mean {speed.mean():.1f} m/s, max {speed.max():.1f} m/s")


if __name__ == "__main__":
    main()
