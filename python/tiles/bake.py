"""Bakes the streamed Earth tiles (docs/plans/20261020-streamed-earth-tiles.md).

Writes <data>/tiles/<layer>/face<f>.tiles from the sources in <data>/sources
(python/tiles/sources.py fetches them on first use). Levels 0 to 4 are baked
for the whole face; deeper levels only for tiles that touch --region.

- height, levels 0 to 9: GEBCO_2025 to level 7. Levels 8 and 9 take land from
  GLO-90 and the sea floor from GEBCO, split by GLO-90's water-body mask.
  Water is the fraction of any water, ocean, lake, or river: GEBCO below
  sea level to level 7, the mask's water codes or a missing geocell at 8 and 9.
  Where a sample is all ocean its height is at most -1 m; lakes and rivers
  keep the DEM's surface. A level 8 or 9 tile that is all ocean is absent, so
  the runtime falls back to its level 7 ancestor.
- surface, levels 0 to 7: Blue Marble NG, July by default: the month whose
  colours the current equirectangular albedo map matches.

Every sample is a bilinear lookup in the source averaged over cells of at most
the level's sample spacing, so coarse levels are area averages, not point
samples. A sample depends only on its position and level, so tiles that share
an edge, on one face or across a face edge, bake identical edge samples.

The bake resumes: finished tiles of an interrupted run wait in
face<f>.tiles.parts/, and tiles already in face<f>.tiles are kept, so a later
run with another region adds to the files.

Run: pixi run -e py python python/tiles/bake.py --data <root> --layer height --levels 9 --region 5,44,12,48
"""

import argparse
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm

import cube
import formats
import sources

LAST_LEVELS = {"height": 9, "surface": 7}
GLOBAL_LEVELS = 4
GEBCO_LAST_LEVEL = 7
# Cells read past a tile's bounds, so its outermost samples have all four
# bilinear neighbours.
WINDOW_MARGIN = 2


def tile_coords(layer):
    return cube.height_coords if layer == "height" else cube.surface_coords


def tile_bounds(layer, face, level, x, y):
    """(lon0, lon1, lat0, lat1) in degrees around a tile's bordered samples.

    Longitudes are unwrapped around the tile's centre, so lon0 may be below
    -180 or lon1 above 180; a tile around a pole spans all longitudes.
    """
    coords = tile_coords(layer)
    u, v = coords(level, x), coords(level, y)
    edge_u = np.concatenate([u, u, np.full(v.size, u[0]), np.full(v.size, u[-1])])
    edge_v = np.concatenate([np.full(u.size, v[0]), np.full(u.size, v[-1]), v, v])
    lon, lat = cube.direction_to_lonlat(cube.face_uv_to_direction(np.full(edge_u.size, face), edge_u, edge_v))
    centre_lon, _ = cube.direction_to_lonlat(cube.face_uv_to_direction(face, u.mean(), v.mean()))
    lon = centre_lon + (lon - centre_lon + 180.0) % 360.0 - 180.0
    if face in (2, 3) and u[0] <= 0.5 <= u[-1] and v[0] <= 0.5 <= v[-1]:
        return (-180.0, 180.0, lat.min(), 90.0) if face == 2 else (-180.0, 180.0, -90.0, lat.max())
    return lon.min(), lon.max(), lat.min(), lat.max()


def touches(bounds, region):
    lon0, lon1, lat0, lat1 = bounds
    west, south, east, north = region
    if lat1 < south or lat0 > north:
        return False
    return any(lon0 <= east + shift and lon1 >= west + shift for shift in (-360.0, 0.0, 360.0))


def global_keys(last_level):
    return [(level, x, y) for level in range(min(last_level, GLOBAL_LEVELS) + 1)
            for y in range(1 << level) for x in range(1 << level)]


def glo90_cells(bounds):
    """Names of the GLO-90 geocells within a degree of `bounds`, listed in the bucket or not."""
    lon0, lon1, lat0, lat1 = bounds
    return {sources.glo90_name(south, (west + 180) % 360 - 180)
            for south in range(max(-90, int(np.floor(lat0)) - 1), min(89, int(np.floor(lat1)) + 1) + 1)
            for west in range(int(np.floor(lon0)) - 1, int(np.floor(lon1)) + 2)}


def region_keys(layer, face, last_level, region):
    """Tiles below GLOBAL_LEVELS that touch `region`, found by descending from
    the touching tiles of GLOBAL_LEVELS (a child's bounds lie in its parent's)."""
    keys = []
    frontier = [(GLOBAL_LEVELS, x, y) for y in range(1 << GLOBAL_LEVELS) for x in range(1 << GLOBAL_LEVELS)]
    while frontier:
        level, x, y = frontier.pop()
        if not touches(tile_bounds(layer, face, level, x, y), region):
            continue
        if level > GLOBAL_LEVELS:
            keys.append((level, x, y))
        if level < last_level:
            frontier += [(level + 1, 2 * x + i, 2 * y + j) for j in range(2) for i in range(2)]
    return keys


def coarsening(level, cells_per_degree):
    """The largest power of two of source cells within the level's sample spacing."""
    cells = np.degrees(cube.sample_spacing(level)) * cells_per_degree
    return 1 << max(0, int(np.floor(np.log2(cells))))


def window(bounds, cell, west, north):
    """Cell range (c0, c1, r0, r1) of a grid of `cell` degrees from (west, north)
    covering `bounds` with a margin for bilinear lookups."""
    lon0, lon1, lat0, lat1 = bounds
    c0 = int(np.floor((lon0 - west) / cell)) - WINDOW_MARGIN
    c1 = int(np.ceil((lon1 - west) / cell)) + WINDOW_MARGIN
    r0 = int(np.floor((north - lat1) / cell)) - WINDOW_MARGIN
    r1 = int(np.ceil((north - lat0) / cell)) + WINDOW_MARGIN
    return c0, c1, r0, r1


def bilinear(grid, c0, r0, cell, west, north, lon, lat):
    """Bilinear lookup at (lon, lat) in `grid`, the cells from (c0, r0) of a
    grid of `cell` degrees whose cell (0, 0) has its corner at (west, north)."""
    columns = round(360.0 / cell)
    x = ((lon - west) / cell - 0.5 - c0) % columns
    y = np.clip((north - lat) / cell - 0.5 - r0, 0.0, grid.shape[0] - 1.0)
    assert x.max() <= grid.shape[1] - 1, "window misses a sample"
    x0 = np.minimum(np.floor(x).astype(int), grid.shape[1] - 2)
    y0 = np.minimum(np.floor(y).astype(int), grid.shape[0] - 2)
    fx, fy = x - x0, y - y0
    if grid.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx
    bottom = grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def grid_lookup(grid, factor, bounds, lon, lat):
    """Bilinear lookups in a GlobalGrid coarsened by `factor`."""
    cell = factor / sources.CELLS_PER_DEGREE
    c0, c1, r0, r1 = window(bounds, cell, -180.0, 90.0)
    return bilinear(grid.read(factor, c0, c1, r0, r1), c0, r0, cell, -180.0, 90.0, lon, lat)


class Baker:
    """The per-process state of a bake: open sources and the bake's settings."""

    def __init__(self, data, layer, month):
        self.data = data
        self.layer = layer
        if layer == "height":
            heights, water = sources.prepare_gebco(data)
            self.gebco = sources.GlobalGrid([heights])
            self.gebco_water = sources.GlobalGrid([water])
            self.glo90 = sources.glo90_list(data)
        else:
            self.blue_marble = sources.GlobalGrid(sources.prepare_blue_marble(data, month), colour=True)

    def bake(self, face, level, x, y):
        bounds = tile_bounds(self.layer, face, level, x, y)
        factor = coarsening(level, sources.CELLS_PER_DEGREE)
        lon, lat = cube.direction_to_lonlat(cube.tile_directions(tile_coords(self.layer), face, level, x, y))
        if self.layer == "surface":
            rgb = grid_lookup(self.blue_marble, factor, bounds, lon, lat)
            return formats.Tile(formats.encode_surface(np.rint(sources.linear_to_srgb(rgb)).astype(np.uint8)))
        if level <= GEBCO_LAST_LEVEL:
            height = grid_lookup(self.gebco, factor, bounds, lon, lat)
            water = grid_lookup(self.gebco_water, factor, bounds, lon, lat)
            ocean = water
        else:
            fused = self.glo90_lookup(level, bounds, lon, lat)
            if fused is None:
                return formats.Tile(b"")
            height, water, ocean = fused
        water = np.rint(water).astype(np.uint8)
        height = np.clip(np.rint(height), -32768, 32767).astype(np.int16)
        height = np.where(np.rint(ocean) >= 255, np.minimum(height, -1), height).astype(np.int16)
        inner = height[cube.BORDER:-cube.BORDER, cube.BORDER:-cube.BORDER]
        return formats.Tile(formats.encode_height(height, water), int(inner.min()), int(inner.max()))

    def glo90_lookup(self, level, bounds, lon, lat):
        """(height, water, ocean) at the samples from GLO-90 and GEBCO, water and
        ocean as fractions of 255; None where the tile is all ocean.

        The DEM and the mask are resampled to GLO-90's 3" lattice (nearest; exact
        below 50 degrees, where the geocells hold 1200 x 1200 samples), fused
        cell by cell, then averaged like the 15" sources.
        """
        names = sorted(glo90_cells(bounds) & self.glo90)
        if not names:
            return None
        factor = coarsening(level, sources.GLO90_CELLS_PER_DEGREE)
        native = 1.0 / sources.GLO90_CELLS_PER_DEGREE
        # Lattice cells centred on multiples of 3", grouped by `factor` from the
        # antimeridian so neighbouring tiles average the same blocks.
        west, north = -180.0 - native / 2, 90.0 + native / 2
        c0, c1, r0, r1 = window(bounds, factor * native, west, north)
        warp_bounds = (west + c0 * factor * native, north - r1 * factor * native,
                       west + c1 * factor * native, north - r0 * factor * native)
        assert -180 <= warp_bounds[0] and warp_bounds[2] <= 180, "GLO-90 windows across the antimeridian"
        dem_paths, wbm_paths = zip(*(sources.glo90_paths(self.data, name) for name in names))
        warp = dict(format="MEM", outputBounds=warp_bounds, xRes=native, yRes=native, resampleAlg="near",
                    dstSRS="EPSG:4326")
        dem = gdal.Warp("", gdal.BuildVRT("", list(map(str, dem_paths))), dstNodata=np.nan, outputType=gdal.GDT_Float32,
                        **warp).ReadAsArray()
        wbm = gdal.Warp("", gdal.BuildVRT("", list(map(str, wbm_paths))), dstNodata=255, outputType=gdal.GDT_Byte,
                        **warp).ReadAsArray()
        ocean = (wbm == sources.WBM_OCEAN) | (wbm == 255) | np.isnan(dem)
        if ocean.all():
            return None
        water = ocean | (wbm == sources.WBM_LAKE) | (wbm == sources.WBM_RIVER)
        rows, columns = dem.shape
        centre_lon = warp_bounds[0] + (np.arange(columns) + 0.5) * native
        centre_lat = warp_bounds[3] - (np.arange(rows) + 0.5) * native
        lattice_lon, lattice_lat = np.meshgrid(centre_lon, centre_lat)
        sea_floor = grid_lookup(self.gebco, 1, bounds, lattice_lon, lattice_lat)
        height = np.where(ocean, np.minimum(sea_floor, -0.5), dem)
        averaged = [sources.box_average(plane.astype(np.float32), factor) for plane in (height, 255.0 * water,
                                                                                         255.0 * ocean)]
        cell = factor * native
        lookups = [bilinear(plane, c0, r0, cell, west, north, lon, lat) for plane in averaged]
        if np.rint(lookups[2]).min() >= 255:
            return None
        return tuple(lookups)


BAKER = None


def start_worker(data, layer, month):
    global BAKER
    gdal.SetCacheMax(512 << 20)
    BAKER = Baker(data, layer, month)


def bake_tile(task):
    face, key = task
    return face, key, BAKER.bake(face, *key)


def parts_directory(path):
    return path.with_name(path.name + ".parts")


def read_parts(path):
    """{key: Tile} of the finished tiles of an interrupted bake, absent ones included."""
    directory = parts_directory(path)
    tiles = {}
    index = directory / "index.tsv"
    if index.exists():
        for line in index.read_text().splitlines():
            level, x, y, low, high, size = map(int, line.split())
            payload = (directory / f"{level}-{x}-{y}.bin").read_bytes() if size else b""
            assert len(payload) == size
            tiles[(level, x, y)] = formats.Tile(payload, low, high)
    return tiles


def write_part(path, key, tile):
    directory = parts_directory(path)
    directory.mkdir(exist_ok=True)
    level, x, y = key
    if tile.payload:
        (directory / f"{level}-{x}-{y}.bin").write_bytes(tile.payload)
    with open(directory / "index.tsv", "a") as index:
        index.write(f"{level} {x} {y} {tile.min} {tile.max} {len(tile.payload)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="data root holding sources/ and tiles/")
    parser.add_argument("--layer", choices=LAST_LEVELS, required=True)
    parser.add_argument("--faces", default="0,1,2,3,4,5", help="comma-separated faces (default: all)")
    parser.add_argument("--levels", type=int, help="the last level (default: the layer's last)")
    parser.add_argument("--region", help="lon0,lat0,lon1,lat1 in degrees, limiting the levels above "
                                         f"{GLOBAL_LEVELS} (default: none of them)")
    parser.add_argument("--month", type=int, default=7, help="Blue Marble month, 1 to 12 (default: 7)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    last_level = LAST_LEVELS[args.layer] if args.levels is None else args.levels
    if not 0 <= last_level <= LAST_LEVELS[args.layer]:
        parser.error(f"--levels: the {args.layer} layer has levels 0 to {LAST_LEVELS[args.layer]}")
    region = tuple(map(float, args.region.split(","))) if args.region else None
    faces = [int(face) for face in args.faces.split(",")]
    data = args.data.resolve()
    output = data / "tiles" / args.layer
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    Baker(data, args.layer, args.month)
    plans = {}
    tasks = []
    for face in faces:
        path = output / f"face{face}.tiles"
        kept = formats.TileFile(path) if path.exists() else None
        done = (kept.present() if kept else {}) | read_parts(path)
        keys = global_keys(last_level)
        if region and last_level > GLOBAL_LEVELS:
            keys += region_keys(args.layer, face, last_level, region)
        plans[face] = (path, max(last_level, kept.last_level if kept else 0))
        tasks += [(face, key) for key in keys if key not in done]
    if args.layer == "height":
        names = set()
        for face, (level, x, y) in tasks:
            if level > GEBCO_LAST_LEVEL:
                names |= glo90_cells(tile_bounds(args.layer, face, level, x, y))
        sources.fetch_glo90(data, names & sources.glo90_list(data))
    print(f"{len(tasks)} tiles to bake on faces {faces}, sources ready in {time.perf_counter() - started:.0f} s")

    # Deep levels first: their tiles are the most numerous and the slowest.
    tasks.sort(key=lambda task: -task[1][0])
    baking = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, start_worker, (data, args.layer, args.month)) as pool:
        for face, key, tile in tqdm(pool.imap_unordered(bake_tile, tasks, chunksize=4), total=len(tasks)):
            write_part(plans[face][0], key, tile)
    print(f"baked in {time.perf_counter() - baking:.0f} s")

    for face, (path, last) in plans.items():
        tiles = formats.TileFile(path).present() if path.exists() else {}
        tiles |= {key: tile for key, tile in read_parts(path).items() if tile.payload}
        formats.write_tiles(path, args.layer, face, last, tiles)
        shutil.rmtree(parts_directory(path), ignore_errors=True)
        print(f"{path}: {len(tiles)} tiles, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
