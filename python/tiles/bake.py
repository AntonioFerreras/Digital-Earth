"""Bakes the streamed Earth tiles (docs/plans/20261020-streamed-earth-tiles.md).

Writes <data>/tiles/<layer>/face<f>.tiles from the sources in <data>/sources
(python/tiles/sources.py fetches them on first use). Levels 0 to 4 are baked
for the whole face; deeper levels for the whole face too, or only for tiles
that touch --region.

- height, levels 0 to 9: GEBCO_2025 heights to level 7. Levels 8 and 9 take
  land from GLO-90 and the sea floor from GEBCO, split by GLO-90's water-body
  mask, and exist only where a GLO-90 geocell lies under them. Water is the
  fraction of any water, ocean, lake, or river, from the water-body masks at
  every level (a missing geocell is ocean), so lakes agree across levels.
  Where a sample is all ocean its height is at most -1 m; lakes and rivers
  keep their surface. A level 8 or 9 tile that is all ocean is absent, so the
  runtime falls back to its level 7 ancestor.
- surface, levels 0 to 7: Blue Marble NG, July by default: the month whose
  colours the current equirectangular albedo map matches.

Every sample is a bilinear lookup in the source averaged over cells of at most
the level's sample spacing, so coarse levels are area averages, not point
samples. A sample depends only on its position and level, so tiles that share
an edge, on one face or across a face edge, bake identical edge samples.

GLO-90 DEMs are fetched for one group of tiles at a time (the descendants of a
level 2 tile), the next group's while the current one bakes, and dropped
after it, so the bake never holds more than two groups' DEMs.

The bake resumes: finished tiles of an interrupted run wait in
face<f>.tiles.parts/, tiles already in face<f>.tiles are kept, and the tiles
baked as absent are listed in face<f>.tiles.absent, so a repeated run bakes
nothing and a run with another region adds to the files.

Run: pixi run -e py python python/tiles/bake.py --data <root> --layer height [--region 5,44,12,48]
"""

import argparse
import multiprocessing
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from osgeo import gdal

import cube
import formats
import sources

LAST_LEVELS = {"height": 9, "surface": 7}
GLOBAL_LEVELS = 4
GEBCO_LAST_LEVEL = 7
GROUP_LEVEL = 2
# Cells read past a tile's bounds, so its outermost samples have all four
# bilinear neighbours.
WINDOW_MARGIN = 2
# Degrees past a tile's bounds whose GLO-90 geocells it reads: more than the
# window margin at levels 8 and 9 plus GLO-90's half-sample overhang.
GLO90_MARGIN = 0.01
PROGRESS_SECONDS = 60


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


def glo90_cells(bounds):
    """Names of the GLO-90 geocells under `bounds`, listed in the bucket or not."""
    lon0, lon1, lat0, lat1 = bounds
    return {sources.glo90_name(south, (west + 180) % 360 - 180)
            for south in range(max(-90, int(np.floor(lat0 - GLO90_MARGIN))),
                               min(89, int(np.floor(lat1 + GLO90_MARGIN))) + 1)
            for west in range(int(np.floor(lon0 - GLO90_MARGIN)), int(np.floor(lon1 + GLO90_MARGIN)) + 1)}


def plan(layer, face, last_level, region, glo90):
    """{(level, x, y): GLO-90 geocells it reads} of every tile a bake covers on one face.

    Levels to GLOBAL_LEVELS cover the face; deeper ones the tiles touching
    `region`, or all without one; height tiles past GEBCO_LAST_LEVEL only those
    over a geocell. A child's bounds lie in its parent's, so a tile that fails
    a test prunes its descendants.
    """
    keys = {}
    frontier = [(0, 0, 0)]
    while frontier:
        level, x, y = frontier.pop()
        deep = layer == "height" and level > GEBCO_LAST_LEVEL
        names = set()
        if deep or (region and level > GLOBAL_LEVELS):
            bounds = tile_bounds(layer, face, level, x, y)
            if region and level > GLOBAL_LEVELS and not touches(bounds, region):
                continue
            if deep:
                names = glo90_cells(bounds) & glo90
                if not names:
                    continue
        keys[(level, x, y)] = names
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
            self.gebco = sources.GlobalGrid([sources.prepare_gebco(data)])
            land, non_ocean = sources.prepare_glo90_masks(data)
            self.land = sources.GlobalGrid([land])
            self.non_ocean = sources.GlobalGrid([non_ocean])
        else:
            self.blue_marble = sources.GlobalGrid(sources.prepare_blue_marble(data, month), colour=True)

    def bake(self, face, level, x, y, names):
        bounds = tile_bounds(self.layer, face, level, x, y)
        factor = coarsening(level, sources.CELLS_PER_DEGREE)
        lon, lat = cube.direction_to_lonlat(cube.tile_directions(tile_coords(self.layer), face, level, x, y))
        if self.layer == "surface":
            rgb = grid_lookup(self.blue_marble, factor, bounds, lon, lat)
            return formats.Tile(formats.encode_surface(np.rint(sources.linear_to_srgb(rgb)).astype(np.uint8)))
        if level <= GEBCO_LAST_LEVEL:
            height = grid_lookup(self.gebco, factor, bounds, lon, lat)
            water = 255.0 - grid_lookup(self.land, factor, bounds, lon, lat)
            ocean = 255.0 - grid_lookup(self.non_ocean, factor, bounds, lon, lat)
        else:
            fused = self.glo90_lookup(level, bounds, lon, lat, names)
            if fused is None:
                return formats.Tile(b"")
            height, water, ocean = fused
        water = np.rint(water).astype(np.uint8)
        height = np.clip(np.rint(height), -32768, 32767).astype(np.int16)
        height = np.where(np.rint(ocean) >= 255, np.minimum(height, -1), height).astype(np.int16)
        inner = height[cube.BORDER:-cube.BORDER, cube.BORDER:-cube.BORDER]
        return formats.Tile(formats.encode_height(height, water), int(inner.min()), int(inner.max()))

    def glo90_lookup(self, level, bounds, lon, lat, names):
        """(height, water, ocean) at the samples from GLO-90 and GEBCO, water and
        ocean as fractions of 255; None where the tile is all ocean.

        The DEM and the mask are resampled to GLO-90's 3" lattice (nearest; exact
        below 50 degrees, where the geocells hold 1200 x 1200 samples), fused
        cell by cell, then averaged like the 15" sources.
        """
        factor = coarsening(level, sources.GLO90_CELLS_PER_DEGREE)
        native = 1.0 / sources.GLO90_CELLS_PER_DEGREE
        columns = 360 * sources.GLO90_CELLS_PER_DEGREE
        # Lattice cells centred on multiples of 3", grouped by `factor` from the
        # antimeridian so neighbouring tiles average the same blocks.
        west, north = -180.0 - native / 2, 90.0 + native / 2
        c0, c1, r0, r1 = window(bounds, factor * native, west, north)
        south_edge, north_edge = north - r1 * factor * native, north - r0 * factor * native
        dem_paths, wbm_paths = zip(*(sources.glo90_paths(self.data, name) for name in sorted(names)))
        # Holes between the geocells read as nodata, which the masks below count as ocean.
        dem_source = gdal.BuildVRT("", list(map(str, dem_paths)), resolution="highest", VRTNodata=np.nan)
        wbm_source = gdal.BuildVRT("", list(map(str, wbm_paths)), resolution="highest", VRTNodata=255)
        # A window across the antimeridian warps as one piece per side.
        dems, wbms = [], []
        start, end = c0 * factor, c1 * factor
        while start < end:
            lap = start // columns
            stop = min(end, (lap + 1) * columns)
            first, last = start - lap * columns, stop - lap * columns
            warp = dict(format="MEM", outputBounds=(west + first * native, south_edge, west + last * native,
                                                    north_edge),
                        xRes=native, yRes=native, resampleAlg="near", dstSRS="EPSG:4326")
            dems.append(gdal.Warp("", dem_source, dstNodata=np.nan, outputType=gdal.GDT_Float32,
                                  **warp).ReadAsArray())
            wbms.append(gdal.Warp("", wbm_source, dstNodata=255, outputType=gdal.GDT_Byte, **warp).ReadAsArray())
            start = stop
        dem, wbm = np.hstack(dems), np.hstack(wbms)
        ocean = (wbm == sources.WBM_OCEAN) | (wbm == 255) | np.isnan(dem)
        if ocean.all():
            return None
        water = ocean | (wbm == sources.WBM_LAKE) | (wbm == sources.WBM_RIVER)
        height = dem
        rows, cols = np.nonzero(ocean)
        if rows.size:
            sea_lon = west + (c0 * factor + cols + 0.5) * native
            sea_lat = north_edge - (rows + 0.5) * native
            height[rows, cols] = np.minimum(grid_lookup(self.gebco, 1, bounds, sea_lon, sea_lat), -0.5)
        cell = factor * native
        lookups = [bilinear(sources.box_average(plane, factor), c0, r0, cell, west, north, lon, lat)
                   for plane in (height, np.float32(255) * water, np.float32(255) * ocean)]
        if np.rint(lookups[2]).min() >= 255:
            return None
        return tuple(lookups)


BAKER = None


def start_worker(data, layer, month):
    global BAKER
    gdal.SetCacheMax(512 << 20)
    BAKER = Baker(data, layer, month)


def bake_tile(task):
    face, key, names = task
    return face, key, BAKER.bake(face, *key, names)


class Parts:
    """The finished tiles of a face not yet in its file: payloads appended to
    one pack, then a line per tile in an index, so a crash loses at most the
    tile being written."""

    def __init__(self, path):
        self.directory = path.with_name(path.name + ".parts")
        self.pack = self.directory / "payloads.bin"
        self.index = self.directory / "index.tsv"
        self.files = None

    def read(self):
        """{key: Tile} of the parts, absent tiles included (empty payloads)."""
        if not self.index.exists():
            return {}
        pack = formats.mapped(self.pack)
        tiles = {}
        for line in self.index.read_text().splitlines():
            fields = line.split()
            if len(fields) != 7:
                continue
            level, x, y, low, high, offset, size = map(int, fields)
            tiles[(level, x, y)] = formats.Tile(pack[offset:offset + size], low, high)
        return tiles

    def append(self, key, tile):
        if self.files is None:
            self.directory.mkdir(exist_ok=True)
            self.files = open(self.pack, "ab"), open(self.index, "a")
        pack, index = self.files
        offset = pack.tell()
        pack.write(tile.payload)
        pack.flush()
        index.write(" ".join(map(str, (*key, tile.min, tile.max, offset, len(tile.payload)))) + "\n")
        index.flush()

    def remove(self):
        if self.files:
            for file in self.files:
                file.close()
        shutil.rmtree(self.directory, ignore_errors=True)


def absent_path(path):
    return path.with_name(path.name + ".absent")


def read_absent(path):
    absent = absent_path(path)
    return {tuple(map(int, line.split())) for line in absent.read_text().splitlines()} if absent.exists() else set()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="data root holding sources/ and tiles/")
    parser.add_argument("--layer", choices=LAST_LEVELS, required=True)
    parser.add_argument("--faces", default="0,1,2,3,4,5", help="comma-separated faces (default: all)")
    parser.add_argument("--levels", type=int, help="the last level (default: the layer's last)")
    parser.add_argument("--region", help="lon0,lat0,lon1,lat1 in degrees, limiting the levels above "
                                         f"{GLOBAL_LEVELS} (default: the whole globe)")
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
    glo90 = sources.glo90_list(data) if args.layer == "height" else set()
    files = {}
    groups = {}
    for face in faces:
        path = output / f"face{face}.tiles"
        kept = formats.TileFile(path) if path.exists() else None
        parts = Parts(path)
        done = set(kept.present() if kept else ()) | set(parts.read()) | read_absent(path)
        files[face] = (path, max(last_level, kept.last_level if kept else 0), parts)
        for key, names in plan(args.layer, face, last_level, region, glo90).items():
            if key in done:
                continue
            level, x, y = key
            shift = max(0, level - GROUP_LEVEL)
            group = (face, formats.morton(x >> shift, y >> shift)) if names else (-1, 0)
            groups.setdefault(group, []).append((face, key, names))
    order = sorted(groups)
    total = sum(len(tasks) for tasks in groups.values())
    print(f"{total} tiles to bake on faces {faces} in {len(order)} groups, planned in "
          f"{time.perf_counter() - started:.0f} s", flush=True)

    def fetch(index):
        if index < len(order):
            sources.fetch_glo90(data, set().union(*(names for _, _, names in groups[order[index]])))

    baking = time.perf_counter()
    baked = 0
    reported = baking
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, start_worker, (data, args.layer, args.month)) as pool, \
            ThreadPoolExecutor(1) as fetcher:
        fetching = fetcher.submit(fetch, 0)
        for index, group in enumerate(order):
            fetching.result()
            fetching = fetcher.submit(fetch, index + 1)
            tasks = groups[group]
            for face, key, tile in pool.imap_unordered(bake_tile, tasks, chunksize=4):
                files[face][2].append(key, tile)
                baked += 1
                now = time.perf_counter()
                if now - reported > PROGRESS_SECONDS:
                    reported = now
                    rate = baked / (now - baking)
                    print(f"{baked}/{total} tiles, group {index + 1}/{len(order)}, {rate:.0f} tiles/s, "
                          f"ETA {(total - baked) / rate / 60:.0f} min", flush=True)
            if index + 1 < len(order):
                fetching.result()
                upcoming = set().union(*(names for _, _, names in groups[order[index + 1]]))
            else:
                upcoming = set()
            sources.drop_glo90_dems(data, set().union(*(names for _, _, names in tasks)) - upcoming)
    print(f"baked {baked} tiles in {time.perf_counter() - baking:.0f} s", flush=True)

    for face, (path, last, parts) in files.items():
        kept = formats.TileFile(path) if path.exists() else None
        new = parts.read()
        if kept and not new and kept.last_level == last:
            continue
        tiles = kept.present() if kept else {}
        tiles |= {key: tile for key, tile in new.items() if len(tile.payload)}
        absent = read_absent(path) | {key for key, tile in new.items() if not len(tile.payload)}
        formats.write_tiles(path, args.layer, face, last, tiles)
        absent_path(path).write_text("".join(f"{level} {x} {y}\n" for level, x, y in sorted(absent)))
        parts.remove()
        print(f"{path}: {len(tiles)} tiles, {len(absent)} absent, {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
