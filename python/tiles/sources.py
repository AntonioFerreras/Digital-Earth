"""Source rasters of the tile bake: fetched once into <data>/sources, prepared
into tiled GeoTIFFs with average overviews, and read as area averages.

- GEBCO_2025 (sources/gebco): the 15" global grid, ice surface elevation.
  Prepared into `gebco.tif` with average overviews.
- Blue Marble NG `world.2004MM` (sources/bluemarble): eight 500 m JPEG tiles
  per month, on the same 15" grid as GEBCO. Prepared into one GeoTIFF per
  tile whose overviews average linear, not encoded, colour.
- Copernicus GLO-90 (sources/glo90): 1-degree DEM and water-body mask COGs,
  copied from the public bucket. Every mask is kept and
  prepared into land and non-ocean fractions on the 15" grid; the DEMs are
  fetched in batches and dropped once their tiles are baked. The bucket has
  no geocell without land, which reads as ocean.

Both 15" sources are area-registered: cell (row, col) spans longitude
-180 + col / 240 and latitude 90 - row / 240 onward. GLO-90 is
point-registered: its sample centres sit on multiples of 3".
"""

import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

GEBCO_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/gebco_2025.zip?download=1"
BLUE_MARBLE_URL = "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base"
GLO90_URL = "https://copernicus-dem-90m.s3.amazonaws.com"
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")
BLUE_MARBLE_TILES = ("A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2")

CELLS_PER_DEGREE = 240
GLOBAL_COLUMNS = 360 * CELLS_PER_DEGREE
GLOBAL_ROWS = 180 * CELLS_PER_DEGREE
# Overview levels kept by the prepared rasters: factors 2 to 32 divide the
# 21600-cell Blue Marble tiles and the 43200 rows exactly.
OVERVIEW_LEVELS = 5
GLO90_CELLS_PER_DEGREE = 1200
# GLO-90 water-body mask codes (Copernicus DEM Product Handbook 5.0, table 8).
WBM_OCEAN, WBM_LAKE, WBM_RIVER = 1, 2, 3

CREATION = ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=ZSTD", "PREDICTOR=2", "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS"]


def download(url, path):
    """Fetches `url` to `path` through a partial file, resuming an interrupted transfer."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"downloading {url}")
    subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", str(partial), url], check=True)
    partial.rename(path)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def box_average(a, factor):
    """Means of factor x factor blocks over the first two axes."""
    if factor == 1:
        return a
    rows, columns = a.shape[0] // factor, a.shape[1] // factor
    return a.reshape(rows, factor, columns, factor, *a.shape[2:]).mean(axis=(1, 3))


def write_raster(path, columns, rows, bands, data_type, west, north, cells_per_degree, strips, colour=False):
    """Writes a tiled GeoTIFF from (row, array) strips and its average overviews.

    Colour overviews average in linear space; strips must then hold a multiple
    of 2^OVERVIEW_LEVELS rows.
    """
    temporary = path.with_name(path.name + ".tmp.tif")
    dataset = gdal.GetDriverByName("GTiff").Create(str(temporary), columns, rows, bands, data_type, CREATION)
    dataset.SetGeoTransform((west, 1 / cells_per_degree, 0, north, 0, -1 / cells_per_degree))
    dataset.SetProjection("EPSG:4326")
    if colour:
        dataset.BuildOverviews("NONE", [2**k for k in range(1, OVERVIEW_LEVELS + 1)])
    for row, strip in strips:
        strip = strip.reshape(strip.shape[0], strip.shape[1], -1)
        for band in range(bands):
            dataset.GetRasterBand(band + 1).WriteArray(strip[..., band], 0, row)
        if colour:
            assert row % (1 << OVERVIEW_LEVELS) == 0 and strip.shape[0] % (1 << OVERVIEW_LEVELS) == 0
            linear = srgb_to_linear(strip)
            for k in range(1, OVERVIEW_LEVELS + 1):
                linear = box_average(linear, 2)
                encoded = np.rint(linear_to_srgb(linear)).astype(np.uint8)
                for band in range(bands):
                    dataset.GetRasterBand(band + 1).GetOverview(k - 1).WriteArray(encoded[..., band], 0, row >> k)
    if not colour:
        dataset.BuildOverviews("AVERAGE", [2**k for k in range(1, OVERVIEW_LEVELS + 1)])
    dataset = None
    temporary.rename(path)


def prepare_gebco(data):
    """Returns the prepared heights GeoTIFF, fetching GEBCO first if needed.

    The netCDF is deleted once prepared: the GeoTIFF and its overviews hold
    everything the bake reads.
    """
    directory = data / "sources" / "gebco"
    heights = directory / "gebco.tif"
    if heights.exists():
        return heights
    netcdf = directory / "GEBCO_2025.nc"
    if not netcdf.exists():
        archive = directory / "gebco_2025.zip"
        download(GEBCO_URL, archive)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extract("GEBCO_2025.nc", directory)
        archive.unlink()
    source = gdal.Open(str(netcdf))
    assert (source.RasterXSize, source.RasterYSize) == (GLOBAL_COLUMNS, GLOBAL_ROWS)
    band = source.GetRasterBand(1)
    strip_rows = 1350
    print("preparing gebco.tif")
    write_raster(heights, GLOBAL_COLUMNS, GLOBAL_ROWS, 1, gdal.GDT_Int16, -180, 90, CELLS_PER_DEGREE,
                 ((row, band.ReadAsArray(0, row, GLOBAL_COLUMNS, strip_rows))
                  for row in range(0, GLOBAL_ROWS, strip_rows)))
    netcdf.unlink()
    return heights


def prepare_blue_marble(data, month):
    """Returns the prepared GeoTIFF of each Blue Marble tile, fetching the month's JPEGs first."""
    directory = data / "sources" / "bluemarble"
    prepared = []
    for tile in BLUE_MARBLE_TILES:
        name = f"world.2004{month:02d}.3x21600x21600.{tile}"
        path = directory / f"{name}.tif"
        prepared.append(path)
        if path.exists():
            continue
        jpeg = directory / f"{name}.jpg"
        download(f"{BLUE_MARBLE_URL}/{MONTHS[month - 1]}/{name}.jpg", jpeg)
        print(f"preparing {path.name}")
        source = gdal.Open(str(jpeg))
        size = source.RasterXSize
        assert source.RasterYSize == size == 21600 and source.RasterCount == 3
        west = -180 + 90 * "ABCD".index(tile[0])
        north = 90 - 90 * (int(tile[1]) - 1)
        strip_rows = 864
        strips = ((row, source.ReadAsArray(0, row, size, strip_rows).transpose(1, 2, 0))
                  for row in range(0, size, strip_rows))
        write_raster(path, size, size, 3, gdal.GDT_Byte, west, north, CELLS_PER_DEGREE, strips, colour=True)
    return prepared


class GlobalGrid:
    """Files placed on the global 15" grid, read as averages of f x f cells.

    `read` returns the cells [c0, c1) x [r0, r1) of the grid coarsened by the
    power of two `factor`. Columns wrap around the antimeridian and rows clamp
    at the poles. Colour reads return linear RGB.
    """

    def __init__(self, paths, colour=False):
        self.colour = colour
        self.files = []
        for path in paths:
            dataset = gdal.Open(str(path))
            west, step, _, north, _, _ = dataset.GetGeoTransform()
            assert round(step * CELLS_PER_DEGREE, 9) == 1.0
            column = round((west + 180) * CELLS_PER_DEGREE)
            row = round((90 - north) * CELLS_PER_DEGREE)
            self.files.append((dataset, column, row, dataset.RasterXSize, dataset.RasterYSize))

    def read(self, factor, c0, c1, r0, r1):
        k = min(int(np.log2(factor)), OVERVIEW_LEVELS)
        scale = 1 << k
        m = factor >> k
        columns, rows = GLOBAL_COLUMNS // scale, GLOBAL_ROWS // scale
        column_index = np.arange(c0 * m, c1 * m) % columns
        row_index = np.clip(np.arange(r0 * m, r1 * m), 0, rows - 1)
        top, bottom = row_index.min(), row_index.max() + 1
        bands = 3 if self.colour else 1
        out = np.full((bottom - top, len(column_index), bands), np.nan, dtype=np.float32)
        # Contiguous runs of wrapped columns, each read straight from the files.
        breaks = np.flatnonzero(np.diff(column_index) != 1) + 1
        for run in np.split(np.arange(len(column_index)), breaks):
            first, last = column_index[run[0]], column_index[run[-1]] + 1
            for dataset, file_column, file_row, width, height in self.files:
                fc0, fr0 = file_column // scale, file_row // scale
                fc1, fr1 = fc0 + width // scale, fr0 + height // scale
                x0, x1 = max(first, fc0), min(last, fc1)
                y0, y1 = max(top, fr0), min(bottom, fr1)
                if x0 >= x1 or y0 >= y1:
                    continue
                for band in range(bands):
                    raster = dataset.GetRasterBand(band + 1)
                    if k:
                        raster = raster.GetOverview(k - 1)
                    block = raster.ReadAsArray(int(x0 - fc0), int(y0 - fr0), int(x1 - x0), int(y1 - y0))
                    at = run[0] + x0 - first
                    out[y0 - top:y1 - top, at:at + x1 - x0, band] = srgb_to_linear(block) if self.colour else block
        out = out[row_index - top]
        return box_average(out, m)[..., 0] if bands == 1 else box_average(out, m)


def glo90_name(lat, lon):
    """The bucket's name of the geocell whose south-west corner is (lat, lon)."""
    return (f"Copernicus_DSM_COG_30_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")


def glo90_corner(name):
    """(lat, lon) of the south-west corner of geocell `name`."""
    lat, lon = name.split("_")[4], name.split("_")[6]
    return (int(lat[1:]) * (1 if lat[0] == "N" else -1), int(lon[1:]) * (1 if lon[0] == "E" else -1))


def glo90_list(data):
    path = data / "sources" / "glo90" / "tileList.txt"
    download(f"{GLO90_URL}/tileList.txt", path)
    return set(path.read_text().split())


def glo90_paths(data, name):
    """Local (DEM, water-body mask) paths of geocell `name`."""
    directory = data / "sources" / "glo90" / name
    return directory / f"{name}.tif", directory / f"{name.removesuffix('_DEM')}_WBM.tif"


def copy_remote(key, path):
    """Copies the bucket's `key` to `path` through a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    subprocess.run(["curl", "-sSf", "--retry", "5", "--retry-all-errors", "-o", str(partial), f"{GLO90_URL}/{key}"],
                   check=True)
    partial.rename(path)


def fetch_glo90(data, names, dem=True, workers=32):
    """Copies the water-body mask, and the DEM if `dem`, of each geocell in `names`
    from the public bucket, many at a time."""
    copies = []
    for name in names:
        dem_path, wbm_path = glo90_paths(data, name)
        copies.append((f"{name}/AUXFILES/{wbm_path.name}", wbm_path))
        if dem:
            copies.append((f"{name}/{dem_path.name}", dem_path))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda copy: copy_remote(*copy), [copy for copy in copies if not copy[1].exists()]))


def drop_glo90_dems(data, names):
    for name in names:
        glo90_paths(data, name)[0].unlink(missing_ok=True)


def prepare_glo90_masks(data):
    """Returns the (land, non-ocean) fraction rasters on the 15" grid, from every
    geocell's water-body mask; fractions run 0 to 255.

    They store land rather than water so that the cells of geocells the bucket
    lacks, never written, read as 0: all ocean. The masks' half-sample offset
    (1.5") against the 15" cells is ignored.
    """
    directory = data / "sources" / "glo90"
    paths = directory / "land.tif", directory / "non_ocean.tif"
    if all(path.exists() for path in paths):
        return paths
    names = sorted(glo90_list(data))
    print(f"fetching {len(names)} GLO-90 water-body masks")
    fetch_glo90(data, names, dem=False)
    options = ["TILED=YES", f"BLOCKXSIZE={CELLS_PER_DEGREE}", f"BLOCKYSIZE={CELLS_PER_DEGREE}", "COMPRESS=ZSTD",
               "SPARSE_OK=TRUE", "BIGTIFF=YES"]
    temporaries = [path.with_name(path.name + ".tmp.tif") for path in paths]
    datasets = []
    for temporary in temporaries:
        dataset = gdal.GetDriverByName("GTiff").Create(str(temporary), GLOBAL_COLUMNS, GLOBAL_ROWS, 1,
                                                       gdal.GDT_Byte, options)
        dataset.SetGeoTransform((-180, 1 / CELLS_PER_DEGREE, 0, 90, 0, -1 / CELLS_PER_DEGREE))
        dataset.SetProjection("EPSG:4326")
        datasets.append(dataset)
    print("preparing land.tif and non_ocean.tif")
    for name in names:
        wbm = gdal.Open(str(glo90_paths(data, name)[1])).ReadAsArray()
        rows, columns = wbm.shape
        # Geocells narrow in longitude toward the poles; repeating columns to a
        # common multiple of 240 keeps the averages exact.
        repeat = np.lcm(columns, CELLS_PER_DEGREE) // columns
        lat, lon = glo90_corner(name)
        for dataset, fraction in zip(datasets, (wbm == 0, wbm != WBM_OCEAN)):
            cells = np.repeat(fraction.astype(np.float32), repeat, axis=1)
            cells = cells.reshape(CELLS_PER_DEGREE, rows // CELLS_PER_DEGREE, CELLS_PER_DEGREE, -1).mean(axis=(1, 3))
            dataset.GetRasterBand(1).WriteArray(np.rint(255 * cells).astype(np.uint8),
                                                (lon + 180) * CELLS_PER_DEGREE, (89 - lat) * CELLS_PER_DEGREE)
    for dataset in datasets:
        dataset.BuildOverviews("AVERAGE", [2**k for k in range(1, OVERVIEW_LEVELS + 1)])
    datasets = None
    for temporary, path in zip(temporaries, paths):
        temporary.rename(path)
    return paths
