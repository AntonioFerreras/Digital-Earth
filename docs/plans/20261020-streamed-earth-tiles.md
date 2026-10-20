# Streamed Earth tiles: format and first packets

Execution plan for steps 2 and 3 of `docs/outlines/20261020-streamed-earth.md`.
The bake (Python) and the runtime (Zig, Slang) meet only at this file format and
the cube mapping, so both are fixed here before either is written.

## Cube mapping

- World frame as today: +Y north, Greenwich on -X, 90 E on +Z (`sphere_uv`).
- Faces, each with the axes its (s, t) coordinates grow along:

  | face | normal | s axis | t axis |
  | --- | --- | --- | --- |
  | 0 | +X | -Z | +Y |
  | 1 | -X | +Z | +Y |
  | 2 | +Y | +X | -Z |
  | 3 | -Y | +X | +Z |
  | 4 | +Z | +X | +Y |
  | 5 | -Z | -X | +Y |

- Direction n to face: the axis of largest |n|. Gnomonic coordinates
  a = dot(n, s_axis) / |n_major|, b = dot(n, t_axis) / |n_major| in [-1, 1].
- Equal-angle warp: (s, t) = (4 / pi) (atan a, atan b) in [-1, 1]; inverse
  a = tan(pi s / 4). Face coordinates (u, v) = ((s + 1) / 2, (t + 1) / 2) in
  [0, 1], v = 0 at the t axis's negative end.
- Tile (face, level, x, y) covers u in [x, x + 1] / 2^level, v likewise; x grows
  with u, y with v.
- The Python bake's implementation is the reference; goldens check the Slang one
  against it (`python/goldens.py`).

## Tiles

- **Height**, levels 0 to 9: vertex-centred, N = 192 intervals, so 193 samples
  span the tile edge to edge and neighbours share their edge samples; 2 border
  samples each side: 197 x 197. Sample (i, j) of the bordered grid sits at
  u = (x + (i - 2) / 192) / 2^level, and likewise v from j. Two planes:
  - height: int16 metres, rounded, sea floor negative (GEBCO), land from
    Copernicus GLO-90 (EGM2008), below 0 where the land is (Caspian).
  - water: uint8 fraction of water 0 to 255 (255 is ocean).
- **Surface**, levels 0 to 7: pixel-centred 192 x 192 texels plus a 2-texel
  border: 196 x 196 RGB, encoded sRGB albedo as the current maps hold it
  (Blue Marble NG `world.2004MM`, the month a bake option).
- Levels past a layer's last level do not exist; a level whose tile is missing
  (all ocean past GEBCO's resolution) means "use the parent".

## Files

`<data>/tiles/<layer>/face<f>.tiles`, one per layer (`height`, `surface`) and face,
little-endian:

| bytes | field |
| --- | --- |
| 4 | magic `ETIL` |
| 4 | version 1 |
| 4 | layer: 0 height, 1 surface |
| 4 | face 0 to 5 |
| 4 | tile intervals N (192) |
| 4 | border (2) |
| 4 | last level |
| 4 | codec: 0 height planes + zstd, 1 JPEG |
| 16 x count | entries, count = (4^(last + 1) - 1) / 3, level by level, each level in Morton order of (x, y) |
| ... | payloads |

- Entry: u64 payload offset, u32 payload size (0 = tile absent), int16 min and
  int16 max height (height layer; 0 for surface). Identical payloads may share
  an offset.
- Height payload: one zstd frame of three planes, row-major over the bordered
  grid: the low bytes, then the high bytes, of zigzag(height - MED(left, up,
  up-left)) with the predictor's missing neighbours taken as 0; then the water
  plane raw (zstd handles it).
- Surface payload: a baseline JPEG, quality 90, 4:4:4.

## Packets

1. **Bake** (`python/tiles/`, `py` environment, adds GDAL, zstandard):
   `cube.py` (mapping), `formats.py` (writer and reader), `bake.py` (CLI:
   `--layer`, `--faces`, `--levels`, `--region lon0,lat0,lon1,lat1` limiting
   the levels above 4 to tiles that touch the region, resumable). Sources:
   GEBCO_2025 netCDF (local), GLO-90 and its water-body masks read over
   `/vsicurl/` from the AWS bucket, Blue Marble NG tiles (local). Coarse levels
   from area-averaged sources (GDAL overviews), not point samples. Goldens for
   the mapping. Verify: a reader round trip, the Alps region baked to level 9,
   tile images that line up across a face edge.
2. **Runtime reading** (`src/assets/TileSet.zig`): parse, look up entries,
   decode payloads (libzstd vendored in the native library, stb for JPEG).
   Verify: decode matches the Python reader on the Alps bake.
3. **GPU pool and lookups** (`src/render/TilePool.zig`,
   `shaders/earth/tiles.slang`): resident levels 0 to 3 of both layers, the
   per-face indirection pyramid, `surface_at(direction)` and
   `height_at(direction)` with ancestor fallback; `sample_surface` switches to
   tiles behind a scene flag. Verify: goldens of the lookups, and a render of
   the Alps against the equirect maps at matching resolution.
