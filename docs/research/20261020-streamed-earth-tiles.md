# Streamed Earth tiles: sources, prior art, and the current code

Evidence for replacing the whole-planet equirectangular maps (1.85 km per texel)
with a streamed cube-sphere quadtree of tiles in a 10 to 20 GB local install.
Numbers marked *measured* come from probes on downloaded samples; the rest are
cited or reasoned as stated.

## Data sources

| Layer | Source | Resolution | Licence |
| --- | --- | --- | --- |
| Land height | Copernicus GLO-90 ([AWS](https://registry.opendata.aws/copernicus-dem/), [CDSE](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)) | 3", 1-degree float32 COGs, 26,475 tiles, no ocean tiles | free (ESA CCM licence), attribution |
| Water mask | GLO-90 `AUXFILES/*_WBM.tif` | 3" | as above |
| Ocean depth | [GEBCO_2025](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid) | 15" (463 m), int16, 7.5 GB netCDF | public domain, attribution |
| Albedo | [EOX Sentinel-2 cloudless](https://cloudless.eox.at/) WMTS, WGS84 z10 = 76 m, z9 = 153 m | 256-px JPEG tiles, ~23 KB each | 2018-2025: CC BY-NC-SA 4.0; 2016/2017: CC BY 4.0 |
| Albedo | [Blue Marble NG](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/) `world.2004MM` (not `topo`/`bathy`) | 500 m, 8 tiles of 21600² | public domain, credit NASA Earth Observatory |

- GLO-90 is a surface model: canopy adds noise. FABDEM (bare earth) is
  non-commercial; CIAT SRTM needs permission for commercial use.
- The ocean mask comes from the water-body mask at 3", not GEBCO's 463 m type
  grid. Heights may stay below 0 on land (Caspian, Dead Sea).
- EOX is BOA reflectance with BRDF correction but graded for display and with
  terrain shading baked in; Blue Marble is closer to albedo but 500 m.
- EOX publishes no rate limit for its WMTS; bulk GeoTIFF delivery is a paid
  order. Harvesting z10 (about 0.6 M land tiles, 14 GB) needs their consent.

## Compression (measured on six GLO-90 tiles and a 1024² EOX z10 mosaic)

Height, bits per texel on 256² tiles:

| Codec | 1 m step | 2 m step | Decode, one core |
| --- | ---: | ---: | --- |
| raw uint16 + zstd-19 | 9.5 (Alps) | | |
| MED predictor, zigzag, byte planes, zstd | 3.98 (1.58-5.58) | 3.14 | 2.3 GB/s (libzstd) |
| JPEG-LS (CharLS) | 3.83 | 3.04 | 96 Mpx/s |
| JPEG XL lossless, effort 7 | 3.43 | 2.61 | 22 Mpx/s |
| LERC + zstd | 6.0 | 4.8 | |

Albedo, bits per texel and PSNR against the source:

| Codec | bpp | PSNR | Decode, one core |
| --- | ---: | ---: | --- |
| AVIF q65 | 2.23 | 35.2 dB | 65 Mpx/s |
| JPEG XL d1 | 2.73 | 38.0 dB | |
| JPEG q85 | 3.58 | 37.5 dB | 320 Mpx/s |
| BC7 + zstd | 7.41 | 41.6 dB | |

- Zig's `std.compress.zstd` decodes at 104 MB/s against libzstd's 2.3 GB/s:
  vendor libzstd. BC7 on disk does not fit the budget; tiles stay RGBA8 in the
  GPU pool or are encoded on the GPU later.
- Install estimates (land only at the finest two levels, GEBCO to 349 m):

  | Tile | Texel (centre / mean) | Heights, MED+zstd 1 m | Heights, JXL 2 m | Ocean | Albedo, AVIF | Total |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | 224², depth 9 | 87 / 80 m | 15.8 GB | 10.5 GB | 3.3 GB | 9.5-12 GB | 23-31 GB |
  | 192², depth 9 | 102 / 94 m | 11.6 GB | 7.7 GB | 2.4 GB | 7-9 GB | 17-23 GB |

  Blue Marble albedo stored at its own 500 m (depth 7) is about 1 GB.

## Proland (BSD-3 C++, GPLv3 demo shaders and configs; design only)

- Six gnomonic cube faces, each a quadtree in one file per face and layer,
  Z-ordered tiles behind an offset table that deduplicates constant tiles
  (`HeightMipmap.cpp:615-669`; uint32 offsets cap files at 4 GB).
- Heights are residuals against the *reconstructed* parent upsampled with a
  (-1, 9, 9, -1)/16 filter, int16 at a 1 m quantum, deflate TIFF: 13.6 KB per
  unique 192² tile, 16 GB for the planet at 65 to 130 m (*measured* from the
  offset tables). Tiles are vertex-centred (193 + 2 x 2 border = 197).
- Colour: JPEG q90 tiles (deflate at face edges), 2-texel borders, linear-space
  box mips; alpha packs a water mask and night lights. 190 MB per face at 400 m.
- Runtime: LRU tile caches over `Texture2DArray` pools, producers with
  dependencies on worker threads, ancestor fallback while a tile loads.
- Not reusable for a path tracer: residency, LOD and geomorphing follow the
  camera; per-quad bounds arrive by readback frames late; no traversal
  structure; ocean and clouds are screen-space or shell shortcuts.

## Streaming for a path tracer

- LOD per ray from ray cones ([Akenine-Möller et al. 2019](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/2019-ray-tracing-gems-chapter-20-akenine-moller-et-al.pdf), [JCGT 2021](https://jcgt.org/published/0010/01/01/paper.pdf)):
  incoherent secondary rays have wide cones and ask for coarse tiles
  ([Christensen et al. 2003](https://graphics.pixar.com/library/RayDifferentials/paper.pdf)).
- Request feedback as in OptiX Toolkit demand loading
  ([OTK](https://github.com/NVIDIA/optix-toolkit/tree/master/DemandLoading)):
  lookups mark the tile they wanted; the host reads the marks back frames
  later behind fences and never waits.
- Indirection without sparse textures: a per-face mip pyramid of u32 entries
  holding the finest resident ancestor's slot and level (id Tech 5, Barrett
  SVT); about 5.6 MB per face at 2^10 leaf tiles per side.
- Height-field traversal with per-tile maximum mipmaps
  ([Tevs et al. 2008](https://pure.mpg.de/pubman/item/item_1325622_5/component/file_3590464/i3d08.pdf), [Dick et al. 2009](https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2009/GPU_Ray-Casting/EG09AreasTerrain.pdf));
  non-resident children fall back to the parent's bound; tile-local f32
  coordinates since f32 at the Earth's radius resolves 0.5 m.
- Progressive rendering: change residency only between accumulation epochs and
  restart when heights near primary hits change.

## The current code

- Equirectangular maps per role in `EarthMaps.zig` (raw cache), split for the
  2D limit by `TilePlan.zig`, uploaded synchronously by `GpuEarthMaps.zig`
  (Q2: 2.5 GB), bound per dispatch by `ModeKernel.zig` (9 of 32 bindings).
- Sampling: `land_sdf` (exact bilinear topography), `sample_surface` (albedo,
  ocean mask, bathymetry, night lights), `cloud_extinction_at`, `sky_radiance`
  (stars), `MapBound` (topography and cloud bounds); used by `path_radiance`,
  `raymarch`, and `datagen` (placeholders).
- The C shim lacks what streaming needs: fences, recorded buffer-to-texture
  copies, persistent upload rings, non-blocking readback, BC formats, and more
  bindings (slang-rhi d825f483 has fences, copies, and BC formats).

## Planet bake (python/tiles)

- Water at every level comes from the GLO-90 water-body masks (Product
  Handbook 5.0, table 8: 0 no water, 1 ocean, 2 lake, 3 river). Levels 8 and 9
  average the masks on GLO-90's own 3" lattice. Levels 0 to 7 read two 15"
  rasters baked once from all 26,475 masks (`sources/glo90/land.tif`,
  `non_ocean.tif`, 85 MB with average overviews), where a geocell the bucket
  lacks is ocean. Both paths average the same masks, so a lake appears at the
  same place at every level; GEBCO's sign (the first Alps bake) put no lake
  above sea level on levels 0 to 7.
- Box-averaging the level 8 water planes into their ancestors would match
  level 8 exactly but needs every level 8 tile before any coarse one, and
  leaves no water where no level 8 tile exists; the 15" rasters give the
  same averages up to the masks' 1.5" half-sample offset.
- The Caspian's special geocells are lakes, so their level 8 and 9 heights are
  the DEM's water surface (-28 m), while levels 0 to 7 keep GEBCO's lake bed.
- The DEMs (about 110 GB) are never all on disk: each group of tiles under a
  level 2 tile fetches its geocells (32 transfers at a time, about 40 MB/s
  measured) while the previous group bakes, and drops them after.
