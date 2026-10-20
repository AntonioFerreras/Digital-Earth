# Streamed Earth: outline

Replace the whole-planet equirectangular ground maps with a cube-sphere quadtree
of tiles baked offline into a 10 to 20 GB install and streamed on demand, so
terrain and albedo reach about 100 m. Evidence:
`docs/research/20261020-streamed-earth-tiles.md`.

## Decided

- Local install, offline bake; no network at runtime.
- Cube-sphere quadtree of six faces; 192² tiles to depth 9 (102 m at a face
  centre, 94 m mean), about 17 to 20 GB.
- Heights: Copernicus GLO-90 on land with its water-body mask, GEBCO_2025 at
  sea; 1 m steps, MED-predicted, zstd.
- Albedo: Blue Marble NG at its own 500 m to start; the tile format takes a
  sharper source later.
- Bake in Python with GDAL (`py` environment); runtime in Zig.
- No procedural detail below the data for now.
- Clouds and stars stay equirectangular (clouds keep their bound map).

## Shape

### Planet addressing (`shaders/earth/cube.slang`, `src/assets/Cube.zig`)

- Equal-angle cube (tangent warp of the gnomonic face coordinates): texels vary
  about 1.3x across a face instead of Proland's 2x, the same data in fewer
  texels.
- `TileKey = (face, level, x, y)`; one function each way between a direction and
  (face, u, v), shared by the bake and the shaders through goldens.

### Tile format (`src/assets/TileSet.zig`)

- One file per face and layer: header, u64 offset table in Z order (identical
  tiles share one payload), payloads.
- Layers:
  - **height**: vertex-centred n + 1 samples with a 2-sample border, signed
    metres (sea floor included), stored as residuals against the reconstructed
    parent (Proland's closed loop), MED-predicted, zstd. Also per tile: min and
    max height for the traversal bounds.
  - **surface**: albedo RGB + water mask in A, n texels with a 2-texel border,
    linear-space box mips; zstd for now (Blue Marble stops at depth 7), an
    image codec once a sharper source arrives.
- Coarse levels 0 to L_base of every layer are always resident (tens of MB).

### Bake (`python/tiles/`)

- Offline in the `py` environment: GDAL/rasterio reads GLO-90 COGs, the water
  mask, GEBCO, and the albedo source, reprojects to cube faces, writes tile
  files. Resumable per face. Regions first (one 1-degree block) to validate,
  then the planet.

### GPU streaming (`src/render/TileStreamer.zig`, `shaders/earth/tiles.slang`)

- Pool: `Texture2DArray`s of fixed slots (heights R32F or R16 + scale; surface
  RGBA8), LRU on the host, about 1 to 2 GB, split across arrays under the
  2048-layer guarantee.
- Indirection: per-face u32 mip pyramid of {slot, resident level} pointing at
  the finest resident ancestor; one dependent fetch per lookup.
- LOD: ray cones carried by `Path`; a lookup wanting a finer level than resident
  sets a bit in a request bitset.
- Feedback: triple-buffered request buffers copied to readback memory and read
  behind fences two or three frames late; worker threads read and decode
  (libzstd vendored in the native library), uploads go through a persistent
  staging ring.
- Epochs: new tiles enter the indirection only at an epoch boundary; a change
  of heights where primary rays hit restarts accumulation, albedo-only changes
  keep it.

### Terrain intersection (`shaders/earth/surface.slang`)

- Two levels: walk the face quadtree front to back with each node's max height
  (skipping like today's bound map), then inside a resident tile step its
  max-mip chain in tile-local f32 coordinates. A missing child uses the parent's
  bound and surface; hits carry their level so shadow and bounce rays start on
  the same surface.

### C shim additions (`native/`)

- Fences (signal on submit, poll), recorded buffer-to-texture copies into array
  layers, persistent mapped upload buffers, recorded copies to readback buffers
  mapped after their fence, BC formats for later.

## Order

1. Shim: fences, recorded copies, readback ring; tests.
2. Addressing and tile format; bake one region (the Alps) at full depth.
3. Pool, indirection, and surface (albedo, mask) lookups with ancestor
   fallback; coarse levels only, no feedback. Equirect maps stay for the rest.
4. Heights: traversal over resident tiles; goldens against the equirect path at
   matching resolution.
5. Feedback-driven streaming, epochs, viewer status (resident tiles, queue).
6. Planet bake; remove the equirect ground maps, Q1/Q2, and `bake-textures`.

## Risks

- Ray-cone LOD for volumetric scattering paths is coarse by design; fine tiles
  load only near primary hits, so close-up shadows may resolve late.
- Sphere-traced terrain normals today use exact bilinear on 1.85 km texels; at
  100 m the canopy noise of GLO-90 shows as roughness.
- A sharper albedo (EOX Sentinel-2) needs EOX's consent for bulk download and
  is CC BY-NC-SA for 2018-2025 imagery.
- GDAL in the `py` environment adds a large dependency to the bake only.
