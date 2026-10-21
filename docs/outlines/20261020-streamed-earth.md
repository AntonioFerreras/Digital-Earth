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
  about 1.3x across a face instead of a plain gnomonic cube's 2x, the same data
  in fewer texels.
- `TileKey = (face, level, x, y)`; one function each way between a direction and
  (face, u, v), shared by the bake and the shaders through goldens.

### Tile format (`src/assets/TileSet.zig`)

- One file per face and layer: header, u64 offset table in Z order (identical
  tiles share one payload), payloads.
- Layers:
  - **height**: vertex-centred n + 1 samples with a 2-sample border, signed
    metres (sea floor included) and a water-fraction plane, MED-predicted,
    zstd; each tile decodes alone. Per tile: min and max height for the
    traversal bounds.
  - **surface**: RGB albedo, n texels with a 2-texel border, baseline JPEG;
    Blue Marble stops at depth 7.
- The binding layout is in `docs/plans/20261020-streamed-earth-tiles.md`.
- Coarse levels 0 to L_base of every layer are always resident (tens of MB).

### Bake (`python/tiles/`)

- Offline in the `py` environment: GDAL/rasterio reads GLO-90 COGs, the water
  mask, GEBCO, and the albedo source, reprojects to cube faces, writes tile
  files. Resumable per face. Regions first (one 1-degree block) to validate,
  then the planet.

### GPU streaming (`src/render/TileStreamer.zig`, `shaders/earth/tiles.slang`)

- Pool: `Texture2DArray`s of 2048 slots, the layer count every device supports
  (heights R16_SINT and water R8, surface RGBA8), about 0.6 GB.
- Indirection: per-face u32 mip pyramid of {slot, resident level} pointing at
  the finest resident ancestor; one dependent fetch per lookup.
- LOD: ray cones; a lookup wanting a finer level than resident sets a bit in a
  request bitset. Only camera rays request: shadow and bounce rays reach far
  and wide and read what the camera's tiles brought.
- Feedback: triple-buffered request buffers copied to readback memory and read
  behind fences three frames late; the render thread decodes up to 32 tiles a
  frame (libzstd's decoder built by build.zig, stb for JPEG) into a staging
  ring whose copies precede the frame's dispatches. A full pool evicts the
  least recently wanted leaf.
- Accumulation restarts when tiles arrive; offline renders stream until quiet
  first.

### Terrain intersection (`shaders/earth/surface.slang`)

- `shaders/earth/tile_terrain.slang`: each tile is bounded by four planes
  through the planet centre, so the march jumps exactly to where the ray leaves
  a tile whose subtree maximum it clears, coarsening the tried level after each
  jump and refining after each miss as a maximum-mipmap traversal does; at the
  resident level it steps the height field (sea level over the sea floor).

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
