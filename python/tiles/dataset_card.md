---
license: other
license_name: copernicus-dem-gebco-nasa
pretty_name: Digital-Earth streamed terrain tiles
---

# Digital-Earth streamed terrain tiles

Cube-sphere quadtree tiles of the Earth's height, water, and albedo for
[Digital-Earth](https://github.com/AntonioFerreras/Digital-Earth), a spectral
path tracer. Fetch them with `earth download`; the format is described in the
repository's `src/assets/TileSet.zig` and `python/tiles/formats.py`, and
`manifest.zon` lists every file's size and SHA-256.

- `height/face<f>.tiles`: heights and water fraction, levels 0 to 9 (about 90 m).
- `surface/face<f>.tiles`: albedo, levels 0 to 7 (about 500 m).
- `wind.raw`: the 10 m wind of 2023-07-15 12:00 UTC on a 0.25 degree grid, which
  roughens the sea.

## Sources and attribution

- Land heights and water bodies: produced using Copernicus WorldDEM-90
  © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
  under COPERNICUS by the European Union and ESA; all rights reserved.
- Sea floor and coarse heights: GEBCO Compilation Group (2025) GEBCO 2025 Grid.
- Albedo: NASA Earth Observatory, Blue Marble: Next Generation (July 2004).
- Wind: contains modified Copernicus Climate Change Service information (2023),
  ERA5 hourly data on single levels, from the ARCO-ERA5 copy.

The tiles are derived works redistributed under the terms of those sources:
the Copernicus DEM licence, GEBCO's public-domain terms, NASA's media usage
guidelines, and the Copernicus (C3S) licence.
