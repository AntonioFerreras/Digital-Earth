# Porting spec: textured Earth path tracer (`main` @ 26a95b7)

Refs are `file:line` at `main`. Shared physics (densities, extinction, spectral
sampling, post) is in `20261018-pathtracer-mlp-spec.md`; this file lists what
`main` adds or does differently.

## Differences from `data_prep`

- Heightfield land (`pathtracer.py:12-46`), clouds (`:49-65`), material
  lookup `get_land_material` (`:285-313`), night-light emission (`:417`),
  stars on escaped primary rays (`:461-463`).
- Up to 25 bounces (`:349`); data_prep uses 10.
- 1 spp per `render` call (renderer.py:305-330).
- `turbidity = 1.06`, `clouds_extinct = 0.1`, Rayleigh
  `density_sea_level = 1.225` (volume_rendering_models.py).
- Mie phase is always Klein-Nishina e=3000 (`:65-71`); no HG switch after the
  first scatter.
- Full `earth_brdf` (surface_rendering_models.py:35).
- Initial camera (-15e6, 0, 15e6). `fov` is the vertical half-angle used
  linearly (default 13.5 deg), not tan.
- Dead deterministic `ray_marcher` (`:472-685`): not ported.

## Textures (`lib/textures.py`)

`TEXTURE_QUALITY = 2` default.

| Role | Q2 (default) | Dims | PNG | Q1 | Q0 |
| --- | --- | --- | --- | --- | --- |
| albedo | earth_color_21K | 21600x10800 | RGB | color_10K | color_4K |
| topography | topography_21K | 21600x10800 | gray | topo_10K | topo_4K |
| ocean mask | earth_landocean_16K | 16200x8100 | gray | landocean_8K | 4K |
| clouds | earth_clouds_21K | 21600x10800 | gray | clouds_8K | 4K |
| bathymetry | earth_bathymetry_21k | 21600x10800 | palette | 10k | 4k |
| emissive | earth_nightlights_21K | 21600x10800 | RGB (use R) | 10K | 4K |
| stars | stars_16K | 16200x8100 | RGB | stars_16K | stars_8K.jpg (missing) |

Q0 is broken (`CLOUDS_TEX_RES` undefined, stars file missing).

- Loading: `ti.tools.imread` returns `array[x, y]` with y flipped, so texel
  v=0 is the south pole. Single-channel maps take channel 0. Stored as UNORM
  `rgba8`/`r8` (no sRGB decode).
- Sampling (math_utils.py:26-44): `n = normalize(pos)`,
  `uv = ((atan2(n.z, -n.x)/pi + 1)/2, asin(n.y)/pi + 0.5)`. +Y north;
  u=0.5 (Greenwich) is -X; u=0.75 (90E) is +Z; seam at +X. LOD 0, linear,
  REPEAT in u and v (port: CLAMP in v).
- Semantics: ocean 255 = water; bathymetry 255 = land/shallow, 0 = deep;
  topography 0 = sea level, 255 = 7800 m; clouds = coverage; nightlights
  mostly 0.

## Land

- SDF `|p| - planet_r - 7800*topo(p)` (`:12-14`); bathymetry never displaces.
- Sphere trace (`:28-46`): start at atmosphere entry or 0; up to 250 steps;
  stop if `t > 10*planet_r` or `|d| < t*1e-4`; -1 on miss.
- Normal (`:17-25`): backward differences along world axes,
  `e = pi*planet_r/TOPO_RES.x` (one equatorial texel), 4 SDF evaluations.
- Surface offset `land_pos *= 1 + 1e-4*7800/12000` (~414 m). Shadow ray is a
  full `intersect_land` toward the sun.

## Material (`get_land_material`, sRGB space, Rec.709 luminance)

```
sat   = mix(lum3(a), a, 6.5)
green = smoothstep(1.5, 1.9, (sat.g/lum(sat))^2)
land  = a / (green*0.7 + 1)
land  = mix(lum3(land), land, 1.4 - green*0.45)
land  = mix(land, land*vec3(1, 128/255, 64/255), 0.2*(1 - green))
ocean = mix(lum3(a), a, 0.75)*0.9
albedo_srgb = mix(land, ocean, oceanMask)
```

Albedo uplifted with `srgb_to_spectrum`.

## BRDF (`earth_brdf`, surface_rendering_models.py:10-37)

- Diffuse: `albedo * disney_diffuse(roughness 0.73) * 0.28` (Burley).
- Land specular: GGX/Smith height-correlated, alpha = 0.73, F0 0.04,
  `D*G2*F / max(4*nl*nv, 1e-5)`.
- Ocean: roughness `mix(0.25, 0.19, smoothstep(0.3, 0.7, bathy))`, F0 0.02;
  `mix(0.65*Beckmann, GGX, clamp(smoothstep(0.2, 0.95, nv), 0.05, 0.94))`;
  Beckmann term (alpha = 2r^2, V-cavity G) lacks `1/(4*nl*nv)`.
- Fresnel: unpolarized dielectric, `eta = (1+sqrt F0)/(1-sqrt F0)`.
- Total: `diffuse + mix(landSpec, oceanSpec, smoothstep(0.6, 1, ocean))*0.5`.

## Clouds

- Shell 6375..6381 km; `clouds_extinct 0.1`, `clouds_density 0.029`.
- Density (`:49-65`): `h = (r - lower)/6000`, `c = clouds.r`; non-zero for
  `h in (0.2 - 0.2c, 0.2 + 0.8c)`, value `max(c, 0.4)*0.029`. No noise.
- Phase: HG+Draine mix, d=8, g = `exp(-0.099/(d-1.67))` (~0.984) first
  bounce, 0.91 after.
- Separate delta-tracking pass with its own majorant; one segment only
  (`intersect_cloud_limits :146-169`); nearer event wins (`:173-207`);
  transmittance = RMO x cloud ratio tracking.
- Bounce > 9: cloud extinction 0.02 and isotropic cloud phase (albedo 0.99).
- Clouds ignore terrain when entered from above.

## Emission

- Night lights at every surface hit: `throughput*emissive*planck(2700K)*1e-4`.
- Escaped primary ray only: sun disk `planck(5778)` inside the cone; stars
  `srgb_to_spectrum(stars(dir))*planck(5778)*1e-7`.

## Path loop (`:317-469`)

Per iteration: `intersect_land`; delta-tracked interaction; sample sun cone;
then absorb (break), volume scatter (NEE with sphere-only visibility + ratio
tracking, phase sampling), surface (emission, NEE with heightfield shadow,
cosine bounce `throughput *= brdf*pi`, no GGX sampling, no MIS), or escape.
Russian roulette when `scatter_count > 3`, `p = max(0.05, 1 - throughput)`.

## Memory and limits

- Q2 GPU textures ~2.5 GB (albedo rgba8 933 MB; r8 maps 233 MB each; stars
  525 MB). Upload via staging, free host copies.
- Five Q2 maps are 21600 wide; `maxImageDimension2D` is 16384 on AMD, Intel,
  Apple. Needs tiling or a fallback to Q1.
- Hardware bilinear has ~8-bit sub-texel weights; heightmap sphere tracing
  and FD normals benefit from manual f32 bilinear.
- f32 world positions at 6.4e6 m: ~0.5 m ulp.
