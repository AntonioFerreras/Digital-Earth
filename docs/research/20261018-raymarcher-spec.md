# Porting spec: ray marcher (`origin/ray-marching-approimation` @ eae9d88)

Refs are `file:line` at eae9d88. Branched from 12edd87 (main = 12edd87 + README
commits). Shares camera, spectral sampling, accumulation, and post with the
path tracer.

## Shape

- Hard-coded replacement of the path tracer (`renderer.py:349`); no toggle.
  Port as a third render mode.
- One wavelength per pixel per frame, progressive accumulation.
- Two spectral LUTs baked offline by `generate_luts.py` into raw `.dat`, loaded
  at startup into r32f 3D textures. Atmosphere constants are frozen into them.
  The port bakes them on the GPU at startup (and on atmosphere change).

## Transmittance LUT (64, 64, 441) f32

- Bruneton (mu, r) parameterization (`lib/lut_sampling.py:17-49`):
  x = mu via distance to top of atmosphere, y = rho/H; z = wavelength
  390 + z (1 nm bins, 390..830).
- Value `exp(-ext*OD)`, 256-step left-endpoint march to top of atmosphere
  (`ray_functions.py:175-202`); planet-hitting rays store 0. Clouds excluded.
- Lookup (`ray_functions.py:205-216`) returns 0 if cos < -0.19 (magic horizon
  cutoff); r = h + planet_r.

## Multiple-scattering LUT (128, 128, 441) f32

- Layout (`generate_luts.py:122-131`, texel centers): x `s = 2u-1; mu = sign(s)*s^2`;
  y `h = atmos_height*v^2`; z `lambda = 390 + z + 0.5`.
  Inverse (`ray_functions.py:219-228`): `u = 0.5*sign(mu)*sqrt|mu| + 0.5`,
  `v = sqrt(h/atmos_height)`, `w = (lambda-390)/441`.
- Bake (`:113-204`): 15 iterations ping-ponging two textures; each texel
  averages `ray_march_scattering` (32 uniform steps) over 512 golden-spiral
  directions. Iteration 0: single scattering (planet `rsi` visibility,
  transmittance LUT, Rayleigh + KN Mie phase) plus ground bounce
  `ground_albedo*max(N.L,0)*T_sun*T_view` with `ground_albedo = 0.04/pi`.
  Iterations >= 1 add `sigma_s*phase*LUT(pos)`. Stored: in-scattered radiance
  per unit sun irradiance (Planck applied in the renderer).

### Bake bugs (fixed in the port)

1. Each pass reads the cumulative LUT: `(1+K)^n S0` instead of `sum K^n S0`.
   Fix: accumulate per-order, or Hillaire 2020 `1/(1 - f_ms)`.
2. Orders >= 1 use anisotropic phase at `dot(dir, sun)` (`:102`); should be
   isotropic. Bugs 1 and 2 partly cancel, so the committed LUT is
   "visually tuned".
3. Ground term added for directions that leave the atmosphere (`:164-171`).
4. Transmittance bake uses `x/RES`, not `(x+0.5)/RES` (half-texel shift).
5. `step_integral` `(1-exp(-OD))/OD` -> 0 for tiny OD and NaN at 0 in f32.

## March (`pathtracer.py:171-343`)

Up to 3 bounces:

1. Intersect: heightmap SDF sphere-trace (250 steps, `geometry.py:27`), then
   atmosphere `rsi`; `t_max` = land hit or atmosphere exit; miss -> break.
2. Sun direction sampled in the cone each bounce.
3. `ray_march_rmo`: 32 uniform steps (no jitter), left-endpoint density;
   energy-conserving step integral; single scattering
   `sigma_s*density*phase*sun_vis*T_LUT(dot(sun, up), h)`; multiple scattering
   `sigma_s*density*MS_LUT(pos, sun)` (isotropic). Ozone absorbs only.
   `accum += throughput*in_scatter*sun_irradiance; throughput *= T`.
4. Surface via `get_land_material` (albedo/ocean/bathymetry/emissive,
   `srgb_to_spectrum` uplift): night lights, direct sun with land self-shadow
   (second SDF march) and `T_LUT`, `earth_brdf`, cosine bounce
   `throughput *= brdf*pi`. No surface hit -> break.

Not present: clouds, sun disk, stars (commented out at `:329-337`), sky-view
or aerial-perspective LUTs. NaN/Inf/negative -> 0.

## Hazards

- `.dat` memory order is NumPy C-order (x, y, lambda): wavelength innermost.
  A `Texture3D` has width innermost. Moot once baked on GPU.
- Trilinear across lambda needs linear filtering of r32f 3D textures; check
  device support, else r16f or manual lerp.
- Bruneton lookup with r < planet_r is NaN; clamp r.
- Replace the -0.19 cutoff with an analytic horizon test.
- Two 250-step SDF marches per surface hit dominate cost.
- Unrelated look change on this branch: OpenDRT contrast 1.0 -> 1.4.
