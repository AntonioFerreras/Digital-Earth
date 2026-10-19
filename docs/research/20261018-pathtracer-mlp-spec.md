# Porting spec: symmetric path tracer + MLP mode (`data_prep` @ 9c7d1df)

Source of truth for the Taichi renderer on `data_prep`. Refs are `file:line` at
9c7d1df. The textured surface, clouds, night lights, and stars live on `main`;
see `20261018-textured-earth-spec.md`.

## State of the branch

- `path_tracer` canonicalizes every camera ray through a Bruneton uvwz
  round-trip, so output depends only on (r, mu, mu_s, nu). This is what makes it
  learnable by the 4->3 MLP. Ground is a smooth sphere, constant sRGB albedo
  0.3; clouds disabled (`clouds_extinct = 0.1*0`).
- Uncommitted on the main checkout: `data_prep.py`, `renderer.py`, `train.py`,
  `mlp_best.pth`; untracked `model_weights/`, `mlp_best_good.pth`, training data
  dirs.
- Does not run here: venv lacks torch, and Taichi's Vulkan init asserts in VMA
  on a Vulkan 1.4 driver. Taichi `arch=ti.cpu` works for reference values.

## Entry points

| File | Role |
| --- | --- |
| `main.py` | `EarthViewer().start()` |
| `earth_viewer.py` | window, camera, GUI, frame loop |
| `renderer.py` | fields, LUT loading, kernels, MLP glue |
| `pathtracer.py` | spectral volumetric path tracer |
| `data_prep.py` | offline MLP training data via `Renderer.batch_path_trace` |
| `train.py`, `model.py` | PyTorch training |
| `LUT/*_generator.py` | offline LUT generators |

Dead code: `land_sdf`, `land_normal`, `intersect_land`, `get_clouds_density`,
`speckle`, `get_land_material`, `ray_march_transmittance` (pathtracer.py);
Disney/GGX/Beckmann in `surface_rendering_models.py:28-36` (only
`albedo/pi*0.5` survives); `lib/AgX.py`; OpenDRT HLG/PQ branches.

## Runtime assets

| Path | Format | GPU form |
| --- | --- | --- |
| `LUT/CIE.dat` | raw f32, 441x2x3, index `(x + y*441)*3` | rgba16f 441x2 texture, bilinear. Row 0 (v=0.25): per-channel normalized CDFs of xbar,ybar,zbar. Row 1 (v=0.75): CIE 1931 2deg CMFs, 390..830 nm at 1 nm |
| `LUT/srgb2spec.dat` | raw f16, 300x3 | Smits-style basis, rows 400..699 nm, (r,g,b) weights; manual lerp |
| `LUT/ozone_cross_section.dat` | raw f32, 441 | cm^2/molecule (x1e-4 -> m^2), nearest `int(lambda-390)` |
| `LUT/camera_response_functions/*.rf` | text, 1024 lines x 4 floats, col 0 dropped | rgba32f 1024xN, bilinear. `Neutral.rf` forced to index 0, rest in `os.listdir` order. Duplicate `agfacolor-futura-100CD (1).rf` |
| `model_weights/mlp_e*.pth` | torch state_dict | highest epoch loaded; fallback `mlp_best.pth` |

Taichi samplers: linear filtering, REPEAT addressing (believed), no sRGB decode.

## GPU state (renderer.py:33-111)

`color_buffer` vec3 1920x1080 accumulator (sum of per-frame means);
`_rendered_image` display; `uvwz_buffer`/`rgb_buffer` MLP I/O; 0-D params
`fov, aspect_scale, exposure, gamma, sun_angle, sun_path_rot, selected_crf,
crf_count, camera_pos, look_at, up`; static LUT textures.

## Frame flow

- Path-trace `render` (:212): per pixel 32 paths (`spp=32.0`); each path:
  `spectrum_sample` -> jittered `get_cast_dir` -> `path_tracer` ->
  `xyz = L*cmf*rcp_pdf` -> `color += XYZ_to_sRGB_D65(xyz)/32`.
- `get_cast_dir(u,v)` (:198): `fu = (2*fov*(u+xi)/H - fov*aspect - 1e-5)*aspect_scale`,
  `fv = 2*fov*(v+xi)/H - fov - 1e-5`; `du = normalize(d x up)`,
  `dv = normalize(du x d)`; `dir = normalize(d + fu*du + fv*dv)`. `fov` is
  tan(half-angle). Pixel origin bottom-left.
- Sun: angular radius 6.95e8/1.4959e11 = 4.646e-3 rad;
  `light_direction = (-sin a, cos a*(-sin rho), cos a*cos rho)`, a = sun_angle,
  rho = sun_path_rot.
- MLP mode `render_with_mlp` (:445): advance camera to top of atmosphere if
  outside; miss -> `uvwz=(0,0,0,-1)`; else compute uvwz. Host does
  GPU->CPU->torch->CPU->GPU each frame; adds 1 sample/frame to `color_buffer`.
- `batch_path_trace(uvwz)` (:322): data generation; 512*32 = 16384 paths with
  float `atomic_add`, one launch per training sample.
- Viewer order (earth_viewer.py:225-364): camera update, setters,
  `accumulate()` (1 frame), `fetch_image`, screenshot, display, GUI, write
  params, reset accumulation if camera/sun/fov/aspect/mode changed (exposure,
  CRF, gamma do not reset).
- RNG: `ti.random()` (stateful, non-reproducible).

## Physics (metres, nanometres, m^-1)

Geometry (volume_rendering_models.py:27-37): planet_r 6371e3, atmos_height
110e3 (r_top 6481e3), cloud shell 6375e3..6381e3.

Densities (h clamped >= 0):

- Rayleigh: `3.68082*exp(-(h+24239.99)^2/532307548.4168)/1.220554684`.
- Mie x turbidity 1.03 (:251): h>11500: `0.0918*exp(-1e-6*(h-11500)^2)`;
  2400<h<=11500: `0.3*exp(-2.5e-9*(h+2500)^2) - 0.092`; 1300<h<=2400:
  `0.65*exp(-5e-6*(h-1300)^2) + 0.18899`; h<=1300: `1 - h/8136.646`.
- Ozone (km): `0.625*exp(-d^2/49) + 0.375*exp(-d^2/256) + max(0, -1.5e-5*(h_km-15)^3)`,
  `d = h_km - 25`.

Extinction:

- Rayleigh `8*pi^3*(n^2-1)^2/(3*N*lambda_m^4)*King`, N = 2.5035422e25,
  n from `air()` dispersion (lambda in um); King factor mixes N2/O2/Ar/CO2
  (CO2 0.0421) with depolarization terms evaluated with lambda in nm.
- Mie (Preetham): `0.434*c*pi*(2pi/lambda_m)^2*K`, `c = (0.6544*T - 0.6510)*4e-18`,
  `K = (0.773335 - 0.00386891*lambda)/(1 - 0.00546759*lambda)`.
- Ozone: `1e-4 * N*0.012588*8e-6 * sigma_LUT[int(lambda-390)]`, 0 outside [390,831).
- Single-scattering albedo: Rayleigh 1.0, aerosol 0.95, ozone 0, cloud 0.99.

Phase: Rayleigh `3/(16pi)(1+cos^2)`, sampled uniformly (weight phase*4pi). Mie:
first scatter Klein-Nishina e=3000 (importance sampled), later HG g=0.8. Cloud:
HG+Draine d=8.

Spectral sampling (colour.py:115): one wavelength per path; 8 bisection steps
on avg(CDF) from the f16 texture; lambda = 390 + 441*mid; response = CMF(mid);
pdf = response*CDF(u=1.0) (unnormalized; REPEAT at u=1 blends to 0.5);
rcp_pdf = 0 if pdf <= 1e-3; XYZ -> linear sRGB D65.

Sun: Planck 5778 K with h = 6.62607015e-16, c = 2.9e17, k = 1.38e-5. NEE
irradiance `radiance*2pi*(1 - cos theta_s)`, uniform cone sampling; sun disk
only on primary escape.

Path loop (pathtracer.py:345-452): up to 10 bounces; delta tracking with
majorant `sum(sigma_i*maxdens_i)`; event by CDF; analog absorption by albedo.
Volume scatter: NEE with planet occlusion + ratio-tracking transmittance
(cutoff 1e-5), then phase sampling. Surface: albedo
`srgb_to_spectrum(0.3)`, BRDF `albedo/pi*0.5`, shadow origin lifted ~414 m,
cosine bounce, `throughput *= albedo*0.5`. Russian roulette after bounce 3,
`p = max(0.05, 1 - throughput)`. NaN/Inf/negative -> 0.

## Post (`_render_to_image`)

1. `lin = color/spp * vignette * 2^exposure`; vignette
   `1 - 0.9*max(dist(uv,(0.5,0.5)), 0)`; exposure default 2.5.
2. OpenDRT v0.2.2: Rec709 in/out, linear EOTF, Lp 100, gb 0.12, contrast 1
   (1.4 on the ray-marching branch), flare 0.005, rw 0.25, bw 0.35, dch 0.35,
   dch_toe 0, hue shifts r 0.3 g -0.1 b -0.2, v_p 0.5, clamp 1.
3. CRF: per-channel lookup `u = min(c + 0.5/1024, 1 - 0.5/1024)`,
   `v = (idx + 0.5)/N`.
4. `pow(gamma)` (default 1), sRGB OETF.

## MLP (`model.py`)

`Linear(4,64) -> Linear(64,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64,3) -> ReLU`.
No activation after the first Linear. GELU is exact (erf). 12,931 params;
state-dict keys `linear_layers.{0,1,3,5}.*`, `output_layer.*`, `_orig_mod.`
prefix from torch.compile; weights `[out, in]` f32.

Input uvwz (bruneton_mappings.py:206-260): u_nu = (nu+1)/2; u_mus Bruneton
`max(1 - a/A, 0)/(1 + a)` with mu_s_min = -0.2; u_mu ground/sky split at 0.5;
u_r = rho/H; all via `GetTextureCoordFromUnitRange` (size 4096; 2048 for mu).
Output: non-negative linear Rec709, same scale as one path-tracer estimate.

Training: MSE on raw HDR, AdamW, lr 3e-4*512/2048, cosine to 0.1x, 120
epochs, batch 512, clip 1.0. Data: text, 65536 lines/file, `u v w z R G B`;
2^25 uniform uvwz, `--sample-sunset` scales y by 0.4; each target a
16384-path mean.

## Viewer

1920x1080, vsync off. Camera (-15e6, 0, 0) looking at origin, up +Y, fov 0.2356.
RMB drag rotate (3 rad per normalized unit); WASD, Space/Ctrl up/down, Shift
x3; speed `0.05*30*clamp(alt, 0, R/2)*dt`, reverted if below R. Q up=radial,
E up=+Y, R teleport to 100 m, I/O save/load `config.txt`, P JPEG screenshot,
G toggle GUI. GUI: MLP toggle, sun angle 0..360 (60), sun path rot -105..105
(-45), fov 1..90, aspect 0.75..1.25, exposure -1..10, CRF, gamma 0.45..2.2.

## Known bugs (fixed in the port, see outline)

1. `rsi` takes `sqrt` of negative discriminant; callers rely on NaN compare.
2. Sun disk dotted against a canonicalized light direction in world space
   (pathtracer.py:340-341, 456).
3. uvwz round-trip clamps mu_s >= -0.2 (night side lit as if sun at -11.5 deg).
4. `srgb_to_spectrum` lerp extrapolates backwards; albedo zero outside 401..699.
5. Spectral pdf unnormalized, CDF sampled at REPEAT u=1.0 (x2), f16 CDF.
6. OpenDRT `vdot` is `M^T v`; cancels only because matrices are inverse pairs.
7. CRF index depends on `os.listdir` order.
8. All f32; camera f64 on host uploaded as f32 (~1 m ulp at 15,000 km).
9. Stale `new_sun_path_rot` when GUI hidden on config load.
