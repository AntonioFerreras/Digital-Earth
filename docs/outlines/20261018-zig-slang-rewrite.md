# Outline: Digital Earth in Zig + Slang

Rewrite the Taichi renderer as a Zig host application driving Slang compute
kernels through slang-rhi. Evidence: `docs/research/20261018-*.md`.

## Goals

- Three render modes over one shared physics library, selectable at runtime:
  - **Path tracer**: spectral volumetric path tracing (`main`'s textured Earth).
  - **Ray marcher**: single scattering plus a multiple-scattering LUT (from
    `ray-marching-approimation`).
  - **MLP**: per-pixel inference of the trained 4->3 network (from `data_prep`).
- Portable GPU: Vulkan on Linux/Windows (NVIDIA, AMD, Intel), Metal on macOS.
- An interactive viewer with an ImGui parameter panel, plus headless commands
  for offline renders, MLP training-data generation, and texture baking.
- Python is kept only for offline work: training, weight export, LUT
  generators, and reference values for tests.

## Non-goals

- No Taichi at runtime. The Taichi code is deleted once each mode passes its
  parity check.
- Wayland-native windows: slang-rhi has no Wayland handle, so SDL runs on X11
  (XWayland).
- Same-pixel parity with the Python output. Known bugs are fixed and listed
  under "Behavior changes"; parity is judged with those deltas isolated.

## Architecture

```
SDL3 window/input ─┐
                   ▼
        Zig host (src/)  ──►  libearth_native (C ABI only)  ──►  slang-rhi ─► Vulkan/Metal/D3D12
        viewer, assets,        native/ : C++ shim, ImGui,         Slang compiler (runtime)
        render orchestration   present pipeline
                   │
                   └── loads shaders/*.slang by module name at runtime
```

- **The native library is the only C++.** One CMake project builds slang-rhi
  (pinned SHA, static, PIC), Dear ImGui, the dear_bindings C API, the SDL3
  platform backend, and our shim into `libearth_native`. It exports only
  `extern "C"` functions, and everything else is hidden. Zig never compiles or
  links C++, so the process has only one C++ runtime and the Metal
  Objective-C++ code stays with AppleClang.
- **`zig build` is the single entry point.** It runs the CMake configure and
  build as build steps, so Ninja makes them incremental. It translates
  `earth_native.h`, `dcimgui.h`, and SDL3 headers through the `translate_c`
  package. It compiles the vendored stb C sources and links everything with an
  rpath to the native install directory.
- **pixi owns the toolchain:** zig 0.17.0, cmake, ninja, cxx-compiler, sdl3,
  Vulkan loader/headers/validation layers, and xorg-libx11. A `py` feature
  holds Python, numpy, PyTorch, and Taichi for the offline tools. pixi tasks
  are thin aliases over `zig build` and Python scripts, and the lock file is
  committed.
- **Shaders compile at runtime.** slang-rhi loads modules by name from
  `shaders/` and keeps a persistent cache. The install step copies `shaders/`
  next to the executable. A `zig build shaders` step compile-checks every
  entry point with `slangc` so CI catches errors without a GPU.

## Repository layout

```
build.zig, build.zig.zon       zig 0.17.0; deps: translate_c, stb
pixi.toml, pixi.lock
native/
  CMakeLists.txt               FetchContent: slang-rhi@d825f483, imgui v1.93.0, dear_bindings v0.26
  include/earth_native.h       the C contract (below)
  src/internal.h               shim-private structs shared by gpu.cpp and ui.cpp
  src/gpu.cpp                  device, resources, kernels, binding, commands
  src/ui.cpp                   ImGui context, SDL3 backend, surface, present pipeline
  shaders/present.slang        fullscreen blit + ImGui vertex/fragment
src/
  main.zig                     CLI: view (default) | render | datagen | bake-textures
  heap.zig                     process allocator: Debug -> DebugAllocator, else smp_allocator
  native.zig                   translate_c module of earth_native.h + dcimgui.h
  gpu.zig  gpu/                Device, Buffer, Texture, Sampler, Kernel, CommandList, Surface
  platform.zig platform/       Window (SDL3), native handle extraction
  assets.zig assets/           Luts, EarthMaps, TilePlan, MlpWeights, Image (stb)
  render.zig render/           Renderer, uniforms, PathTracer, RayMarcher, MlpMode, Post, Datagen
  viewer.zig viewer/           Viewer loop, Camera, Controls, Config (ZON), Panel (ImGui)
  test_runner.zig
shaders/
  shared/uniforms.slang        structs mirrored by src/render/uniforms.zig
  common/                      math (ray-sphere), rng (PCG hash), sampling, constants
  spectral/                    cie sampling, srgb_to_spectrum, planck, xyz->rgb
  atmosphere/                  densities, extinction, phase, bruneton, transmittance
  earth/                       sphere uv, tiled maps, heightfield, material, brdf, clouds
  modes/                       pathtrace.slang, raymarch_lut.slang, raymarch.slang, mlp.slang
  post/                        opendrt.slang, crf.slang, display.slang
  tests/                       test kernels evaluating library functions on golden inputs
python/                        train.py, model.py, export_weights.py, goldens.py, lut/ generators
tests/goldens/                 reference values from the Taichi code (CPU arch)
presets/                       *.zon camera/scene presets converted from "config - *.txt"
docs/
```

Source conventions: barrel files beside same-named directories; files with
top-level fields are TitleCase types; explicit allocators; specific error sets;
`//!` per file and `///` on public declarations; inline tests plus
`test_<Thing>.zig` for GPU tests. The custom test runner supports repeatable
`--test-filter`, checks for leaks per test, and treats `error.SkipZigTest`
(no GPU) as a skip.

## The C contract (`native/include/earth_native.h`)

Opaque handles, POD descriptors, `en_status` return codes, and a thread-local
`en_last_error()` for detail such as compiler diagnostics. No exception
crosses the boundary. The header is written first and is frozen while
the native and host work run in parallel. The groups are:

- **Device:** create (backend, validation, shader search paths, cache path),
  destroy, info (adapter, texture dimension limits), wait idle.
- **Buffer:** create (size, element size, usage bits, memory kind, initial
  data), upload, blocking read.
- **Texture:** 2D, 2D array, and 3D. Formats are r8/rgba8 unorm, r16f/rgba16f,
  and r32f/rgba32f. Calls are region upload and blocking layer read.
- **Sampler:** filter, and address mode per axis.
- **Kernel:** compile `module` + `entry`, thread-group size, parameter lookup
  by path (returning an opaque offset, the kind, and the size), and field
  lookup (offset and size of `param.field`) for layout validation.
- **Commands:** begin, dispatch with an array of bindings (data, buffer,
  texture, or sampler per parameter handle), submit. A UAV barrier separates
  consecutive dispatches.
- **Surface and present:** create from a tagged native window handle (Xlib,
  Win32, Cocoa), configure, and present. Present draws an rgba8 image
  stretched to the swapchain, then the ImGui draw data, and returns
  `EN_ERR_SURFACE_OUT_OF_DATE` when the swapchain must be reconfigured.
- **UI:** init with the SDL window, process SDL events, new frame. Widget
  calls go straight to the exported `dcimgui` API.

**Layout rule for uniforms.** Shared uniform structs use only 4-byte scalars
and `float4`/`uint4`, so std140, scalar, and C layouts coincide on every
backend. `gpu.Kernel` checks each Zig `extern struct` field's offset and size
against Slang reflection when the kernel is created.

## Render design

- **Frame:** `Renderer.frame()` writes `FrameUniforms`, which hold the camera
  basis, image-plane scale, sun direction and cone, the frame index, and the
  mode and scene flags. It then dispatches the active mode, which adds one
  sample per pixel per frame into an rgba32f accumulation texture, and finally
  dispatches `display`. `display` runs post-processing into an rgba8 image
  that `en_present` blits.
- **Resetting accumulation:** the camera, sun, FOV, aspect, mode, scene preset,
  or texture set reset it. Exposure, CRF, and gamma do not, because they only
  re-run `display`.
- **Spectral sampling:** one wavelength per sample, drawn by inverting the CIE
  CDF from an f32 buffer, with the pdf normalized. Accumulation is in linear
  Rec.709 (D65).
- **RNG:** a PCG hash keyed by (pixel, frame, sample dimension), so the output
  is reproducible.
- **Scene presets:** a struct of flags and constants, so one path-tracer
  kernel serves both scenes. They differ in scene content only; the
  atmosphere and the light transport are shared (KN Mie on the first scatter
  and HG g = 0.8 after it, turbidity 1.03, Rayleigh fit normalized to 1 at sea
  level, 25 bounces):
  - `textured` (from `main`): heightfield land, materials, clouds, night
    lights, stars.
  - `symmetric` (from `data_prep`): sphere ground with albedo 0.3, no clouds,
    and canonicalization through uvwz. It exists for MLP parity and data
    generation.
- **Earth maps:**
  - `bake-textures` decodes the NASA PNGs once with stb_image into a raw cache
    (header plus u8 rows, south row first, matching the Taichi orientation).
  - Upload splits each map into longitudinal tiles of a `Texture2DArray` with
    a one-texel apron, sized to the device's `max_texture_dimension_2d`. A map
    of 16384 or less is a single tile. Sampling picks the tile from u.
  - The heightmap uses manual f32 bilinear filtering for sphere tracing and
    normals.
  - Texture quality presets Q1 and Q2 are selectable. Q0 is dropped because
    it is broken upstream.
- **Ray-marcher LUTs:** baked on the GPU at startup and whenever an atmosphere
  parameter changes. Both are spectral with 441 bins:
  - Transmittance is 64x64x441 with texel-centred parameterization.
  - Multiple scattering is 32x32x441 using Hillaire 2020's closed form
    `L_2nd / (1 - f_ms)`, with 64 directions and 20 steps. This replaces the
    15-iteration offline bake, which over-counted scattering orders.
- **MLP:** `export_weights.py` converts the state dict into a flat f32 blob
  with a small header (layer sizes). `mlp.slang` runs the forward pass per
  pixel from a `StructuredBuffer<float>`, using exact GELU through an erf
  approximation with error below 2e-7. There is no host round trip.
- **Data generation:** `earth datagen` runs the symmetric path tracer over
  uvwz samples in batches and writes binary float32 records (`u v w z R G B`).
  `train.py` reads that format.
- **Viewer:**
  - Controls and the sliders from `main`, plus a mode selector, scene preset,
    texture quality, and spp/s readout.
  - Configs are saved and loaded as ZON, and the shipped
    `config - *.txt` files are converted to `presets/*.zon`.
  - Screenshots are written as PNG with stb_image_write.

## Behavior changes (bugs fixed during the port)

| # | Python behavior | Port |
| --- | --- | --- |
| 1 | `rsi` takes sqrt of a negative and relies on NaN comparisons | explicit miss result; same hit/miss decisions |
| 2 | spectral pdf unnormalized; CDF read through f16 and REPEAT at u=1 (x2) | f32 CDF buffer, normalized pdf; default exposure retuned to match brightness |
| 3 | `srgb_to_spectrum` lerp runs backwards; zero outside 401..699 nm | forward lerp; edge rows clamped |
| 4 | sun disk tested in canonical frame against world direction (data_prep) | world frame |
| 5 | OpenDRT `vdot` computes M^T v | M v (superseded by 16) |
| 6 | CRF order from `os.listdir` | Neutral first, then sorted by filename; duplicate file dropped |
| 7 | MS LUT over-counts orders, anisotropic phase on orders >= 1, ground term on escaping rays | Hillaire closed form |
| 8 | transmittance bake at `x/RES`; -0.19 horizon cutoff | texel centres; analytic horizon test |
| 9 | `(1-exp(-OD))/OD` -> 0 or NaN for tiny OD | series form below 1e-4 |
| 10 | Earth maps REPEAT in v (pole bleed) | CLAMP in v |
| 11 | clouds ignore terrain from above; single shell segment | full segment up to the land hit |
| 12 | `ti.random` stateful | counter-based PCG hash |
| 13 | stale `sun_path_rot` on config load with GUI hidden | n/a (single source of state) |
| 14 | heightfield sphere trace stops at `abs(d) < 1e-4 t` with t from the camera, so from orbit hits land up to ~4 km above sea level in concentric bands | tolerance relative to the distance marched from atmosphere entry |
| 15 | ray march: 32 uniform left-endpoint steps per segment | 12 to 32 steps by segment length (Hillaire 2020's variable count), uniform in a 16-cell optical-depth CDF mixed with a 25% uniform share, jittered within each step; see `docs/research/20261019-raymarch-step-placement.md` |
| 16 | OpenDRT v0.2.2, contrast 1.0 (1.4 on the ray-marching branch) | OpenDRT v1.0.0b43 (upstream c8d698f) with the Default look, Hue Shift RGB 0.15/0.2/0.2, DaGrinchi tonescale, Oklab saturation 1.5, contrast offset 0, one setting for every mode; goldens from `python/opendrt.py` |
| 17 | ozone: two Gaussians at 25 km plus a floor below 15 km, scaled to a 173 DU column; 293 K cross sections read nearest-lower from bins [390 + i, 391 + i) nm | measured profile (`LUT/ozone_profile.dat`, McPeters and Labow 2011 global annual mean, 292 DU, 500 m samples interpolated linearly); 223 K cross sections (Serdyuchenko et al. 2014) averaged over bins centred on 390 + i nm and interpolated linearly; generators in `python/lut/`. The ozone goldens are numpy references, and the `bruneton` table and the networks, trained on the old ozone, need a rebake and retraining |
| 18 | ray march: one wavelength per sample; per-wavelength transmittance (64 x 64 x 441) and multiple-scattering (32 x 32 x 441) tables | four wavelengths per sample, stratified in the CIE CDF (`sample_wavelength_bundle`); transmittance from a 256 x 256 table of log column densities (exact for every wavelength, filtered in f32); the multiple-scattering table holds one bundle per texel (32 x 32 x 128 x rgba); see `docs/research/20261019-twilight-variance.md` |
| 19 | path tracer: one wavelength per path; surface bounces cosine weighted | a four-wavelength bundle per path: a hero drawn uniformly from the bundle decides the collisions under a majorant shared by all members, and the members are weighted by spectral MIS (balance heuristic, Wilkie et al. 2014); clouds are tracked before the RMO media; in both modes, bounces off the textured surface mix cosine weighting with the ocean lobes' GGX normals by one-sample MIS; see `docs/research/20261019-twilight-variance.md` |
| 20 | `main` and `data_prep` differ in physics: turbidity 1.06 vs 1.03, Rayleigh fit normalized by 1.225 vs 1.220554684, KN Mie on every scatter vs KN then HG g = 0.8, 25 vs 10 bounces | both presets share `data_prep`'s physics with 25 bounces (`atmosphere` in `src/render/scene.zig`); they differ in scene content only, so the ray marcher's tables bake identically for both |
| 21 | terrain: up to 250 sphere-tracing steps across the whole atmosphere segment, and a march that runs out of steps counts as a hit, which put a band of false ground above the horizon for cameras near sea level | the march spans the terrain shell only and skips stretches that a coarse bound map (`src/assets/MapBounds.zig`) clears, landing on the planet sphere directly where the bound is 0; only field steps count towards the 250, so the false band is gone; see `docs/research/20261019-path-tracer-performance.md` |
| 22 | path tracer: the sun's transmittance through the Rayleigh, aerosol, and ozone media ratio-tracked; clouds delta- and ratio-tracked under one global majorant | that transmittance read from the ray marcher's column-density table (now baked once in `GpuLuts` and shared by every mode), deterministic and within 1e-3 of a fine integral except about 1% on rays that graze the aerosol profile's kinks; clouds tracked under per-stretch majorants from the bound map; the terrain searched only in front of a cloud collision |

Kept on purpose: the Beckmann ocean lobe without `1/(4 nl nv)` (a tuned
look), and the `mu_s >= -0.2` clamp inside uvwz (it defines the MLP's input
domain). Existing MLP weights are trained against the unnormalized pdf, so MLP
mode applies one documented scale constant until the network is retrained on
`datagen` output.

## Verification

- **Zig unit tests:** asset parsers, tile planning, camera math, config round
  trip, and weight loading.
- **GPU tests,** skipped when no device is present:
  - The shim on its own: buffer and texture round trips, dispatch, parameter
    binding, and the layout check.
  - Every shared Slang library function against `tests/goldens`, which
    `python/goldens.py` generates by running the original Taichi functions on
    `ti.cpu` (plus a torch forward pass for the MLP).
  - The table above sets the expected differences.
- **Mode parity:** low-resolution headless renders (`earth render`) of fixed
  presets, compared against Taichi CPU renders with the known deltas isolated.
  Each mode is reported as a mean relative difference with side-by-side PNGs.
- **Shaders:** `zig build shaders` compile-checks every entry point. On Linux,
  Vulkan validation layers are enabled in Debug builds.

## Work breakdown

The groups run in order; packets within a group run in parallel, each in its
own git worktree branched from `zig-slang` and merged back in order.

1. **Scaffold** (main context): pixi.toml, build.zig and zon, CMake skeleton,
   `earth_native.h`, `internal.h`, the shader directory and uniform structs,
   the test runner, and an empty viewer that compiles.
2. **Foundations** (parallel):
   - **A. Native GPU:** `gpu.cpp`, the CMake build of slang-rhi, and a C smoke
     test.
   - **B. Native UI and present:** `ui.cpp` and `present.slang`.
   - **C. Zig host:** the `gpu/` wrappers, platform window, viewer loop,
     camera, controls, and config.
   - **D. Shared Slang library:** common, spectral, atmosphere, and post
     modules, plus `goldens.py` and the test kernels.
   - **E. Assets:** LUT loaders, the image cache and tile plan, weight export
     and loading, and the datagen file format.
3. **Modes** (parallel, after group 2 is merged):
   - **F. Path tracer:** both scene presets, Earth maps, BRDF, and clouds.
   - **G. Ray marcher:** the LUT bake and the march.
   - **H. MLP:** MLP mode and `datagen`.
   - **I. Panel and finishing:** the ImGui panel, display and screenshot,
     presets, and the `render` CLI.
4. **Integration** (main context and one reviewer): parity reports, removing
   Taichi, and a README rewrite.

## Risks

- **slang-rhi API churn.** Mitigated by pinning the SHA and keeping every
  slang-rhi call inside `gpu.cpp`/`ui.cpp`.
- **Zig 0.17 build API** (`translate_c` package). The scaffold proves it before
  any group starts.
- **Q2 texture memory** (about 2.5 GB) on integrated GPUs. The texture quality
  preset falls back to Q1.
- **Linear filtering of r32f 3D textures** is not universal. The LUT samplers
  query support and fall back to r16f.
- **This development host:** the 4090 also drives the display, so tests keep
  their resolutions modest and run one GPU process at a time.
