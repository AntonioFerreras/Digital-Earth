# Plan: foundations (group 2 of the outline)

Base: branch `zig-slang` at the scaffold commit. Each packet works in its own
worktree on branch `zs-<packet>` and commits there; the integrator merges in
order native -> host -> slang -> assets and resolves `build.zig` conflicts.

Shared facts for every packet:

- Environment: `pixi run -e default <cmd>` inside the worktree (zig 0.17.0,
  cmake, ninja, gcc 15, SDL3, Vulkan loader + validation layers). Python tools
  use `pixi run -e py <cmd>` (feature `py`; install it on first use).
- `zig build` configures, builds, and installs the native CMake project into
  `build/native/install`, translates `src/c.h` (earth_native.h, dcimgui.h,
  SDL3) into the `c` module, and links `earth`. `zig build test -- --test-filter X`
  runs tests with the build root as cwd. Zig 0.17 notes: `std.process.Init`
  main, `std.Io`, `b.addTranslateC`, `run.addPassthruArgs()`, optimize enum is
  `.debug/.safe/.fast/.small`, default Linux target pins glibc 2.28 (host libc
  detection fails inside conda).
- Contracts frozen for this group: `native/include/earth_native.h`,
  `shaders/shared/uniforms.slang` <-> `src/render/uniforms.zig`. A packet that
  needs a contract change stops and reports instead of editing it.
- Specs: `docs/research/20261018-*.md`; design: `docs/outlines/20261018-zig-slang-rewrite.md`.
- Dev host: RTX 4090 that also drives the X11 display; keep GPU tests small
  (<= 512x512, < 1 GB) and run one GPU process at a time.

## Packet N: native library (`native/`)

Implement `gpu.cpp` and `ui.cpp` against `earth_native.h` with slang-rhi at the
pinned SHA (headers in `build/native/_deps/slang_rhi-src/include`).

- Device: backend mapping, validation with a debug callback to stderr,
  shader search paths, info (limits, r32f linear-filter support via format
  support query).
- Resources: buffers, textures (2D, 2D array, 3D; region upload; layer read),
  samplers.
- Kernels: load module + entry, compute pipeline, thread-group size via
  reflection, `find_param` (ShaderOffset packed into `en_param`, kind from the
  type layout) and `find_field` (field offset/size in a ConstantBuffer or struct
  parameter), diagnostics into `en_last_error`.
- Commands: dispatch with bindings, barriers so consecutive dispatches observe
  each other's writes, submit.
- Surface/present: X11/Win32/Cocoa handles, configure, out-of-date mapping,
  render pipeline in `native/shaders/present.slang` drawing the image
  (fullscreen triangle, linear sampler) then ImGui draw lists (scissor,
  vertex/index buffers, ImGui 1.92+ texture protocol
  `ImGuiBackendFlags_RendererHasTextures`).
- UI: ImGui context, SDL3 platform backend, new frame, shutdown.
- Tests: a CMake-optional C test executable (`EN_BUILD_TESTS`) covering buffer
  and texture round trips, dispatch + param binding + field lookup, two
  dependent dispatches, compile-error diagnostics, and a headless surfaceless
  path; run it on Vulkan.

## Packet H: Zig host (`src/`)

- `heap.zig`; full `test_runner.zig` (repeatable `--test-filter`, per-test
  leak check via `std.testing.allocator` reset, PASS/SKIP/FAIL with timing,
  `error.SkipZigTest` as skip).
- `gpu.zig` + `gpu/`: Device, Buffer, Texture, Sampler, Kernel, CommandList,
  Surface wrapping the `c` module with Zig error sets and `en_last_error`
  logging. `Kernel.bindingsFor(comptime T, path)` style helpers: validate an
  extern struct `T` against `en_kernel_find_field` for every field (offset
  and size) once at kernel creation.
- `platform.zig` + `platform/Window.zig`: SDL3 window (force the X11 video
  driver on Linux), native handle via window properties, event pump forwarding
  to `en_ui_process_event`, resize handling.
- `viewer.zig` + `viewer/`: Camera (f64 host state, the Taichi controls and
  speeds from the specs), Controls, Config (ZON save/load), Viewer loop that
  opens the window, creates device + surface + UI, presents a placeholder
  image, and draws an empty ImGui panel.
- `main.zig`: CLI `view` (default), `render`, `datagen`, `bake-textures`
  (the last three are stubs).
- Unit tests for camera math and config round trip; GPU tests skip while the
  native stubs return `EN_ERR_UNSUPPORTED`.

## Packet S: shared Slang library (`shaders/`, `python/goldens.py`)

- Modules: `common/` (constants, ray-sphere with explicit miss, PCG hash RNG,
  sampling: cone, hemisphere cosine, sphere uniform), `spectral/` (CIE CDF
  inversion over an f32 buffer with normalized pdf, CMF lookup, srgb_to_spectrum
  fixed, Planck, XYZ->Rec.709), `atmosphere/` (densities, extinction, phase
  functions incl. KN/HG/Draine, Bruneton uvwz both directions, step integral),
  `earth/` (sphere uv, material grading, earth_brdf, cloud density profile),
  `post/` (OpenDRT with M v, CRF lookup, vignette, sRGB OETF).
- Pure functions with resources passed as parameters, so every mode reuses
  them. Behavior changes per the outline table, nothing else.
- `python/goldens.py`: imports the original Taichi functions from the repo
  root (`lib/`, `pathtracer.py`), runs them on `ti.cpu` over deterministic
  input grids, and writes `tests/goldens/<module>.json`
  (`{function, inputs: [...], outputs: [...], tolerance}`); for fixed bugs it
  records the corrected reference (numpy) and notes which.
- `shaders/tests/*.slang`: one compute entry per golden set reading inputs
  from a `StructuredBuffer<float>` and writing outputs.
- `zig build shaders`: runs `build/native/install/bin/slangc` on every entry
  point in `shaders/modes`, `shaders/post`, `shaders/tests` for SPIR-V and
  Metal targets.

## Packet A: assets (`src/assets/`, `python/`)

- Move `train.py`, `model.py`, `LUT/*_generator.py` into `python/` (git mv);
  `python/export_weights.py` (state dict -> header + f32 blob, strips
  `_orig_mod.`); `train.py` reads the binary datagen format
  (`<magic, count>` header + `u v w z R G B` f32 records) in addition to text.
- Zig: `assets/Luts.zig` (CIE.dat, srgb2spec.dat f16 -> f32, ozone, CRF `.rf`
  with Neutral first then filename order, duplicate dropped),
  `assets/MlpWeights.zig`, `assets/Image.zig` (stb_image/stb_image_write
  vendored as C via build.zig.zon), `assets/EarthMaps.zig` (texture-set table
  Q1/Q2, PNG -> raw cache with south-first rows and channel-0 extraction,
  cache reader), `assets/TilePlan.zig` (longitude tiles with one-texel apron
  for a max dimension; pure, unit-tested), `assets/datagen_format.zig`.
- Unit tests on real files in the repo (`LUT/`) and on small synthetic PNGs;
  Earth PNGs live in the main checkout at `~/Digital-Earth/textures/`
  (gitignored); do not bake the 21K set in tests.
