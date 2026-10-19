# Plan: renderer core (group 3 prerequisite)

Base: `zig-slang` after merging the four foundation packets. Branch `zs-core`.
The three render modes and the panel are built on top of this in parallel
afterwards, so the interfaces here must be explicit and documented.

## Scope

- `render/GpuLuts.zig`: uploads `assets.Luts` (CIE buffer verbatim, srgb2spec
  f32 buffer, ozone buffer, CRF rgba32f texture) and owns the shared samplers
  (linear clamp; linear REPEAT-u CLAMP-v for Earth maps).
- `render/Scene.zig`: presets `textured` (main) and `symmetric` (data_prep)
  producing `SceneUniforms` (flags, bounces, Russian roulette start,
  turbidity, cloud extinction/density, Rayleigh sea-level density in
  `media.w`, constant albedo, land height scale, night-light and star scales).
  Values from `docs/research/20261018-*-spec.md`.
- `render/Renderer.zig`: owns the rgba32f accumulation texture, the rgba8
  display texture, the frame index, and the active mode. `frame()` builds
  `FrameUniforms` from the camera and sun, records the mode's dispatch (one
  sample per pixel), then `display`. Resets accumulation when camera, sun,
  FOV, aspect, mode, preset, or resolution change; exposure/CRF/gamma only
  re-run `display`. Mode interface: a tagged union of mode structs, each with
  `init(gpa, device, shared)`, `deinit`, `record(cmd, uniforms, shared)`, and
  `invalidate(scene)` (for LUT rebakes). Implement a `debug` mode
  (`shaders/modes/debug.slang`: sphere + atmosphere-free sky, sun-lit Lambert)
  to prove camera, accumulation, and post end to end.
- `render/Post.zig` + `shaders/post/display.slang`: `color/spp * vignette *
  2^(exposure + bias)` -> OpenDRT (contrast per mode: 1.0 path tracer and MLP,
  1.4 ray marcher) -> CRF -> gamma -> sRGB OETF -> rgba8. `bias = -7.406`
  stops compensates the normalized spectral pdf (fix #2) so user-facing
  exposure keeps the Python meaning (default 2.5).
- `src/render/test_goldens.zig`: drives every set in `tests/goldens/*.json`
  through `shaders/tests/*.slang` per `tests/goldens/README.md`.
- CLI `earth render`: headless, `--mode --preset --config <zon> --width
  --height --spp --out <png>`, writes PNG via `assets.Image`.
- Viewer: replace the placeholder image with `Renderer`, minimal ImGui
  controls for mode, preset, exposure, CRF, gamma, sun angle and rotation, FOV.
