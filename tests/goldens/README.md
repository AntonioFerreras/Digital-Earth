# Golden values for the shared Slang library

`python/goldens.py` writes one JSON file per library area. Each holds the
reference outputs of the original Taichi functions (run on `ti.cpu`, from the
fixed commits in `SOURCES`), of the trained MLP in torch, of
`python/export_weights.py`'s reference forward pass, or of `python/opendrt.py`'s
port of the upstream OpenDRT DCTL, for the test kernels
in `shaders/tests`. Regenerate with

```
pixi run -e py python python/goldens.py --mlp-checkpoint <model_weights/mlp_e120.pth> [module ...]
```

The checkpoint is untracked; the legacy network set binds `mlp.emlp`, its
version 1 export, which the script checks against the checkpoint. The neural
sets' small networks are written to `neural/<set>.emlp`.

## File format

```
{
  "module": "atmosphere",
  "shader": "shaders/tests/test_atmosphere.slang",
  "sets": [
    {
      "function": "media_density",          // library function under test
      "entry": "test_media_density",        // compute entry point in `shader`
      "cases": 36,
      "input_stride": 3,                    // floats per case in `inputs`
      "output_stride": 3,                   // floats per case in `outputs`
      "reference": "taichi",                // or "corrected" / "port", below
      "fix": 9,                             // optional: outline behavior-change number
      "lut": {"path": "LUT/CIE.dat", "dtype": "f32"},   // optional
      "source": "...", "note": "...",       // provenance and layout notes
      "tolerance": {"abs": 1e-7, "rel": 2e-5},
      "inputs": [...],                      // cases * input_stride floats
      "outputs": [...]                      // cases * output_stride floats
    }
  ]
}
```

A value passes when `|got - want| <= abs + rel * |want|`. Each tolerance
covers the f32 rounding of the original formula plus FMA contraction on the
GPU. The values were checked against the library compiled as C++, both with
strict IEEE arithmetic and with FMA contraction.

`reference` values:

- `taichi`: the original function's output.
- `corrected`: a numpy reference of the fixed behavior, for the fix number
  given in `fix`. For `ray_sphere` only the miss rows differ from Taichi.
- `port`: a function that is new in the port, or one that had no standalone
  original (the CRF coordinates, the vignette). The numpy reference follows the
  rule the spec describes.
- `torch`: `python/model.py` with the checkpoint's weights, in float64.
- `upstream`: a float64 numpy port of a third-party reference that replaced the
  original's version (OpenDRT v1.0.0b43, `python/opendrt.py`).

Several sets can share an entry point, for example `scattering_uvwz` and
`scattering_uvwz_near_ground`. They differ only in input domain and tolerance.

## Driving a set

Every entry point in `shaders/tests` has the same interface. It is declared in
`shaders/tests/harness.slang`:

| Parameter | Type | Contents |
| --- | --- | --- |
| `inputs` | `StructuredBuffer<float>` | the set's `inputs` |
| `outputs` | `RWStructuredBuffer<float>` | `cases * output_stride` floats |
| `lut` | `StructuredBuffer<float>` | the set's `lut` file as f32. Convert f16 files. Omit it when the set has no `lut`. |
| `case_count` | `uniform uint` | `cases` |

`src/render/test_goldens.zig` drives every set as the Zig test
`golden <module>.<function>`; a set without such a test fails the suite.

Thread group size is 64 x 1 x 1. Dispatch `ceil(cases / 64)` groups, then
read back `outputs`. Thread `i` reads `inputs[i * input_stride ..]` and writes
`outputs[i * output_stride ..]`.

The lookup tables are uploaded exactly as the asset loader reads them:

| `lut.path` | Elements | Layout |
| --- | --- | --- |
| `LUT/CIE.dat` | 2 x 441 x 3 f32 | `[3*i + c]`: CDF of channel c at 390 + i nm (0 at i = 0, 1 at i = 440). `[3*(441 + i) + c]`: CIE 1931 2-degree CMF |
| `LUT/srgb2spec.dat` | 300 x 3 f16 | `[3*i + c]`: weight of sRGB channel c at 400 + i nm |
| `LUT/ozone_cross_section.dat` | 441 f32 | cm^2 per molecule, mean over the 1 nm bin centred on 390 + i nm |
| `LUT/ozone_profile.dat` | 221 f32 | ozone number density (m^-3) at altitude 500 i m |
| `tests/goldens/mlp.emlp`, `tests/goldens/neural/*.emlp` (`dtype` `emlp`) | the whole file | a network as `assets.Network` loads it and the neural mode uploads it; the kernel is specialized for it |

## Sets

| Entry | In | Out | Input layout -> output layout |
| --- | --- | --- | --- |
| `test_ray_sphere` | 7 | 2 | pos, dir, radius -> t_near, t_far |
| `test_rng` | 4 | 4 | pixel.x, pixel.y, frame, dimension -> `Rng.next4()` |
| `test_sample_cone` | 6 | 3 | u.xy, cos_theta_max, axis -> direction |
| `test_sample_hemisphere_cosine` | 5 | 3 | u.xy, n -> direction |
| `test_sample_sphere_uniform` | 2 | 3 | u.xy -> direction |
| `test_sample_wavelength` | 1 | 5 | u -> wavelength, cmf xyz, rcp_pdf |
| `test_cie_cmf` | 1 | 3 | wavelength -> cmf xyz |
| `test_xyz_to_rec709` | 3 | 3 | XYZ -> linear Rec.709 |
| `test_srgb_to_spectrum` | 4 | 1 | rgb, wavelength -> reflectance |
| `test_planck` | 2 | 1 | temperature, wavelength -> radiance |
| `test_media_density` | 3 | 3 | h, turbidity, Rayleigh sea-level density -> Rayleigh, Mie, ozone (m^-3) |
| `test_media_extinction` | 2 | 3 | wavelength, turbidity -> Rayleigh, Mie (per metre), ozone (m^2) |
| `test_step_integral` | 1 | 1 | optical depth -> integral |
| `test_phase` | 3 | 6 | cos, g, alpha -> Rayleigh, HG(g), KN(3000), Draine(g, alpha), cloud, cloud reduced |
| `test_sample_phase` | 6 | 12 | u.xyz, view -> HG(0.8), KN(3000), cloud, cloud reduced |
| `test_scattering_uvwz` | 4 | 5 | r, mu, mu_s, nu -> uvwz, intersects_ground |
| `test_scattering_params_from_uvwz` | 4 | 5 | uvwz -> r, mu, mu_s, nu, intersects_ground |
| `test_scattering_params_from_ray` | 9 | 5 | pos, dir, sun -> r, mu, mu_s, nu, intersects_ground |
| `test_canonical_ray` | 4 | 9 | r, mu, mu_s, nu -> pos, dir, sun |
| `test_transmittance_uv` | 2 | 2 | mu, r -> uv (64 x 64 LUT) |
| `test_transmittance_mu_r` | 2 | 2 | uv -> mu, r (64 x 64 LUT) |
| `test_sphere_uv` | 3 | 2 | unit n -> uv |
| `test_grade_albedo` | 4 | 3 | albedo rgb, ocean -> graded rgb |
| `test_earth_brdf` | 12 | 2 | albedo, ocean, bathymetry, v, n, l -> brdf, n_dot_l |
| `test_cloud_density` | 3 | 1 | r, coverage, max density -> density |
| `test_opendrt` | 6 | 3 | rgb, saturation, contrast, tonescale preset -> display rgb |
| `test_crf_coords` | 5 | 4 | display rgb, index, count -> u rgb, v |
| `test_vignette` | 4 | 1 | pixel x, y, width, height -> factor |
| `test_srgb_oetf` | 3 | 3 | linear rgb -> encoded rgb |
| `test_network` | 4 | 3 | uvwz -> linear Rec.709 (the renderer's scale) |
| `test_erf` | 1 | 1 | x -> erf(x) |

Booleans are written as 0.0 or 1.0. Integer inputs are stored as exact floats.
