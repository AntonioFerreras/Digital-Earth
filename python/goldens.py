"""Golden reference values for the shared Slang library.

Evaluates the original Taichi functions on `ti.cpu` over deterministic input
grids and writes tests/goldens/<module>.json, one entry per test kernel in
shaders/tests (layout in tests/goldens/README.md). Functions whose behavior the
port fixes get a numpy reference of the corrected behavior instead, marked
`"reference": "corrected"` with the fix number from the outline's
"Behavior changes" table; functions new to the port are marked "port".

Sources are the Taichi renderer at three fixed revisions (`SOURCES`): the
`data_prep` branch (symmetric scene, MLP domain), `main` (textured Earth), and
`ray-marching-approimation` (LUT mappings). Each is
extracted from git history into a temporary directory as its own package, so
same-named modules load side by side. The neural sets come from a torch
checkpoint, `--mlp-checkpoint`, and from python/export_weights.py's reference;
OpenDRT, newer than the original's, from python/opendrt.py's port of upstream.

Run: pixi run -e py python python/goldens.py [module ...]
"""

import argparse
import importlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import taichi as ti
import torch

import export_weights
import opendrt
from tiles import cube

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"

ti.init(arch=ti.cpu, random_seed=0)

# The original samplers draw with ti.random(); the goldens feed them scripted
# numbers instead, so the Slang samplers can take the same `u` as input.
# Kernels serialize their loop, which makes the draw order deterministic.
MAX_DRAWS = 1 << 16
SCRIPTED = ti.field(ti.f32, shape=MAX_DRAWS)
CURSOR = ti.field(ti.i32, shape=())
# Index of the case being evaluated, read by the texture-sampling stand-in.
CASE = ti.field(ti.i32, shape=())


@ti.func
def scripted_random():
    value = SCRIPTED[CURSOR[None]]
    CURSOR[None] += 1
    return value


ti.random = scripted_random

# Package name -> the commit of the Taichi renderer it holds; commits, not
# branch names, so the goldens regenerate identically after the branches move.
SOURCES = {"dp_lib": "9c7d1df", "main_lib": "26a95b7", "rm_lib": "eae9d88"}


def extract_revision(revision: str, package: str, root: Path) -> None:
    """Writes `revision`'s lib/*.py and pathtracer.py as package `package` under root."""
    names = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "--name-only", revision, "lib/"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    target = root / package
    target.mkdir()
    (target / "__init__.py").write_text("")
    for path in [n for n in names if n.endswith(".py")] + ["pathtracer.py"]:
        source = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{revision}:{path}"],
            check=True, capture_output=True, text=True,
        ).stdout
        source = re.sub(r"\b(from|import) lib\.", rf"\1 {package}.", source)
        (target / Path(path).name).write_text(source)


scratch = tempfile.TemporaryDirectory(prefix="goldens-")
for package, revision in SOURCES.items():
    extract_revision(revision, package, Path(scratch.name))
sys.path.insert(0, scratch.name)

dp_math = importlib.import_module("dp_lib.math_utils")
dp_sampling = importlib.import_module("dp_lib.sampling")
dp_colour = importlib.import_module("dp_lib.colour")
dp_volume = importlib.import_module("dp_lib.volume_rendering_models")
dp_bruneton = importlib.import_module("dp_lib.bruneton_mappings")
main_volume = importlib.import_module("main_lib.volume_rendering_models")
main_surface = importlib.import_module("main_lib.surface_rendering_models")
main_pt = importlib.import_module("main_lib.pathtracer")
rm_lut = importlib.import_module("rm_lib.lut_sampling")

PLANET_R = 6371e3
ATMOS_R = PLANET_R + 110e3

CIE = np.fromfile(ROOT / "LUT" / "CIE.dat", dtype=np.float32)
SRGB2SPEC = np.fromfile(ROOT / "LUT" / "srgb2spec.dat", dtype=np.float16).astype(np.float32)
OZONE = np.fromfile(ROOT / "LUT" / "ozone_cross_section.dat", dtype=np.float32)
OZONE_PROFILE = np.fromfile(ROOT / "LUT" / "ozone_profile.dat", dtype=np.float32)
# Lookup tables a set binds as `lut`: the file and its on-disk element type;
# the host uploads them as f32.
CIE_LUT = {"path": "LUT/CIE.dat", "dtype": "f32"}
SRGB2SPEC_LUT = {"path": "LUT/srgb2spec.dat", "dtype": "f16"}
OZONE_LUT = {"path": "LUT/ozone_cross_section.dat", "dtype": "f32"}
OZONE_PROFILE_LUT = {"path": "LUT/ozone_profile.dat", "dtype": "f32"}
# The legacy network as the original export wrote it: version 1.
EMLP_LUT_PATH = GOLDENS / "mlp.emlp"
assert CIE.size == 2 * 441 * 3 and SRGB2SPEC.size == 300 * 3 and OZONE.size == 441
assert OZONE_PROFILE.size == 221

O3_FIELD = ti.field(ti.f32, shape=441)
O3_FIELD.from_numpy(OZONE)

# Map stand-ins for main's get_land_material and get_clouds_density: every
# "sampler" is a per-case field, indexed by the case being evaluated.
ALBEDO_MAP = ti.Vector.field(4, ti.f32, shape=MAX_DRAWS)
OCEAN_MAP = ti.Vector.field(4, ti.f32, shape=MAX_DRAWS)
ZERO_MAP = ti.Vector.field(4, ti.f32, shape=MAX_DRAWS)


@ti.func
def sample_map(sampler: ti.template(), pos):
    return sampler[CASE[None]]


main_pt.sample_sphere_texture = sample_map

rng = np.random.default_rng(20261018)


def evaluate(func, inputs, out_dim, draws_per_case=0, draws=None):
    """Runs ti.func `func(x) -> vector` over the rows of `inputs` on ti.cpu."""
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    n, k = inputs.shape
    if draws_per_case:
        assert draws.shape == (n, draws_per_case) and draws.size <= MAX_DRAWS
        SCRIPTED.from_numpy(np.resize(draws.astype(np.float32).ravel(), MAX_DRAWS))
    CURSOR[None] = 0
    source = ti.field(ti.f32, shape=(n, k))
    result = ti.field(ti.f32, shape=(n, out_dim))
    source.from_numpy(inputs)

    @ti.kernel
    def run():
        ti.loop_config(serialize=True)
        for i in range(n):
            CASE[None] = i
            x = ti.Vector([source[i, j] for j in ti.static(range(k))])
            y = func(x)
            for j in ti.static(range(out_dim)):
                result[i, j] = y[j]

    run()
    assert CURSOR[None] == n * draws_per_case, "draw count differs from the scripted layout"
    return result.to_numpy()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_units(n):
    return unit(rng.normal(size=(n, 3)))


def golden_set(function, entry, inputs, outputs, tolerance, reference="taichi",
               source=None, fix=None, lut=None, note=None):
    inputs = np.asarray(inputs, dtype=np.float32)
    outputs = np.asarray(outputs, dtype=np.float32)
    assert inputs.ndim == 2 and outputs.ndim == 2 and len(inputs) == len(outputs)
    assert np.all(np.isfinite(outputs)), f"{function}: non-finite reference values"
    entry_set = {
        "function": function,
        "entry": entry,
        "cases": len(inputs),
        "input_stride": inputs.shape[1],
        "output_stride": outputs.shape[1],
        "reference": reference,
        "source": source,
        "tolerance": tolerance,
    }
    if fix is not None:
        entry_set["fix"] = fix
    if lut is not None:
        entry_set["lut"] = lut
    if note is not None:
        entry_set["note"] = note
    entry_set["inputs"] = [float(f"{x:.9g}") for x in inputs.ravel()]
    entry_set["outputs"] = [float(f"{x:.9g}") for x in outputs.ravel()]
    return entry_set


def tol(abs_tol, rel_tol):
    return {"abs": abs_tol, "rel": rel_tol}


def write(module, shader, sets):
    path = GOLDENS / f"{module}.json"
    document = {"module": module, "shader": shader, "sets": sets}
    path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    size = path.stat().st_size
    assert size < 200_000, f"{path} is {size} bytes"
    print(f"{path.relative_to(ROOT)}: {len(sets)} sets, {size} bytes")


# common ----------------------------------------------------------------------


def pcg4d(v):
    v = v.astype(np.uint32)
    with np.errstate(over="ignore"):
        v = v * np.uint32(1664525) + np.uint32(1013904223)
        for shift in (True, False):
            v[:, 0] += v[:, 1] * v[:, 3]
            v[:, 1] += v[:, 2] * v[:, 0]
            v[:, 2] += v[:, 0] * v[:, 1]
            v[:, 3] += v[:, 1] * v[:, 2]
            if shift:
                v ^= v >> np.uint32(16)
    return v


def common_sets():
    sets = []

    # Ray-sphere: outside/inside origins, horizon rays 1 km above and below
    # tangency (far beyond f32 discriminant noise), and clean misses.
    radii = np.array([PLANET_R, PLANET_R + 4000.0, PLANET_R + 10000.0, ATMOS_R])
    rows = []
    for radius in radii:
        rows.append([-15e6, 0, 0, 1, 0, 0, radius])
        rows.append([-15e6, 0, 15e6, *unit([1, 0, -1]), radius])
        rows.append([0, PLANET_R + 100.0, 0, 1, 0, 0, radius])
        rows.append([0, PLANET_R + 100.0, 0, 0, 1, 0, radius])
        rows.append([0, PLANET_R + 100.0, 0, 0, -1, 0, radius])
        for offset in (-1000.0, 1000.0):
            rows.append([-2e7, radius + offset, 0, 1, 0, 0, radius])
        rows.append([-15e6, 0, 0, -1, 0, 0, radius])
        rows.append([-15e6, 9e6, 0, 1, 0, 0, radius])
    for pos_r in (PLANET_R + 10.0, PLANET_R + 5000.0, PLANET_R + 50e3, 7e6, 2e7):
        for pos, d in zip(random_units(6) * pos_r, random_units(6)):
            rows.append([*pos, *d, radii[rng.integers(4)]])
    rows = np.array(rows, dtype=np.float32)

    @ti.func
    def rsi(x):
        return dp_math.rsi(ti.Vector([x[0], x[1], x[2]]), ti.Vector([x[3], x[4], x[5]]), x[6])

    out = evaluate(rsi, rows, 2)
    miss = np.isnan(out).any(axis=1)
    out[miss] = -1.0
    sets.append(golden_set(
        "ray_sphere", "test_ray_sphere", rows, out, tol(64.0, 1e-5),
        reference="corrected", fix=1, source="lib/math_utils.py rsi",
        note="Taichi output where it hits; (-1, -1) where the original returns NaN. "
             "Horizon rows sit 1 km off tangency; the absolute tolerance covers f32 "
             "cancellation in b*b - |p|^2 + r^2 at planet scale.",
    ))

    keys = np.array([
        [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [1919, 1079, 0, 0], [1919, 1079, 1000000, 63], [511, 511, 4096, 7],
        *rng.integers(0, 2048, size=(24, 4)),
    ], dtype=np.uint32)
    bits = pcg4d(keys)
    sets.append(golden_set(
        "Rng.next4", "test_rng", keys.astype(np.float32),
        (bits >> np.uint32(8)).astype(np.float64) / 16777216.0, tol(0.0, 0.0),
        reference="port", fix=12, source="pcg4d, Jarzynski and Olano 2020",
        note="Inputs are pixel.x, pixel.y, frame, dimension; outputs are exact.",
    ))

    u = np.array([[0.0, 0.0], [0.5, 0.25], [0.999999, 0.999999], [0.25, 0.75], [1e-6, 0.5]])
    axes = np.vstack([[0, 1, 0], [0, -1, 0], [1, 0, 0], unit([0.3, 0.95, 0.1]), random_units(4)])
    rows = np.array([[*uu, c, *a] for uu in u for c in (math.cos(6.95e8 / 1.4959e11), 0.5, -1.0)
                     for a in axes])

    @ti.func
    def cone(x):
        return dp_sampling.sample_cone_oriented(x[2], ti.Vector([x[3], x[4], x[5]]))

    sets.append(golden_set(
        "sample_cone", "test_sample_cone", rows,
        evaluate(cone, rows, 3, 2, rows[:, :2]), tol(2e-5, 0.0),
        source="lib/sampling.py sample_cone_oriented",
    ))

    def sphere_point(uu):
        z = 1.0 - 2.0 * uu[0]
        phi = 2.0 * math.pi * uu[1]
        return np.array([math.sqrt(1.0 - z * z) * math.cos(phi), math.sqrt(1.0 - z * z) * math.sin(phi), z])

    # The sampler normalizes n plus a unit-sphere point; where that point is
    # antipodal to n the result is rounding noise, so those rows are left out.
    rows = np.array([[*uu, *a] for uu in u for a in axes if np.linalg.norm(a + sphere_point(uu)) > 1e-3])

    @ti.func
    def hemisphere(x):
        return dp_sampling.sample_hemisphere_cosine_weighted(ti.Vector([x[2], x[3], x[4]]))

    sets.append(golden_set(
        "sample_hemisphere_cosine", "test_sample_hemisphere_cosine", rows,
        evaluate(hemisphere, rows, 3, 2, rows[:, :2]), tol(2e-5, 0.0),
        source="lib/sampling.py sample_hemisphere_cosine_weighted",
    ))

    rows = np.vstack([u, rng.random((16, 2))])

    @ti.func
    def sphere(x):
        return dp_sampling.sample_sphere(ti.Vector([x[0], x[1]]))

    sets.append(golden_set(
        "sample_sphere_uniform", "test_sample_sphere_uniform", rows,
        evaluate(sphere, rows, 3), tol(2e-5, 0.0), source="lib/sampling.py sample_sphere",
    ))
    return sets


# spectral --------------------------------------------------------------------


def cie_cdf_sums_f32():
    cdf = CIE[: 441 * 3].reshape(441, 3)
    return (cdf[:, 0] + cdf[:, 1]) + cdf[:, 2]


def cie_cmf_reference(wavelength):
    cmf = CIE[441 * 3:].reshape(441, 3).astype(np.float64)
    x = np.clip(wavelength - 390.0, 0.0, 440.0)
    i = np.minimum(x.astype(np.int64), 439)
    t = (x - i)[:, None]
    return cmf[i] * (1.0 - t) + cmf[i + 1] * t


def sample_wavelength_reference(u):
    """Bisection on the f32 CDF sums exactly as the shader evaluates it."""
    sums = cie_cdf_sums_f32()
    assert sums[0] == 0.0 and sums[-1] == 3.0
    rows = []
    for value in u.astype(np.float32):
        target = value * np.float32(3.0)
        lo, hi = 0, 440
        for _ in range(9):
            mid = (lo + hi) // 2
            if sums[mid] <= target:
                lo = mid
            else:
                hi = mid
        i = min(lo, 439)
        rise = sums[i + 1] - sums[i]
        t = min(max((np.float64(target) - sums[i]) / np.float64(rise), 0.0), 1.0) if rise > 0 else 0.0
        wavelength = 390.0 + i + t
        rows.append([wavelength, *cie_cmf_reference(np.array([wavelength]))[0],
                     3.0 / np.float64(rise) if rise > 0 else 0.0])
    return np.array(rows)


def original_spectral_scale():
    """Mean ratio of the original spectral estimator to the normalized one.

    The original divides L * cmf by 0.5 * sum(cmf) (its CDF row read at u = 1
    under REPEAT averages the last and first texels). The normalized estimator
    divides by the true density, so the expected Rec.709 output changes by this
    factor; exposure and the MLP scale compensate for it.
    """
    cmf = CIE[441 * 3:].reshape(441, 3).astype(np.float64)
    density = np.gradient(cie_cdf_sums_f32().astype(np.float64)) / 3.0
    weight = np.where(cmf.sum(axis=1) > 0, density / (0.5 * cmf.sum(axis=1)), 0.0)
    # Expected Y of a flat unit spectrum under each estimator.
    original = np.sum(cmf[:, 1] * weight)
    normalized = np.sum(cmf[:, 1])
    return original / normalized


def spectral_sets():
    sets = []
    layout = "Outputs: wavelength nm, cmf xyz, rcp_pdf (nm). "
    u = np.concatenate([[0.0, 1e-7, 1e-4, 0.25, 0.5, 0.75], np.linspace(0.0, 0.99, 34)[1:],
                        rng.random(24) * 0.99])
    sets.append(golden_set(
        "sample_wavelength", "test_sample_wavelength", u[:, None], sample_wavelength_reference(u),
        tol(1e-4, 1e-4), reference="corrected", fix=2, lut=CIE_LUT,
        source="lib/colour.py spectrum_sample",
        note=layout + "With the REPEAT read at u = 1, the original estimator's expected "
             f"output is {original_spectral_scale():.6g} x the normalized one "
             f"({math.log2(original_spectral_scale()):.3f} stops).",
    ))
    u = np.array([0.995, 0.9999, 0.999999, 1.0 - 2.0**-24])
    sets.append(golden_set(
        "sample_wavelength_tail", "test_sample_wavelength", u[:, None], sample_wavelength_reference(u),
        tol(5e-2, 1e-4), reference="corrected", fix=2, lut=CIE_LUT,
        source="lib/colour.py spectrum_sample",
        note=layout + "Red-tail CDF steps are a few f32 ulps, so contracting 3u - cdf into an "
             "FMA moves the wavelength within its 1 nm segment.",
    ))

    wavelengths = np.array([380.0, 390.0, 390.5, 399.75, 400.0, 555.3, 699.0, 699.5, 829.9, 830.0, 840.0])
    sets.append(golden_set(
        "cie_cmf", "test_cie_cmf", wavelengths[:, None], cie_cmf_reference(wavelengths),
        tol(1e-7, 1e-5), reference="port", fix=2, lut=CIE_LUT,
        note="Linear interpolation of the 1 nm CMF rows; the original read them through a "
             "texture half a texel off.",
    ))

    rows = np.vstack([np.eye(3), [[0.95, 1.0, 1.089], [0.2, 0.3, 0.1], [1e-3, 2e-3, 5e-4]],
                      rng.random((8, 3)) * 2.0])

    @ti.func
    def xyz_to_rgb(x):
        return dp_colour.xyzToRGBMatrix_D65 @ x

    sets.append(golden_set(
        "xyz_to_rec709", "test_xyz_to_rec709", rows, evaluate(xyz_to_rgb, rows, 3),
        tol(1e-6, 1e-6), source="lib/colour.py xyzToRGBMatrix_D65",
    ))

    basis = SRGB2SPEC.reshape(300, 3).astype(np.float64)
    wl = np.array([380.0, 390.0, 399.5, 400.0, 400.5, 401.0, 455.25, 550.0, 698.7, 699.0, 699.5, 830.0])
    colours = np.array([[0.3, 0.3, 0.3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.8, 0.5, 0.2]])
    rows = np.array([[*c, w] for c in colours for w in wl])
    x = np.clip(rows[:, 3] - 400.0, 0.0, 299.0)
    i = np.minimum(x.astype(np.int64), 298)
    t = (x - i)[:, None]
    spectrum = np.sum(rows[:, :3] * (basis[i] * (1.0 - t) + basis[i + 1] * t), axis=1)
    sets.append(golden_set(
        "srgb_to_spectrum", "test_srgb_to_spectrum", rows, spectrum[:, None], tol(1e-6, 1e-5),
        reference="corrected", fix=3, lut=SRGB2SPEC_LUT,
        source="lib/colour.py srgb_to_spectrum",
    ))

    rows = np.array([[t, w] for t in (2700.0, 5778.0, 10000.0)
                     for w in (390.0, 400.0, 500.0, 555.5, 699.0, 830.0)])

    @ti.func
    def planck(x):
        return ti.Vector([dp_colour.plancks(x[0], x[1])])

    sets.append(golden_set(
        "planck", "test_planck", rows, evaluate(planck, rows, 1), tol(0.0, 2e-5),
        source="lib/colour.py plancks",
    ))
    return sets


# atmosphere ------------------------------------------------------------------


def ozone_density_reference(heights):
    """LUT/ozone_profile.dat linearly interpolated per 500 m from sea level, 0 above 110 km."""
    x = np.maximum(heights, 0.0) / 500.0
    i = np.minimum(x.astype(np.int64), OZONE_PROFILE.size - 2)
    t = x - i
    profile = OZONE_PROFILE.astype(np.float64)
    density = profile[i] * (1.0 - t) + profile[i + 1] * t
    return np.where(x > OZONE_PROFILE.size - 1, 0.0, density)


def ozone_extinction_reference(wavelengths):
    """LUT/ozone_cross_section.dat (cm^2) linearly interpolated per nm from 390 nm, clamped, in m^2."""
    x = np.clip(wavelengths - 390.0, 0.0, OZONE.size - 1.0)
    i = np.minimum(x.astype(np.int64), OZONE.size - 2)
    t = x - i
    table = OZONE.astype(np.float64)
    return 1e-4 * (table[i] * (1.0 - t) + table[i + 1] * t)


def mie_density_reference(heights):
    """shaders/atmosphere/media.slang mie_density at unit turbidity, altitude clamped at sea level."""
    h = np.maximum(heights.astype(np.float64), 0.0)
    stratosphere = (h - 20000.0) / 4000.0
    return 0.85603 * np.exp(-h / 729.1) + 0.14397 * np.exp(-h / 3650.0) + 0.003818 * np.exp(-stratosphere * stratosphere)


def atmosphere_sets():
    sets = []
    heights = np.array([-100.0, 0.0, 1.0, 1299.0, 1300.0, 1301.0, 2399.0, 2400.0, 2401.0, 8000.0,
                        11499.0, 11500.0, 11501.0, 15000.0, 25000.0, 25250.0, 50000.0, 109999.0,
                        110000.0, 120000.0])
    blocks = []
    for module, turbidity, sea_level in ((dp_volume, 1.03, 1.220554684), (main_volume, 1.06, 1.225)):
        rows = np.array([[h, turbidity, sea_level] for h in heights])

        @ti.func
        def density(x):
            return module.get_density(x[0])

        outputs = evaluate(density, rows, 3)
        outputs[:, 1] = turbidity * mie_density_reference(heights)
        outputs[:, 2] = ozone_density_reference(heights)
        blocks.append((rows, outputs))
    sets.append(golden_set(
        "media_density", "test_media_density", np.vstack([b[0] for b in blocks]),
        np.vstack([b[1] for b in blocks]), tol(1e-7, 2e-5), reference="corrected", fix=17,
        lut=OZONE_PROFILE_LUT,
        source="lib/volume_rendering_models.py get_density (data_prep: turbidity 1.03, "
               "Rayleigh 1.220554684; main: 1.06, 1.225)",
        note="Inputs: altitude m, turbidity, Rayleigh sea-level density. Rayleigh is Taichi's; "
             "Mie is the fit to Elterman's profile (mie_density_reference) in place of the "
             "original's plateau; ozone is the number density (m^-3) of the measured profile "
             "(ozone_density_reference) in place of the original's analytic shape.",
    ))

    wavelengths = np.array([390.0, 400.0, 450.5, 555.0, 699.0, 829.9, 830.0, 830.9])
    blocks = []
    for module, turbidity in ((dp_volume, 1.03), (main_volume, 1.06)):
        rows = np.array([[w, turbidity] for w in wavelengths])

        @ti.func
        def extinction(x):
            return ti.Vector([module.spectra_extinction_rayleigh(x[0]),
                              module.spectra_extinction_mie(x[0]),
                              module.spectra_extinction_ozone(x[0], O3_FIELD)])

        outputs = evaluate(extinction, rows, 3)
        outputs[:, 2] = ozone_extinction_reference(wavelengths)
        blocks.append((rows, outputs))
    sets.append(golden_set(
        "media_extinction", "test_media_extinction", np.vstack([b[0] for b in blocks]),
        np.vstack([b[1] for b in blocks]), tol(0.0, 1e-3), reference="corrected", fix=17,
        lut=OZONE_LUT,
        source="lib/volume_rendering_models.py spectra_extinction_{rayleigh,mie,ozone}",
        note="Inputs: wavelength nm, turbidity. Outputs: Rayleigh, Mie per metre at unit density; "
             "ozone cross section in m^2 (ozone_extinction_reference), the original's "
             "nearest-lower lookup times its fixed number density replaced. n*n - 1 for air "
             "(n ~ 1.0003) keeps ~3e-4 relative f32 rounding, which an FMA contraction removes, "
             "hence the relative tolerance; every value is far from 0.",
    ))

    od = np.array([0.0, 1e-12, 1e-8, 1e-5, 9.9e-5, 1e-4, 1e-3, 9.9e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4])
    integral = np.where(od > 0, -np.expm1(-od) / np.where(od > 0, od, 1.0), 1.0)
    sets.append(golden_set(
        "step_integral", "test_step_integral", od[:, None], integral[:, None], tol(0.0, 2e-5),
        reference="corrected", fix=9,
        source="ray-marching-approimation pathtracer.py ray_march_rmo step_integral",
        note="Exact (1 - exp(-od)) / od.",
    ))

    cosines = np.array([-1.0, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0])
    rows = np.array([[c, g, a] for c in cosines for g, a in ((0.8, 1.0), (0.5409, 20.33))])

    @ti.func
    def phases(x):
        return ti.Vector([dp_volume.rayleigh_phase(x[0]),
                          dp_volume.hg_phase(x[0], x[1]),
                          dp_volume.klein_nishina_phase(x[0], dp_volume.mie_asymmetry),
                          dp_volume.draine_phase(x[0], x[1], x[2]),
                          dp_volume.cloud_phase(x[0], False),
                          dp_volume.cloud_phase(x[0], True)])

    sets.append(golden_set(
        "phase", "test_phase", rows, evaluate(phases, rows, 6), tol(1e-6, 2e-3),
        source="lib/volume_rendering_models.py",
        note="Inputs: cos_theta, g, alpha. Outputs: Rayleigh, HG(g), Klein-Nishina(3000), "
             "Draine(g, alpha), cloud, cloud with reduced peak. The relative tolerance covers "
             "f32 cancellation in 1 + g^2 - 2 g cos near the forward peak.",
    ))

    @ti.func
    def sample_hg(x):
        return dp_volume.sample_hg_phase(ti.Vector([x[3], x[4], x[5]]), dp_volume.mie_g)

    @ti.func
    def sample_kn(x):
        return dp_volume.sample_klein_nishina_phase(ti.Vector([x[3], x[4], x[5]]), dp_volume.mie_asymmetry)

    @ti.func
    def sample_cloud(x):
        return dp_volume.sample_cloud_phase(ti.Vector([x[3], x[4], x[5]]), False)

    @ti.func
    def sample_cloud_reduced(x):
        return dp_volume.sample_cloud_phase(ti.Vector([x[3], x[4], x[5]]), True)

    def sample_phases(u):
        views = np.vstack([[0, 1, 0], [1, 0, 0], random_units(2)])
        rows = np.array([[*uu, *v] for uu in u for v in views])
        return rows, np.hstack([
            evaluate(sample_hg, rows, 3, 2, rows[:, :2]),
            evaluate(sample_kn, rows, 3, 2, rows[:, :2]),
            evaluate(sample_cloud, rows, 3, 3, rows[:, :3]),
            evaluate(sample_cloud_reduced, rows, 3, 3, rows[:, :3]),
        ])

    source = "lib/volume_rendering_models.py sample_{hg_phase,klein_nishina_phase,cloud_phase}"
    layout = "Inputs: u xyz, view. Outputs: HG(0.8), Klein-Nishina(3000), cloud, cloud with reduced peak. "
    rows, out = sample_phases(np.vstack([[[0.5, 0.5, 0.5], [0.999, 0.25, 0.75], [0.2, 0.999, 0.01]],
                                         rng.random((12, 3)) * 0.98 + 0.01]))
    sets.append(golden_set(
        "sample_phase", "test_sample_phase", rows, out, tol(2e-3, 0.0), source=source,
        note=layout + "The Draine inversion loses about 1e-3 in f32.",
    ))
    rows, out = sample_phases(np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.9999, 0.0, 0.9999]]))
    sets.append(golden_set(
        "sample_phase_edges", "test_sample_phase", rows, out, tol(2e-2, 0.0), source=source,
        note=layout + "u at 0 or 1 sends cos_theta to +-1, where sin_theta = sqrt(1 - cos^2) "
             "amplifies f32 rounding of the inversions to ~1e-2.",
    ))

    sets += bruneton_sets()
    return sets


def bruneton_sets():
    sets = []
    horizon = lambda r: -math.sqrt(1.0 - (PLANET_R / r) ** 2)

    def forward_rows(radii):
        rows = []
        for r in radii:
            mus = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
            if r > PLANET_R:
                mus += [horizon(r) - 0.01, horizon(r) + 0.01]
            for mu in mus:
                for mu_s in (-1.0, -0.5, -0.2001, -0.2, -0.19, 0.0, 0.3, 1.0):
                    rows.append([r, mu, mu_s, [-1.0, 0.0, 0.7, 1.0][len(rows) % 4]])
        return np.array(rows)

    @ti.func
    def forward(x):
        uvwz = dp_bruneton.GetScatteringTextureUvwzFromRMuMuSNu(
            x[0], x[1], x[2], x[3], dp_bruneton.RayIntersectsGround(x[0], x[1]))
        return ti.Vector([uvwz[0], uvwz[1], uvwz[2], uvwz[3],
                          ti.select(dp_bruneton.RayIntersectsGround(x[0], x[1]), 1.0, 0.0)])

    rows = forward_rows([PLANET_R + 1e3, PLANET_R + 1e4, PLANET_R + 5e4, ATMOS_R - 1.0, ATMOS_R])
    sets.append(golden_set(
        "scattering_uvwz", "test_scattering_uvwz", rows, evaluate(forward, rows, 5),
        tol(2e-4, 0.0), source="lib/bruneton_mappings.py GetScatteringTextureUvwzFromRMuMuSNu",
        note="Inputs: r, mu, mu_s, nu. Outputs: u_nu, u_mu_s, u_mu, u_r, intersects_ground. "
             "mu_s below -0.2 maps to the domain edge.",
    ))
    rows = forward_rows([PLANET_R, PLANET_R + 1.0, PLANET_R + 100.0])
    sets.append(golden_set(
        "scattering_uvwz_near_ground", "test_scattering_uvwz", rows, evaluate(forward, rows, 5),
        tol(2e-2, 0.0), source="lib/bruneton_mappings.py GetScatteringTextureUvwzFromRMuMuSNu",
        note="r within 100 m of the ground: r*r - R*R cancels in f32, so u_r and the "
             "ground half of u_mu carry ~1e-2 of rounding noise in the original too.",
    ))

    edge = np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
                     [0.5, 0.5, 0.4999, 0.5], [0.5, 0.5, 0.5001, 0.5], [0.25, 0.9, 0.75, 0.01],
                     [0.75, 0.1, 0.25, 0.99]])
    rows = np.vstack([edge, rng.random((40, 4))])

    @ti.func
    def inverse(x):
        r, mu, mu_s, nu, ground = dp_bruneton.GetRMuMuSNuFromScatteringTextureUvwz(x)
        return ti.Vector([r, mu, mu_s, nu, ti.select(ground, 1.0, 0.0)])

    sets.append(golden_set(
        "scattering_params_from_uvwz", "test_scattering_params_from_uvwz", rows,
        evaluate(inverse, rows, 5), tol(2e-3, 1e-6),
        source="lib/bruneton_mappings.py GetRMuMuSNuFromScatteringTextureUvwz",
        note="Outputs: r, mu, mu_s, nu, intersects_ground. The absolute tolerance covers "
             "f32 cancellation in the cosines near the horizon; r is relative.",
    ))

    rows = []
    for pos_r in (PLANET_R + 10.0, PLANET_R + 2000.0, PLANET_R + 60e3, ATMOS_R):
        for pos, d, s in zip(random_units(5) * pos_r, random_units(5), random_units(5)):
            rows.append([*pos, *d, *s])
    rows = np.array(rows)

    @ti.func
    def from_ray(x):
        p = ti.Vector([x[0], x[1], x[2]])
        d = ti.Vector([x[3], x[4], x[5]])
        r, mu, mu_s, nu = dp_bruneton.RayParamsToBruneton(p, d, ti.Vector([x[6], x[7], x[8]]))
        return ti.Vector([r, mu, mu_s, nu, ti.select(dp_bruneton.RayIntersectsGround(r, mu), 1.0, 0.0)])

    sets.append(golden_set(
        "scattering_params_from_ray", "test_scattering_params_from_ray", rows,
        evaluate(from_ray, rows, 5), tol(1e-5, 1e-6),
        source="lib/bruneton_mappings.py RayParamsToBruneton, RayIntersectsGround",
    ))

    rows = np.array([[r, mu, mu_s, nu] for r in (PLANET_R, PLANET_R + 1e4, ATMOS_R)
                     for mu in (-1.0, -0.3, 0.0, 0.6, 1.0) for mu_s, nu in ((-0.2, 0.1), (0.5, 0.9), (1.0, 1.0))])

    @ti.func
    def canonical(x):
        pos, d, s = dp_bruneton.BrunetonToRayParams(x[0], x[1], x[2], x[3])
        return ti.Vector([pos[0], pos[1], pos[2], d[0], d[1], d[2], s[0], s[1], s[2]])

    sets.append(golden_set(
        "canonical_ray", "test_canonical_ray", rows, evaluate(canonical, rows, 9),
        tol(1e-5, 1e-6), source="lib/bruneton_mappings.py BrunetonToRayParams",
    ))

    texel_centres = (np.arange(64) + 0.5) / 64.0
    corners = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    rows = np.vstack([corners, [[u, v] for u in texel_centres[::7] for v in texel_centres[::9]]])

    @ti.func
    def mu_r(x):
        mu, h = rm_lut.uv_to_mu_r(x)
        return ti.Vector([mu, h])

    out = evaluate(mu_r, rows, 2).astype(np.float64)
    out[:, 1] += PLANET_R
    sets.append(golden_set(
        "transmittance_mu_r", "test_transmittance_mu_r", rows, out, tol(2e-4, 1e-6),
        source="ray-marching-approimation lib/lut_sampling.py uv_to_mu_r (64 x 64 texels)",
        note="Outputs: mu, r (the original returns r - planet radius). The bake evaluates "
             "texel centres (fix 8); the corner rows are the original's x / RES grid.",
    ))

    rows = np.array([[mu, r] for r in (PLANET_R, PLANET_R + 1.0, PLANET_R + 2e3, PLANET_R + 6e4, ATMOS_R)
                     for mu in (-1.0, -0.5, horizon(r + 1e-3) + 0.01 if r > PLANET_R else -0.01, 0.0, 0.3, 1.0)])

    @ti.func
    def uv(x):
        return rm_lut.mu_r_to_uv(x[0], x[1])

    sets.append(golden_set(
        "transmittance_uv", "test_transmittance_uv", rows, evaluate(uv, rows, 2), tol(2e-4, 0.0),
        source="ray-marching-approimation lib/lut_sampling.py mu_r_to_uv (64 x 64 texels)",
        note="Inputs: mu, r.",
    ))
    return sets


# earth -----------------------------------------------------------------------


def earth_sets():
    sets = []
    rows = np.vstack([np.eye(3), -np.eye(3), unit([[1, 1, 0], [0, 0.999999, 0.001], [-1, 0.2, -1e-6]]),
                      random_units(12)])

    @ti.func
    def uv(x):
        return dp_math.sphere_UV_map(x)

    sets.append(golden_set(
        "sphere_uv", "test_sphere_uv", rows, evaluate(uv, rows, 2), tol(2e-6, 0.0),
        source="lib/math_utils.py sphere_UV_map",
    ))

    colours = np.vstack([[[0.2, 0.5, 0.1], [0.8, 0.6, 0.4], [0.05, 0.1, 0.3], [0.5, 0.5, 0.5],
                          [1.0, 1.0, 1.0], [0.3, 0.32, 0.29]], rng.random((10, 3)) * 0.9 + 0.02])
    rows = np.array([[*c, o] for c in colours for o in (0.0, 0.5, 1.0)])
    ALBEDO_MAP.from_numpy(np.resize(np.hstack([rows[:, :3], np.ones((len(rows), 1))]), (MAX_DRAWS, 4)).astype(np.float32))
    OCEAN_MAP.from_numpy(np.resize(np.repeat(rows[:, 3:4], 4, axis=1), (MAX_DRAWS, 4)).astype(np.float32))

    @ti.func
    def grade(x):
        albedo, _, _, _ = main_pt.get_land_material(ALBEDO_MAP, OCEAN_MAP, ZERO_MAP, ZERO_MAP, ti.Vector([0.0, 1.0, 0.0]))
        return albedo

    sets.append(golden_set(
        "grade_albedo", "test_grade_albedo", rows, evaluate(grade, rows, 3), tol(1e-5, 1e-5),
        source="main pathtracer.py get_land_material",
        note="Inputs: albedo map rgb, ocean mask. Black texels are excluded: the original "
             "divides 0 / 0 there.",
    ))

    rows = []
    n = np.array([0.0, 1.0, 0.0])
    for albedo, ocean, bathy in ((0.3, 0.0, 0.0), (0.05, 1.0, 0.2), (0.1, 1.0, 0.9), (0.2, 0.8, 0.5)):
        for v, l in ((unit([0, 1, 0]), unit([0.3, 1, 0])), (unit([1, 0.2, 0]), unit([-1, 0.2, 0])),
                     (unit([0.5, 0.5, 0.1]), unit([-0.5, 0.5, -0.1])), (unit([0.2, 1, 0.3]), unit([0.9, 0.05, 0])),
                     (unit([0, 1, 0]), unit([0, 1, 0])), (unit([1, 0.01, 0]), unit([-1, 0.01, 0]))):
            rows.append([albedo, ocean, bathy, *v, *n, *l])
        for v, nn, l in zip(random_units(4), random_units(4), random_units(4)):
            v, l = v * np.sign(v @ nn), l * np.sign(l @ nn)
            rows.append([albedo, ocean, bathy, *v, *nn, *l])
    rows = np.array(rows)

    @ti.func
    def brdf(x):
        value, n_dot_l = main_surface.earth_brdf(
            x[0], x[1], x[2], ti.Vector([x[3], x[4], x[5]]), ti.Vector([x[6], x[7], x[8]]),
            ti.Vector([x[9], x[10], x[11]]))
        return ti.Vector([value, n_dot_l])

    sets.append(golden_set(
        "earth_brdf", "test_earth_brdf", rows, evaluate(brdf, rows, 2), tol(1e-6, 1e-3),
        source="main lib/surface_rendering_models.py earth_brdf",
        note="Inputs: albedo, ocean, bathymetry, v, n, l. Outputs: brdf, n_dot_l. The "
             "relative tolerance covers the grazing Beckmann and Smith terms.",
    ))

    base = PLANET_R + 4000.0
    # No height sits on a band edge 0.2 - 0.2c or 0.2 + 0.8c: there the inside
    # test turns on the last bit of a division the GPU need not round correctly.
    rows = np.array([[base + h * 6000.0, c, 0.029] for h in (-0.01, 0.0, 0.01, 0.1, 0.185, 0.21, 0.5, 0.79, 0.81, 1.0, 1.01)
                     for c in (0.0, 0.05, 0.3, 0.6, 1.0)])
    clouds = ti.Vector.field(4, ti.f32, shape=MAX_DRAWS)
    clouds.from_numpy(np.resize(np.repeat(rows[:, 1:2], 4, axis=1), (MAX_DRAWS, 4)).astype(np.float32))

    @ti.func
    def density(x):
        return ti.Vector([main_pt.get_clouds_density(clouds, ti.Vector([0.0, x[0], 0.0]))])

    sets.append(golden_set(
        "cloud_density", "test_cloud_density", rows, evaluate(density, rows, 1), tol(1e-9, 1e-6),
        source="main pathtracer.py get_clouds_density",
        note="Inputs: r, coverage, max density (0.029 on main).",
    ))
    return sets


# cube ------------------------------------------------------------------------


def cube_sets():
    sets = []
    # Face centres, near-edge and near-corner directions (off the exact ties,
    # where the face choice is a convention: the first of x, y, z wins), the
    # Alps, and random directions, some not unit length.
    near_edges = [[1, 0.999, 0.2], [-0.3, 1, -0.999], [0.999, -0.999, 1], [-1, 0.5, 0.9999], [0.2, -1, 0.99]]
    alps = cube.lonlat_to_direction(np.array([5.0, 7.5, 12.0]), np.array([44.0, 46.0, 48.0]))
    rows = np.vstack([np.eye(3), -np.eye(3), unit(near_edges), alps, random_units(16),
                      random_units(4) * np.array([[3.7], [0.01], [250.0], [1.0]])])
    face, u, v = cube.direction_to_face_uv(rows)
    sets.append(golden_set(
        "cube_face_uv", "test_cube_face_uv", rows, np.stack([face, u, v], axis=1), tol(3e-6, 0.0),
        reference="port", source="python/tiles/cube.py direction_to_face_uv",
        note="Inputs: a direction, not necessarily unit. Outputs: face, u, v.",
    ))

    corners = [[0, 0], [1, 1], [0, 1], [1, 0], [0.5, 0.5], [0.25, 0.75]]
    borders = [[-2 / 192, 0.5], [1 + 2 / 192, 0.3], [0.7, -2 / 192], [1 + 2 / 192, 1 + 2 / 192]]
    rows = np.array([[f, *uv] for f in range(6) for uv in corners + borders]
                    + [[f, *uv] for f, uv in zip(rng.integers(0, 6, 12), rng.random((12, 2)))])
    sets.append(golden_set(
        "cube_direction", "test_cube_direction", rows,
        cube.face_uv_to_direction(rows[:, 0].astype(int), rows[:, 1], rows[:, 2]), tol(3e-6, 0.0),
        reference="port", source="python/tiles/cube.py face_uv_to_direction",
        note="Inputs: face, u, v; u and v run 2/192 past the face edges, where tile borders sample. "
             "Outputs: the unit direction.",
    ))
    return sets


# post ------------------------------------------------------------------------


def post_sets():
    sets = []
    colours = np.vstack([[[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0],
                          [100.0, 50.0, 20.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                          [1e-4, 2e-4, 5e-5], [0.5, -0.01, 0.2], [3.0, 2.0, 8.0]],
                         rng.random((9, 3)) * np.array([[1.0], [4.0], [16.0]]).repeat(3, axis=0)[:9]])
    # Its own generator keeps the global draws, and so the later sets, unchanged.
    drt_rng = np.random.default_rng(20261019)
    intensities = 2.0 ** np.arange(-12.0, 13.0, 2.0)
    primaries = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
    out_of_gamut = np.array([[1.0, -0.2, -0.1], [-0.3, 1.0, 0.2], [-0.05, -0.1, 1.0], [-1.0, -1.0, -1.0],
                             [2.0, -3.0, 0.5], [0.0, 0.0, -1e-3], [-5.0, 0.2, 0.1]])
    colours = np.vstack([colours, intensities[:, None].repeat(3, axis=1),
                         (intensities[::3, None, None] * primaries).reshape(-1, 3), out_of_gamut,
                         2.0 ** drt_rng.uniform(-10.0, 8.0, (60, 1)) * drt_rng.dirichlet((0.5, 0.5, 0.5), 60)])
    presets = list(opendrt.TONESCALE_PRESETS)
    default_controls = (1.5, 0.0, presets.index("dagrinchi_tone"))
    controls = [default_controls, (1.0, 0.0, default_controls[2]), (0.0, 0.0, default_controls[2]),
                (3.6, 0.0, default_controls[2]), (1.5, -0.75, default_controls[2]), (1.5, 1.0, default_controls[2])]
    controls += [(1.5, 0.0, index) for index in range(len(presets)) if index != default_controls[2]]
    rows = np.vstack([np.hstack([colours, np.tile(c, (len(colours), 1))]) for c in controls])
    outputs = np.vstack([opendrt.opendrt(colours, saturation, contrast, presets[int(preset)])
                         for saturation, contrast, preset in controls])
    sets.append(golden_set(
        "opendrt", "test_opendrt", rows, outputs, tol(1e-5, 1e-5), reference="upstream",
        source="python/opendrt.py, OpenDRT v1.0.0b43 DCTL at jedypod/open-display-transform c8d698f",
        note="Inputs: rgb, saturation, contrast, tonescale preset index (TONESCALE_PRESETS order).",
    ))

    rows = np.array([[0.0, 0.5, 1.0, 0, 16], [-0.1, 1.2, 0.9995, 3, 16], [0.25, 0.75, 0.999, 15, 16],
                     [0.1, 0.2, 0.3, 0, 1]], dtype=np.float64)
    offset = 0.5 / 1024.0
    u = np.minimum(np.clip(rows[:, :3], 0.0, 1.0) + offset, 1.0 - offset)
    v = (rows[:, 3:4] + 0.5) / rows[:, 4:5]
    sets.append(golden_set(
        "crf_coords", "test_crf_coords", rows, np.hstack([u, v]), tol(1e-7, 0.0), reference="port",
        source="renderer.py camera_response",
        note="Inputs: display rgb, CRF index, CRF count. Outputs: u per channel, v.",
    ))

    rows = np.array([[0, 0, 1920, 1080], [960, 540, 1920, 1080], [1919, 1079, 1920, 1080],
                     [100, 900, 1920, 1080], [255, 0, 512, 512]], dtype=np.float64)
    uv = rows[:, :2] / rows[:, 2:]
    vignette = 1.0 - 0.9 * np.maximum(np.linalg.norm(uv - 0.5, axis=1), 0.0)
    sets.append(golden_set(
        "vignette", "test_vignette", rows, vignette[:, None], tol(1e-6, 0.0), reference="port",
        source="renderer.py _render_to_image", note="Inputs: pixel x, y, width, height.",
    ))

    rows = np.array([[0.0, 0.001, 0.0031308], [0.0032, 0.18, 0.5], [1.0, 2.0, 1e-6]])

    @ti.func
    def oetf(x):
        return dp_colour.srgb_transfer(x)

    sets.append(golden_set(
        "srgb_oetf", "test_srgb_oetf", rows, evaluate(oetf, rows, 3), tol(1e-6, 1e-5),
        source="lib/colour.py srgb_transfer",
    ))
    return sets


# neural ----------------------------------------------------------------------

# One small network per encoding of the neural mode, exercising every head the
# shader specializes: staged unrolled (raw, Fourier), constant (hash grid,
# K-Planes); the legacy network covers the rolled 64-wide layers.
NEURAL_SPECS = {
    "raw": {"encoding": {"kind": "raw"}, "head": {"width": 16, "hidden": 2, "activation": "gelu"}},
    "fourier": {"encoding": {"kind": "fourier", "octaves": 2}, "head": {"width": 32, "hidden": 1, "activation": "relu"}},
    # 2^12 entries: the coarsest level (5 x 9 x 9 x 5 vertices) stays dense, the others hash.
    "hash_grid": {"encoding": {"kind": "hash", "levels": 4, "log2_table": 12, "cells_max": [8, 16, 16, 8]},
                  "head": {"width": 16, "hidden": 1, "activation": "relu"}},
    "k_planes": {"encoding": {"kind": "kplanes", "features": 4, "base": 4, "init_low": 0.5, "init_high": 1.5},
                 "head": {"width": 16, "hidden": 1, "activation": "relu"}},
}
NEURAL_DIR = GOLDENS / "neural"


def neural_inputs(seed):
    edge = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5],
                     [0.3, 0.6, 0.4999, 0.9], [0.3, 0.6, 0.5001, 0.9]])
    return np.vstack([edge, np.random.default_rng(seed).random((251, 4))]).astype(np.float32)


def neural_sets(checkpoint):
    """The trained legacy network through torch in float64, small random networks of
    every encoding through export_weights.reference, and the shader's erf.

    The legacy set binds the committed version 1 file, checked against the
    checkpoint here. The other networks are written to tests/goldens/neural/.
    """
    from study import encodings

    network, model = export_weights.mlp_checkpoint_network(checkpoint, export_weights.TAICHI_OUTPUT_SCALE)
    legacy = export_weights.decode(EMLP_LUT_PATH.read_bytes())
    for (weight, bias, activation), (legacy_weight, legacy_bias, legacy_activation) in zip(network["layers"],
                                                                                           legacy["layers"]):
        assert np.array_equal(weight, legacy_weight) and np.array_equal(bias, legacy_bias)
        assert activation == legacy_activation
    rows = neural_inputs(120)
    outputs = export_weights.torch_radiance(model, rows.astype(np.float64), export_weights.TAICHI_OUTPUT_SCALE)
    scale = export_weights.TAICHI_OUTPUT_SCALE
    sets = [golden_set(
        "legacy_mlp", "test_network", rows, outputs, tol(2e-6 * scale, 1e-4), reference="torch",
        lut={"path": str(EMLP_LUT_PATH.relative_to(ROOT)), "dtype": "emlp"},
        source=f"python/model.py MLP, {checkpoint.name}",
        note="Inputs: uvwz. Outputs: linear Rec.709 on the renderer's scale (the version 1 file's legacy "
             "output scale). The reference runs in float64; f32 sums of the last layer cancel to ~1e-6 "
             "before the scale.",
    )]

    NEURAL_DIR.mkdir(exist_ok=True)
    for index, (name, spec) in enumerate(NEURAL_SPECS.items()):
        torch.manual_seed(index)
        spec = dict(spec, output="identity", mu_rotation=True)
        model = encodings.build(spec)
        if name == "hash_grid":
            # Instant-NGP's 1e-4 initialization would leave the encoding invisible.
            torch.nn.init.uniform_(model.encoding.table, -1.0, 1.0)
        path = NEURAL_DIR / f"{name}.emlp"
        export_weights.export(export_weights.study_model_network(model, spec), model, path)
        network = export_weights.decode(path.read_bytes())
        rows = neural_inputs(index)
        outputs = export_weights.reference(network, rows, half_tables=True)
        tolerance = tol(1e-6, 1e-4)
        note = "Inputs: uvwz. Outputs: linear Rec.709, tables at the half precision the GPU stores."
        if name == "k_planes":
            # The texture unit's fixed-point weights move outputs by up to about 1 % of
            # their RMS (docs/research/20261019-neural-inference-cost.md).
            tolerance = tol(0.03 * float(np.sqrt(np.mean(outputs**2))), 0.0)
            note += " Planes interpolate exactly here and in the texture unit's 8-bit weights on the GPU."
        sets.append(golden_set(
            name, "test_network", rows, outputs, tolerance, reference="port",
            lut={"path": str(path.relative_to(ROOT)), "dtype": "emlp"},
            source=f"python/export_weights.py reference, NEURAL_SPECS['{name}'], torch seed {index}", note=note,
        ))

    x = np.concatenate([np.linspace(-5.0, 5.0, 1001), [-1.0, 1.0, np.nextafter(np.float32(1), np.float32(0)),
                                                       -4.0, 4.0, 1e-6, -1e-30, 0.0]]).astype(np.float32)
    erf = np.vectorize(math.erf)(x.astype(np.float64))
    sets.append(golden_set(
        "erf", "test_erf", x[:, None], erf[:, None], tol(2e-7, 0.0), reference="port",
        note="The shader's erf approximation against math.erf; GELU is exact through it.",
    ))
    return sets


MODULES = ("common", "spectral", "atmosphere", "earth", "cube", "post", "neural")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlp-checkpoint", type=Path, default=ROOT / "model_weights" / "mlp_e120.pth",
                        help="the network the original viewer loaded (highest epoch)")
    parser.add_argument("modules", nargs="*", help=f"the modules to write, of {', '.join(MODULES)} (default: all)")
    args = parser.parse_args()
    if unknown := set(args.modules) - set(MODULES):
        parser.error(f"unknown modules: {', '.join(sorted(unknown))}")
    GOLDENS.mkdir(parents=True, exist_ok=True)
    sets = {
        "common": common_sets,
        "spectral": spectral_sets,
        "atmosphere": atmosphere_sets,
        "earth": earth_sets,
        "cube": cube_sets,
        "post": post_sets,
        "neural": lambda: neural_sets(args.mlp_checkpoint),
    }
    for module in args.modules or MODULES:
        write(module, f"shaders/tests/test_{module}.slang", sets[module]())


if __name__ == "__main__":
    main()
