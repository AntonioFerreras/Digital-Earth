"""Writes LUT/ozone_profile.dat: the global, annual mean ozone number density.

Source: McPeters and Labow 2011 ozone profile climatology (Aura MLS 2004-2010
merged with ozonesondes 1988-2010), monthly means in 18 latitude bands of 10
degrees, DU per 1 km layer of pressure altitude Z* = 16 log10(1013 hPa / p).

  LUT/O3CLIM_ML_SEASONAL_DU.txt, retrieved 2026-10-19 from
  https://acdisc.gesdisc.eosdis.nasa.gov/test/data/Ozone_Climatology/O3CLIM_ML_SEASONAL.1/
  (the /data/ path needs an Earthdata login; the file matches the production
  granule's CRC32 2067042653, 117270 bytes). DOI 10.5067/7A8GLBK5GM2M.
  McPeters, R. D. and Labow, G. J.: Climatology 2011: An MLS and sonde derived
  ozone climatology for satellite retrieval algorithms, J. Geophys. Res., 117,
  D10303, https://doi.org/10.1029/2011JD017006, 2012.

Method:
  1. Average the 12 months and weight the bands by their exact area,
     sin(latitude north edge) - sin(latitude south edge).
  2. Place the Z* layer edges at geometric altitude through the pressure of the
     US Standard Atmosphere 1976, the atmosphere the Rayleigh profile is fitted to.
  3. Above Z* = 64 km, an exponential fitted to the layers from Z* = 55 to 64 km
     (scale height about 4 km) stands in for the data up to ATMOSPHERE_HEIGHT.
     It replaces the 64-65 km layer too, a fifth of the layer below it and out of
     line with the decay; the source's 66-90 km residual stays unused.
  4. Sample i is the mean number density over [z_i - 0.25 km, z_i + 0.25 km]
     (clipped at 0) of that piecewise-constant profile, at z_i = 0.5 i km, so the
     linear interpolation the shader applies keeps the column.

Output: 221 f32, number density in molecules per m^3 at 0, 0.5, ..., 110 km.

Run: pixi run -e py python python/lut/ozone_profile_generator.py
"""

import re
from pathlib import Path

import numpy as np

LUT_DIR = Path(__file__).resolve().parents[2] / "LUT"
SOURCE = LUT_DIR / "O3CLIM_ML_SEASONAL_DU.txt"
OUTPUT = LUT_DIR / "ozone_profile.dat"

DOBSON_UNIT = 2.6867e20  # molecules per m^2 of column
ALTITUDE_STEP_M = 500.0
ALTITUDE_MAX_M = 110e3  # ATMOSPHERE_HEIGHT in shaders/common/constants.slang
SAMPLE_COUNT = round(ALTITUDE_MAX_M / ALTITUDE_STEP_M) + 1

BAND_COUNT = 18
MONTH_COUNT = 12
# Rows "00-01" ... "64-65" then "66-90", the column above the table.
LAYER_COUNT = 65
TAIL_FIT_LAYERS = slice(55, 64)

# US Standard Atmosphere 1976 below 84.852 km geopotential: layer base
# geopotential altitude (m) and temperature lapse rate (K/m).
US76_BASE_M = np.array([0.0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 84.852e3])
US76_LAPSE = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3])
US76_SEA_LEVEL_TEMPERATURE = 288.15
US76_SEA_LEVEL_PRESSURE = 101325.0
US76_G0_M_OVER_R = 9.80665 * 0.0289644 / 8.31432  # g0 M / R*, K per m
US76_EARTH_RADIUS_M = 6356766.0


def read_climatology():
    """DU per Z* layer, shape (band, layer, month), bands south to north."""
    bands = []
    for line in SOURCE.read_text().splitlines():
        if "CLIMATOLOGY" in line:
            bands.append([])
        elif re.match(r"\s*\d\d-\d\d\s", line):
            bands[-1].append([float(x) for x in line.split()[1:]])
    du = np.array(bands)
    assert du.shape == (BAND_COUNT, LAYER_COUNT + 1, MONTH_COUNT)
    assert np.all(du >= 0.0)
    return du


def us76_base_states():
    """Temperature (K) and pressure (Pa) at each US76 layer base."""
    temperature = [US76_SEA_LEVEL_TEMPERATURE]
    pressure = [US76_SEA_LEVEL_PRESSURE]
    for layer, lapse in enumerate(US76_LAPSE):
        thickness = US76_BASE_M[layer + 1] - US76_BASE_M[layer]
        top = temperature[-1] + lapse * thickness
        if lapse == 0.0:
            ratio = np.exp(-US76_G0_M_OVER_R * thickness / temperature[-1])
        else:
            ratio = (temperature[-1] / top) ** (US76_G0_M_OVER_R / lapse)
        temperature.append(top)
        pressure.append(pressure[-1] * ratio)
    return np.array(temperature), np.array(pressure)


def us76_geometric_altitude(pressure_pa):
    """Inverts the US76 pressure profile: geometric altitude (m) at each pressure."""
    temperature, base_pressure = us76_base_states()
    assert np.all(pressure_pa <= base_pressure[0] * 1.001)
    assert np.all(pressure_pa > base_pressure[-1])
    layer = np.clip(np.searchsorted(-base_pressure, -pressure_pa, side="right") - 1, 0, len(US76_LAPSE) - 1)
    lapse = US76_LAPSE[layer]
    ratio = pressure_pa / base_pressure[layer]
    isothermal = -temperature[layer] * np.log(ratio) / US76_G0_M_OVER_R
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = temperature[layer] / lapse * (ratio ** (-lapse / US76_G0_M_OVER_R) - 1.0)
    geopotential = US76_BASE_M[layer] + np.where(lapse == 0.0, isothermal, gradient)
    return US76_EARTH_RADIUS_M * geopotential / (US76_EARTH_RADIUS_M - geopotential)


def build_profile():
    du = read_climatology()
    edges = np.radians(np.arange(-90, 91, 10))
    band_weight = np.diff(np.sin(edges))
    band_weight /= band_weight.sum()
    layer_du = np.einsum("b,blm->l", band_weight, du) / MONTH_COUNT

    z_star_km = np.arange(LAYER_COUNT + 1, dtype=np.float64)
    edge_m = np.maximum(us76_geometric_altitude(1013.0e2 * 10.0 ** (-z_star_km / 16.0)), 0.0)
    density = layer_du[:LAYER_COUNT] * DOBSON_UNIT / np.diff(edge_m)
    midpoint_m = 0.5 * (edge_m[:-1] + edge_m[1:])

    fit = np.polyfit(midpoint_m[TAIL_FIT_LAYERS], np.log(density[TAIL_FIT_LAYERS]), 1)
    scale_height_m = -1.0 / fit[0]
    tail_base_m = edge_m[TAIL_FIT_LAYERS.stop]
    tail_base_density = np.exp(np.polyval(fit, tail_base_m))

    def column(z_m):
        """Molecules per m^2 between the ground and `z_m`."""
        below = np.concatenate([[0.0], np.cumsum(density[: TAIL_FIT_LAYERS.stop] * np.diff(edge_m[: TAIL_FIT_LAYERS.stop + 1]))])
        inside = np.interp(np.minimum(z_m, tail_base_m), edge_m[: TAIL_FIT_LAYERS.stop + 1], below)
        above = tail_base_density * scale_height_m * -np.expm1(-np.maximum(z_m - tail_base_m, 0.0) / scale_height_m)
        return inside + above

    altitude_m = np.arange(SAMPLE_COUNT) * ALTITUDE_STEP_M
    lower = np.maximum(altitude_m - 0.5 * ALTITUDE_STEP_M, 0.0)
    upper = altitude_m + 0.5 * ALTITUDE_STEP_M
    profile = (column(upper) - column(lower)) / (upper - lower)

    tail_column_du = column(np.inf) - column(edge_m[LAYER_COUNT])
    print(f"global mean column {layer_du.sum():.2f} DU in the source, "
          f"{column(ALTITUDE_MAX_M) / DOBSON_UNIT:.2f} DU in the profile")
    print(f"exponential tail: scale height {scale_height_m / 1e3:.2f} km, column above Z* 65 km "
          f"{tail_column_du / DOBSON_UNIT:.4f} DU (source 66-90 km: {layer_du[LAYER_COUNT]:.4f} DU)")
    return altitude_m, profile


def main():
    altitude_m, profile = build_profile()
    table = profile.astype(np.float32)
    assert table.shape == (SAMPLE_COUNT,)
    assert np.all(np.isfinite(table))
    assert np.all(table >= 0.0)
    peak = int(np.argmax(table))
    trapezoid_du = np.sum(0.5 * (table[1:].astype(np.float64) + table[:-1]) * ALTITUDE_STEP_M) / DOBSON_UNIT
    print(f"peak {table[peak]:.4e} m^-3 at {altitude_m[peak] / 1e3:.1f} km; "
          f"table column (trapezoid) {trapezoid_du:.2f} DU")
    table.tofile(OUTPUT)


if __name__ == "__main__":
    main()
