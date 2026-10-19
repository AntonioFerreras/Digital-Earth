"""Writes LUT/ozone_cross_section.dat: ozone absorption cross sections at one temperature.

Source: Serdyuchenko et al. 2014 and Gorshelev et al. 2014 (IUP Bremen), cm^2
per molecule at 11 temperatures from 193 K to 293 K on a 0.01 nm grid of
vacuum wavelengths, version 22.07.2013.

  LUT/serdyuchenkogorshelevversionjuly2013.zip, retrieved 2026-10-19 from
  https://www.iup.uni-bremen.de/gruppen/molspec/downloads/serdyuchenkogorshelevversionjuly2013.zip
  (linked from https://www.iup.uni-bremen.de/gruppen/molspec/databases/referencespectra/o3spectra2011/).
  Serdyuchenko, A. et al.: High spectral resolution ozone absorption
  cross-sections - Part 2: Temperature dependence, Atmos. Meas. Tech., 7,
  625-636, https://doi.org/10.5194/amt-7-625-2014, 2014.
  Gorshelev, V. et al.: Part 1: Measurements, data analysis and comparison
  with previous measurements around 293 K, Atmos. Meas. Tech., 7, 609-624,
  https://doi.org/10.5194/amt-7-609-2014, 2014.

The default, 223 K, is the measured temperature nearest the ozone layer's: the
US Standard Atmosphere 1976 is 221.6 K at the 25 km density peak of
LUT/ozone_profile.dat and 227 K averaged over its column (213 K and 233 K are
both farther).

Entry i is the mean over the 1 nm bin centred on 390 + i nm (the trapezoid
rule over [390 + i - 0.5, 390 + i + 0.5] nm), the wavelength at which the
shader's linear interpolation places it; i runs to 830 nm, 441 entries.

Run: pixi run -e py python python/lut/ozone_cross_section_generator.py [--temperature K]
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

LUT_DIR = Path(__file__).resolve().parents[2] / "LUT"
SOURCE = LUT_DIR / "serdyuchenkogorshelevversionjuly2013.zip"
SOURCE_MEMBER = "SerdyuchenkoGorshelevVersionJuly2013.dat"
OUTPUT = LUT_DIR / "ozone_cross_section.dat"

# Source columns after the wavelength.
TEMPERATURES_K = (293, 283, 273, 263, 253, 243, 233, 223, 213, 203, 193)
SOURCE_STEP_NM = 0.01
WAVELENGTH_MIN_NM = 390
WAVELENGTH_COUNT = 441
SAMPLES_PER_BIN = round(1.0 / SOURCE_STEP_NM)


def read_source():
    """Rows of (vacuum wavelength nm, cross section per temperature), written with decimal commas."""
    text = zipfile.ZipFile(SOURCE).read(SOURCE_MEMBER).decode("latin-1")
    rows = [line.replace(",", ".") for line in text.splitlines()
            if line[:1].isdigit() and len(line.split()) == 1 + len(TEMPERATURES_K)]
    table = np.loadtxt(io.StringIO("\n".join(rows)))
    assert np.allclose(np.diff(table[:, 0]), SOURCE_STEP_NM)
    return table


def bin_means(table, temperature_k):
    column = 1 + TEMPERATURES_K.index(temperature_k)
    first = int(np.argmin(np.abs(table[:, 0] - (WAVELENGTH_MIN_NM - 0.5))))
    means = np.empty(WAVELENGTH_COUNT)
    for i in range(WAVELENGTH_COUNT):
        start = first + i * SAMPLES_PER_BIN
        samples = table[start: start + SAMPLES_PER_BIN + 1]
        assert abs(samples[0, 0] - (WAVELENGTH_MIN_NM + i - 0.5)) < 1e-6
        assert abs(samples[-1, 0] - (WAVELENGTH_MIN_NM + i + 0.5)) < 1e-6
        values = samples[:, column]
        means[i] = (values.sum() - 0.5 * (values[0] + values[-1])) / SAMPLES_PER_BIN
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--temperature", type=int, choices=TEMPERATURES_K, default=223,
                        help="measurement temperature in kelvin (default: %(default)s)")
    args = parser.parse_args()
    lut = bin_means(read_source(), args.temperature).astype(np.float32)
    assert np.all(lut > 0.0)
    assert np.all(lut < 1e-19)
    lut.tofile(OUTPUT)
    print(f"{OUTPUT.name}: {args.temperature} K, peak {lut.max():.4e} cm^2 at "
          f"{WAVELENGTH_MIN_NM + int(np.argmax(lut))} nm")


if __name__ == "__main__":
    main()
