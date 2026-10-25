"""Measures the sky's brightness against the sun's height and prints the auto exposure
table of src/render/Post.zig.

Sky-only views, path traced from far beyond the cloud bench's cloud, look out at four
azimuths; their mean luminance above the horizon is the sky's, Y, for each cosine of
the sun's zenith angle. The table's exposure E holds the displayed sky, log2(Y) + E,
at its brightness for PLATEAU from overhead down to SUNSET, then lets it fall by a
smoothstep to its brightness for TWILIGHT, the lowest sun allowed. The sky never
brightens as the sun sinks, and both brightness and exposure are smooth.

Run from the repository root after changing the atmosphere or the sky, and paste the
output into Post.auto_exposure_table:
  pixi run -e py python python/validation/auto_exposure.py
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clouds.score import read_pfm  # noqa: E402

# (cos zenith, stops): the anchors the curve was set by.
PLATEAU = (0.3, 3.0)
SUNSET = 0.002
TWILIGHT = (-0.07, 10.0)
COS_ZENITHS = sorted(set(np.round(np.concatenate([
    np.linspace(TWILIGHT[0], 0.0, 8), np.linspace(0.0, 0.3, 16), np.linspace(0.3, 1.0, 15)]), 4)))
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
SIZE = 64
SPP = 4
FOV_DEGREES = 90.0
PITCH = math.radians(20.0)
# Far enough beyond the cloud, which sits over the pole, that none of it is in view.
DISTANCE_M = 60e3
CENTRE = np.array([0.0, 6371e3 + 2400.0, 0.0])
REC709_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def sky_luminance(scratch: Path) -> list[float]:
    jobs, views = [], []
    for index, cos_zenith in enumerate(COS_ZENITHS):
        for azimuth in AZIMUTHS:
            out = np.array([math.sin(math.radians(azimuth)), 0.0, math.cos(math.radians(azimuth))])
            forward = math.cos(PITCH) * out + np.array([0.0, math.sin(PITCH), 0.0])
            position = CENTRE + DISTANCE_M * out
            config = scratch / f"sky_{index:02d}_{azimuth:03.0f}.zon"
            config.write_text(
                ".{ .camera = .{ .position = .{ %r, %r, %r }, .forward = .{ %r, %r, %r }, .up = .{ 0, 1, 0 }, "
                ".vertical_fov_degrees = %g, .aspect_scale = 1 }, .bench_sun = .{ .cos_zenith = %r, .azimuth_degrees = 65 }, "
                ".mode = .cloud_bench, .preset = .symmetric }\n" % (*position, *forward, FOV_DEGREES, float(cos_zenith)))
            jobs.append((config, scratch / config.stem, "reference", SPP))
            views.append((index, forward))
    validate.render(jobs, SIZE, None, scratch / "jobs.zon")
    span = math.tan(math.radians(FOV_DEGREES) / 2) * ((np.arange(SIZE) + 0.5) / SIZE * 2 - 1)
    x, y = np.meshgrid(span, -span)
    sums = np.zeros(len(COS_ZENITHS))
    for (index, forward), (_, out, _, _) in zip(views, jobs):
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        above = (forward[1] + x * right[1] + y * up[1]) > 0
        sums[index] += (read_pfm(out.with_suffix(".pfm")) @ REC709_LUMINANCE)[above].mean()
    return list(sums / len(AZIMUTHS))


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        log_sky = np.log2(sky_luminance(Path(scratch)))
    mu = np.array(COS_ZENITHS)
    plateau = np.interp(PLATEAU[0], mu, log_sky) + PLATEAU[1]
    twilight = np.interp(TWILIGHT[0], mu, log_sky) + TWILIGHT[1]
    t = np.clip((mu - TWILIGHT[0]) / (SUNSET - TWILIGHT[0]), 0.0, 1.0)
    shown = twilight + (plateau - twilight) * t * t * (3 - 2 * t)
    for cos_zenith, stops in zip(mu, shown - log_sky):
        print(f"    .{{ {cos_zenith:g}, {stops:.4f} }},")


if __name__ == "__main__":
    main()
