# Twilight noise: measurements and fixes

With the textured preset, a sun below the horizon made both `raymarch` and
`path_tracer` converge far more slowly than any other view. This note records
where the variance came from, the three fixes, and what they bought at equal
time.

## Method

- **View.** The reported view: camera at 35 m above the sea looking up,
  vertical field of view 104 degrees, ocean in the lower 40 % of the frame,
  sun 0.83 degrees below the horizon, exposure 7.94 stops, `textured`, Q2.
  Variants move the sun along its path to -5.7, +3, and +10 degrees.
- **Variance.** 16 independent renders of 4 samples per pixel at 640x360,
  keyed with `--seed` 4 samples apart. The per-sample variance of a pixel is 4
  times the variance of its 16 means. Two statistics per region:
  - *median relvar*: the median over pixels of variance / mean^2 of luminance Y
    (or of the RGB sum). A sample count n gives a relative error of
    sqrt(relvar / n).
  - *regional relvar*: the region's summed variance over its squared mean Y,
    which fireflies dominate.
  - Sky is the top 53 % of the frame, ground the bottom 39 %.
- **Attribution.** Each hypothesis has a temporary shader toggle that removes
  its randomness: 16 stratified wavelengths per sample sharing all other
  random numbers (spectral), the sun at the cone centre (sun cone), 256
  steps (step placement), the pixel centre (pixel jitter), one segment (ground
  bounce), the BRDF without its specular term (surface lobe). The share of a
  cause is the variance the toggle removes.
- **Time.** ms per sample at 1920x1080, 128 samples (raymarch) or 16 (path
  tracer), the builds interleaved three times. The GPU also drives a display,
  so only ratios within one run are compared.
- **Display error.** RMS error of the 8-bit PNG against a 4096-sample
  reference, which includes the tone curve and saturation.

## Attribution

Median relvar per sample, luminance unless noted, at the reported view.

| mode | region | cause | baseline | toggle on | share |
| --- | --- | --- | ---: | ---: | ---: |
| raymarch | sky | spectral (H2) | 0.739 | 0.012 | 98 % |
| raymarch | sky | sun cone (H3) | 0.739 | 0.727 | 2 % |
| raymarch | sky | step placement (H1) | 0.739 | 0.731 | 1 % |
| raymarch | sky | pixel jitter | 0.739 | 0.739 | 0 % |
| raymarch | sky | all four | 0.739 | 0.00027 | 99.96 % |
| raymarch | ground | surface lobe (H5), regional | 306 | 1.2 | 99.6 % |
| raymarch | ground | surface lobe (H5), median | 7.24 | 0.86 | 88 % |
| raymarch | ground | spectral, RGB median | 8.9 | 2.2 | 75 % |
| path_tracer | sky | spectral (H2) | 30.6 | 4.6 | 85 % |
| path_tracer | sky | spectral (H2), regional | 631 | 37 | 94 % |
| path_tracer | sky | sun cone, pixel jitter | 30.6 | 30.6 | 0 % |

- **Spectral sampling (H2)** is nearly all of the sky's variance in both modes,
  and in daylight too: the ray-marched sky has the same median relvar at +10
  degrees (0.776) as at -0.8 (0.739). Twilight only makes the remainder after
  more wavelengths larger: 16 wavelengths leave 0.003 at +10 degrees, 0.012 at
  -0.8, 0.043 at -5.7.
- **Step placement (H1)** and **the sun cone (H3)** remove 3 % and 6 % at -5.7
  degrees, less at the reported view. Most of the air above a camera at the
  sea is still sunlit at -0.8 degrees (the shadow reaches 0.7 km overhead).
- **The ocean (H5).** The twilight sea is lit almost only through its glossy
  reflection of the bright horizon: removing the bounce darkens the ground 12x,
  removing the specular term 4.6x. Cosine-weighted bounces find that lobe
  rarely and then weight it heavily, so the ground's variance is fireflies.
  In daylight the sun's direct term dominates the sea and hides this.
- **Path tracer, collisions in shadowed air (H4).** Single scattering is 14 %
  of the path-traced twilight sky and about 2 % of its variance (median relvar
  31.8 against its own mean 0.036). The variance is in multiple scattering:
  most paths collide low, in air whose sun transmittance is near 0, and return
  nothing, which also makes the twilight sky's relvar 3.3x the daylight one.

## Fixes

| fix | mode, region | median relvar | regional relvar | time | equal-time gain |
| --- | --- | --- | --- | --- | --- |
| four-wavelength bundles | raymarch, sky | 0.739 -> 0.0547 (Y), 1.24 -> 0.0433 (RGB) | 0.959 -> 0.072 | 1.47x | 9x (Y), 19x (RGB) |
| same, +10 degrees | raymarch, sky | 0.776 -> 0.0352, 1.17 -> 0.0345 | | 1.43x | 15x, 24x |
| same, -5.7 degrees | raymarch, sky | 0.954 -> 0.111, 1.41 -> 0.080 | | | |
| ocean-lobe bounces | raymarch, ground | 8.6 -> 0.53 | 175 -> 0.78 | 1.11x | 14x, 200x |
| bundles with spectral MIS, ocean-lobe bounces | path_tracer, sky | 30.6 -> 21.3 (Y), 34 -> 11.3 (RGB) | 631 -> 238 | 1.40x (1.60x at +10) | 1.0x (Y), 2.1x (RGB) |

Display RMS error at equal time, path tracer: 84 -> 73 at the reported view
(32 against 23 samples), 60 -> 42 at +10 degrees with the exposure matched.

### Ray marcher: wavelength bundles

`sample_wavelength_bundle` draws four wavelengths from one uniform number u:
member k inverts the CIE CDF at (u + k) / 4. The ray marcher marches all four
along one geometric path, so densities, step placement, surface hits, and
bounces are shared. Four lookups per step into the per-wavelength tables
would have cost 4x (measured: 3.9x), so the tables changed shape:

- **Column densities instead of transmittance.** Extinction separates into
  Rayleigh, Mie, and ozone, so a table of their column densities to the top
  of the atmosphere gives the exact transmittance at every wavelength from
  one lookup. Bilinear interpolation of the columns near the horizon was off
  by up to 8 % in transmittance at 64x64: the Mie profile's kinks at 1.3 and
  2.4 km cross the grazing rays, and the old table had a 2 % error at the
  same test ray. The table is 256x256, stores the logarithms, and is filtered
  in f32 in the shader. Against a march that integrates every sun column
  exactly, the table's error in the deep-twilight sky is 0.02 %; the old
  transmittance table's was 0.5 %.
- **Multiple scattering by bundle.** Slice i of the 32x32x128 rgba table
  holds, in channel k, the radiance at member k of the bundle with u = i /
  127, so one lookup returns the whole bundle.
- With one lookup per member and table, bundles of 2, 3, and 4 measured 1.9x,
  4.8x, and 3.3x equal-time gains in Y and 4.4x, 4.2x, and 7.0x in RGB; the
  4x lookups were most of the cost, which the tables above remove.

### Bounces off the ocean

`sample_earth_bounce` mixes cosine weighting with the GGX normal
distributions at the ocean's two roughnesses (GGX at the Beckmann alpha
stands in for the narrow lobe), chosen in proportion to the lobes' shares of
the BRDF, and divides by the mixture density (one-sample MIS, balance
heuristic). At a view 3 degrees above grazing, 28 % of the reflected
directions point below the surface and are lost; visible-normal sampling
would keep them.

### Path tracer: spectral MIS

The path carries the same bundle. A hero drawn uniformly from the bundle
decides each collision; all members share one majorant, so free flights have
the same density for every member, and each member's estimate is weighted by
the balance heuristic over the four possible heroes (Wilkie et al. 2014). The
weights stay bounded by 4 because they are renormalized to mean 1 after every
collision. A fixed hero (the bluest member) was biased by 1.8x in blue: the
balance heuristic needs every member to be the hero equally often. Clouds are
tracked before the Rayleigh, aerosol, and ozone media, so that the weights
observe only collisions in front of a cloud hit.

The shared majorant is the bluest member's, so redder members take more null
collisions, and the time per sample rises 1.4-1.6x. Chromatic noise falls 3x
and fireflies 2.7x, but luminance only 1.4x.

## Correctness

Converged means against the baseline, as ratios of the summed linear RGB
over a region, with the noise floor from two independent halves of the same
build.

| mode, view | region | samples | new / baseline (R, G, B) | noise floor |
| --- | --- | ---: | --- | --- |
| raymarch, -0.8 | sky | 2048 | 0.998, 0.998, 1.001 | 0.0005 |
| raymarch, -5.7 | sky | 2048 | 0.998, 0.995, 0.993 | 0.001 |
| raymarch, +10 | sky | 2048 | 1.000, 0.999, 1.000 | 0.0007 |
| raymarch, -5.7, bundles against exact sun columns | sky | 1024 | 1.0002 (Y) | same random numbers |
| raymarch, ocean-lobe bounces against bundles alone | ground, -0.8 | 2048 | 1.003, 1.001, 1.001 | 0.002 (bundles alone) |
| path_tracer, -0.8 | sky | 16384 | 1.002, 1.001, 0.999 | 0.011 |
| path_tracer, +10, sun disk masked | sky | 16384 | 1.000, 1.000, 0.999 | 0.001 |
| path_tracer, symmetric preset | whole image | 16384 | 1.000, 1.000, 1.000 | 0.0002 |

The ray marcher's sky shifts by up to 0.7 % in deep twilight, all of it the
old transmittance table's interpolation error: against exactly integrated
sun columns the new table is within 0.02 % and the old one 0.5 % bright.

## Open issues

- **Path tracer at twilight.** The remaining variance is paths that collide in
  air whose sun transmittance is near 0. Next-event estimation only at the
  collision cannot find the sunlit air further along the segment. Sampling an
  extra point per segment in proportion to the scattering times an estimate of
  the sun's transmittance (the ray marcher's column table, or a coarse march
  like its step placement), with ratio-tracked transmittances and MIS against
  the collision's own estimate, would put the samples there.
- **Hero majorant.** Tracking each member under its own majorant, as pbrt-v4
  does, would cut the null collisions of the redder members.
- **Visible-normal sampling** of the ocean lobes would stop losing a quarter of
  the specular bounces at grazing views.
