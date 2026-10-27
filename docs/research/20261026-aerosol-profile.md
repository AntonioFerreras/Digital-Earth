# The aerosol profile

The atmosphere's Mie density by altitude, `mie_density` in
shaders/atmosphere/media.slang, and what it was replaced with on 2026-10-26.

## What the port inherited

The Taichi renderer's profile, four pieces joined continuously: linear to 1.3 km,
a Gaussian drop to 2.4 km, a slow Gaussian to 11.5 km, and a 1 km Gaussian tail.
Against Elterman's 1968 measured profile (searchlight soundings over White Sands,
averaged over 79 clear nights, extended to sea level by Penndorf's model and to
50 km by a fit; 23 km surface visibility), tabulated at 0.55 µm in a NASA report
(Table 4.4) and used by LOWTRAN, CMAQ and the Herman sky models since:

| layer | inherited share | Elterman share | inherited / Elterman at the layer top |
| --- | --- | --- | --- |
| 0 to 2.4 km | 54% | 69% | 1.4x |
| 2.4 to 5 km | 15% | 11% | 5.3x |
| 5 to 11.5 km | 28% | 9% | 4.8x |
| 11.5 to 20 km | 3% | 9% | 0 |
| above 20 km | 0% | 2% | 0 |
| column, km of sea-level density | 3.07 | 1.51 | |

The inherited profile holds a plateau of 0.09 to 0.19 of the sea-level density from
2.4 km to the tropopause and a lid at 13 km: a uniform haze slab seen from altitude
as a solid band with a sharp top. The measured profile falls to 3% of the surface by
5 km, stays at 2% through the free troposphere, and tails off through the
stratosphere with a sulphate layer about 20 km. The inherited profile's surface, on
the other hand, is very clean: with turbidity 1.03 and Preetham's coefficient the
sea-level extinction at 550 nm is 0.011 /km, a visibility of 170 km, and the aerosol
optical depth 0.034, against Elterman's 0.158 /km and 0.24. The two errors
compensated in the look: thin surface air under a thick mid-troposphere.

## The fit

Three terms, fitted in log space: a boundary layer and a free troposphere each
exponential, over a Gaussian stratospheric layer. The data are Elterman's 0 to 12 km
and, above, anchors for a modern background stratosphere (Elterman's own 1963 to 1965
stratosphere carries Agung's ash, about twice today's), with the Gaussian fixed at
20 km, 4 km wide, with a 550 nm optical depth of 0.005 relative to Elterman's surface
extinction. Normalised to 1 at sea level, which is also the peak:

    0.85603 exp(-h / 729.1 m) + 0.14397 exp(-h / 3650 m) + 0.003818 exp(-((h - 20 km) / 4 km)^2)

| altitude km | fit | Elterman | fit / Elterman |
| --- | --- | --- | --- |
| 1 | 0.327 | 0.440 | 0.74 |
| 2 | 0.138 | 0.190 | 0.73 |
| 3 | 0.077 | 0.080 | 0.97 |
| 5 | 0.037 | 0.032 | 1.18 |
| 8 | 0.016 | 0.021 | 0.75 |
| 11 | 0.0071 | 0.019 | 0.38 |
| 15 | 0.0032 | 0.017 | 0.19 |
| 20 | 0.0044 | 0.0094 | 0.47 |
| 30 | 0.00005 | 0.0008 | 0.06 |

Up to 8 km the fit stays within 0.73 to 1.18 of Elterman; from 11 km up it follows
the modern background by design, below Elterman's Agung-era values. CALIPSO's global
mean profile decays faster through the free troposphere than Elterman's plateau, so
the fit sits between the two there. The column is 1.18 km of sea-level density, 38%
of the inherited profile's, so the aerosol optical depth falls from 0.034 to 0.013 at
the same turbidity while the surface extinction is unchanged; the sky above the boundary layer clears, and the horizon band seen from
altitude thins. The three-term profile has no joins, which removes the 1% error on
rays grazing the inherited profile's kinks that the column-density table's check
found.

Turbidity still scales the whole profile, as before. Scaling the boundary layer
alone, as the NASA report does with visibility, is the next step if the surface haze
is to be tuned separately from the free troposphere.

## What depends on it

The ray marcher's column-density and multiple-scattering tables and the cloud
bench's sky bake at start-up, so they follow the shader. The neural mode's training
data and the Bruneton tables were generated with the inherited profile and are stale
until rerun. The golden `media_density` set's Mie column is this fit
(python/goldens.py `mie_density_reference`).

## Sources

- L. Elterman, UV, Visible, and IR Attenuation for Altitudes to 50 km, AFCRL-68-0153,
  1968; tabulated in NASA report 19850013446, Table 4.4,
  https://ntrs.nasa.gov/api/citations/19850013446/downloads/19850013446.pdf
- Shettle 1989 layered models as libRadtran's default (rural boundary layer to 2 km,
  background troposphere, background stratosphere), https://www.libradtran.org/doc/libRadtran.pdf
- Hess, Koepke, Schult 1998, OPAC, four-layer height profiles
- MODTRAN's boundary layer, tropospheric, stratospheric and upper regions,
  https://web.gps.caltech.edu/~vijay/pdf/modrept.pdf
- CALIPSO boundary-layer versus free-troposphere shares of the optical depth,
  https://www.nature.com/articles/s41598-020-79361-4
