# Path tracer performance: where the time went and what bought it back

The textured path tracer took 72 ms per sample at 1080p on the reported sunset
view and over 50 ms on the hurricane preset, clouds and twilight being the slow
cases. This note records the attribution, the changes that paid, and the ones
that did not.

## Method

- **Time.** ms per sample at 1920x1080, 16 samples, RTX 4090 (it also drives the
  display), Vulkan. Views: the reported one (camera 35 m above the sea looking
  up at a sun 0.8 degrees below the horizon, overcast), and the `florida`,
  `sunset-hurricane`, and `apollo-11` presets; `symmetric` from the reported
  camera. Runs repeat within 2 %.
- **Attribution.** Temporary counters in the kernel, written to the image in
  place of radiance and averaged with `--out-linear`: steps of each loop per
  path. Ablations replace one piece with a stub (cloud shadow rays fully
  transmitting, the terrain as the planet sphere) and time the rest.
- **Equivalence.** 1024 to 2048 samples at 320x180 against the previous build:
  region means within 0.1 %, 10x10-pixel blocks within the noise (p1 to p99 of
  their ratio 0.97 to 1.03), except the false-ground band of row 21 in the
  outline's behaviour table.

## Where the time went

Steps per path at the reported view before the changes (3.8 bounces per path):

| loop | steps per path | each step |
| --- | ---: | --- |
| terrain sphere tracing | 270 | four texel loads, `atan2`, `asin` |
| cloud ratio tracking (shadow rays) | 245 | `atan2`, `asin`, a filtered fetch |
| RMO ratio tracking (shadow rays) | 143 | profile evaluation |
| cloud delta tracking | 47 | as the shadow rays |
| RMO delta tracking | 7 | profile evaluation |

The cloud majorant, 0.0029 / m, puts a step every 345 m; a sunset shadow ray
grazes some 300 km of the 6 km cloud shell on each side of its tangent point.
The terrain march stepped by the height above the terrain, metres for grazing
rays from a sea-level camera, and its 250-step cap then counted as a hit.

## What paid

ms per sample, cumulative:

| change | reported | florida | hurricane | symmetric |
| --- | ---: | ---: | ---: | ---: |
| before | 72.5 | 28.7 | 58.3 | 3.7 |
| bound map (terrain shell and skips, per-stretch cloud majorants); sun transmittance from the column-density table | 31.0 | | | 1.4 |
| terrain bound 0 lands on the sphere | 12.7 | 14.1 | 19.5 | |
| cloud map coordinates interpolated within a stretch | 10.7 | 12.4 | 16.7 | |
| terrain searched only before a cloud collision | 10.1 | 9.2 | 14.2 | |
| no dynamic indexing of bundle vectors | 10.0 | 9.2 | 14.2 | 1.1 |

`apollo-11` went from 18.1 to 7.3 before the last three rows. Steps per path at
the reported view afterwards: 15 cloud tracking, 23 cloud shadow, 19 terrain
skips and no field steps over the ocean, no RMO shadow steps.

- **Bound map.** A 4096 x 2048 map of the largest topography and cloud
  coverage that bilinear filtering can reach from each cell's 3 x 3
  neighbourhood (`src/assets/MapBounds.zig`, built in 0.1 s at load, 32 MiB).
  A ray outside the planet sphere cannot leave the neighbourhood within
  PLANET_RADIUS times the cell angle times the cosine of the latitude, which
  makes each lookup valid for about 9.7 km at the equator. The coverage map
  alone allows little skipping (3.5 % of 0.5-degree cells are clear), but a
  column of coverage c fills only the heights (0.2 - 0.2 c, 0.2 + 0.8 c) of the
  shell, so the bound's column and density cut the null steps to about a third.
- **Map coordinates.** Adding two `sphere_uv` per cloud step cost 5.6 ms of
  12.4, so the inverse trigonometry was the step's main cost. Over a stretch a
  straight ray's map coordinates stay within 1e-6 rad of a line, against
  2.9e-4 rad per Q2 texel; they are now exact at the stretch ends only.
- **Shadow rays through the RMO media** read the table the ray marcher already
  validated (`test_raymarch`), so the 143 steps became one lookup.

## What did not pay

- **Decomposition and residual ratio tracking** (Kutz et al. 2017; Novak et
  al. 2014) with a coverage minimum as control: 97 % of the cloud shadow rays at
  the reported view end blocked, but only a fifth of those were ended by the
  control, since broken cloud has a minimum of 0 at the 37 km scale. Splitting
  stretches into pieces of constant bounds cost 2.4 ms, a net loss of 3 ms.
- **Russian roulette in the cloud ratio tracking** below transmittance 0.05:
  a third fewer steps, no change in time.
- **Persistent threads with path regeneration** (2 to 32 pixels per thread,
  the bounce loop resumable): 12.3 to 13.1 ms against 10.6, at every count.
  Idling half the lanes cuts the time by 31 % rather than the 50 % of a
  throughput-bound kernel, so warps do wait on their slowest lanes, but the
  restructured loop's code generation lost more than regeneration recovered.
- **Q1 maps** (33 MB of clouds, inside the 72 MB L2) time the same as Q2: not
  bandwidth bound.

## What remains

- The kernel compiles to about 130 to 150 registers (from the CUDA output of
  the same Slang, as a proxy), so a third of the warps fit, and timings swing by
  30 % with code that does not run: removing the cloud shadow rays' code but
  not their work made the reported view slower. Splitting the megakernel
  (wavefront path tracing) is the next large step.
- `Media` scans the 221-entry ozone profile for its maximum on every path:
  0.2 ms, 15 % of `symmetric`.
- Terrain skips over the ocean: 19 to 22 per path at the reported and the
  hurricane views. A coarser level of the bound map would take them a few
  hundred kilometres at a time.
