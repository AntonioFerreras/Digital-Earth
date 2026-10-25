# Cloud bench: validation and first scores

The `cloud_bench` mode renders one dense cloud volume two ways: an exact volumetric
path tracer (the reference) and the ray marcher's single scattering plus its
multiple-scattering model, `clouds/multiple_scattering.slang` (the approximation).
The earth marcher calls the same model. The earth's cloud shell is 2.5D and thin, so
a model fitted there need not hold for real 3D clouds. The bench exposes it to a
cumulus that shows every multiple-scattering effect: bright edges, dark cores and
valleys, silver linings, and light diffusing through thick cloud.

## Data

The Walt Disney Animation Studios cloud data set
(<https://disneyanimation.com/resources/clouds/>, CC BY-SA 3.0), unpacked under
`sources/wdas_cloud/`. `python/clouds/convert_vdb.py` turns a VDB into the bench's
`.ecld` file: dense f16 density and majorants per cell of 8^3 voxels. The
quarter-resolution cloud is 498 x 338 x 613 voxels of 0.833 units (207 MB). It
converts in 2 s. The half-resolution one, the bench's default since, is 994 x 676 x
1225 voxels of 0.417 units (1.65 GB) and converts in 36 s. The full one, about 13 GB
dense, would need sparse storage.

```sh
pixi run -e py python python/clouds/convert_vdb.py \
    sources/wdas_cloud/wdas_cloud/wdas_cloud_half.vdb cache/clouds/wdas_cloud_half.ecld
```

At 666 x 360, the reference costs 17 ms/spp at half resolution against 15 at quarter.
The approximation steps by fractions of a voxel, so it costs 32 ms/spp against 8,
and 85 ms per frame at 1920 x 1080.

## Reference against Mitsuba

The bench first reproduced the data set's Mitsuba scene: its camera, sun, uniform
sky, ground disk, Henyey-Greenstein g of 0.877, and density scale of 4. At 333 x 180
and 1024 spp, against `wdas_cloud_mitsuba_scene.exr`:

| quantity                            | value                  |
| ----------------------------------- | ---------------------- |
| image mean, ours                    | 0.1757, 0.2042, 0.3297 |
| image mean, Mitsuba                 | 0.1764, 0.2053, 0.3323 |
| cloud luminance ratio, ours/Mitsuba | 0.993                  |
| cloud log-rms per pixel             | 0.128                  |

The per-pixel residual is mostly Mitsuba's own noise at 1024 spp, plus the
quarter-resolution grid against Mitsuba's full one. The reference costs 12 ms/spp at
333 x 180.

## Approximation, Mitsuba lighting

Scored by `python/clouds/score.py`: ratios of mean luminance, approximation over
reference, overall and per third of the reference's cloud pixels by luminance.

| run                 | mean  | dark  | mid   | bright | log-rms |
| ------------------- | ----- | ----- | ----- | ------ | ------- |
| single scattering   | 1.015 | 1.043 | 1.009 | 1.017  | 0.078   |
| multiple scattering | 0.196 | 0.116 | 0.190 | 0.223  | 3.66    |

- **Single scattering:** it matches once the sun's optical depth is marched at every
  step. A grid of it at twice the voxel size gave 0.80. The cloud's mean free path,
  0.25 units at full density, is shorter than a voxel, so the shadow near a lit
  surface changes within one texel.
- **Multiple scattering:** the model's constants were fitted to the earth's thin
  cloud deck, and here it reaches a fifth of the reference. In this view the sun is
  behind and to the side, so the cloud's light is almost all multiply scattered.
- **Cost:** the approximation costs about 12 ms per frame at 1920 x 1080. The
  reference keeps accumulating while the model's constants change.

## Approximation, atmosphere lighting

The bench now lights the cloud as the earth does. The sky is the ray marcher's
atmosphere, baked at the cloud's centre as Hillaire's sky-view table, over a black
planet with no ground. The sun is the solar spectrum through the atmosphere. The
cloud's base sits about 1 km up, and it scatters by the droplet phase.
`presets/wdas-cloud.zon` keeps the Mitsuba camera and sun. Scored at 333 x 180,
512 spp:

| run                 | mean  | dark  | mid   | bright | log-rms |
| ------------------- | ----- | ----- | ----- | ------ | ------- |
| single scattering   | 1.015 | 1.065 | 1.014 | 1.014  | 0.496   |
| multiple scattering | 0.215 | 0.153 | 0.209 | 0.229  | 4.28    |

The single-scattering log-rms is dominated by the reference's noise. Light scattered
once is a faint backscatter in this view.

## Approximation, real cumulus

Since then the cloud is 10 m per volume unit: 2.8 km tall, 4 x 5 km across, with its
base 1 km up. Its extinction is set per metre, 0.1 /m at unit density by default,
about a real cumulus's, and it uses the half-resolution grid. The approximation's
steps span at most a quarter of a cell's majorant optical depth, scaled by one
plus the optical depth already marched. Its light falls as exp(-depth), so deep
steps can be coarse. 666 x 360, coverage above 0.5:

| run                           | mean  | dark  | mid   | bright | log-rms |
| ----------------------------- | ----- | ----- | ----- | ------ | ------- |
| single scattering, 256 spp    | 1.003 | 1.151 | 1.009 | 1.000  | 0.830   |
| multiple scattering, 128 spp  | 0.246 | 0.250 | 0.278 | 0.227  | 2.64    |

The reference costs 9 ms/spp at 666 x 360. The approximation costs 18 ms/spp, and
47 ms per frame at 1920 x 1080.

## Dimensional profile

`python/clouds/convert_vdb.py` stores the distance from inside the cloud to its
edge on a grid twice as coarse as the density. The cloud is where that grid's mean
density exceeds 0.1, and the distance comes from SciPy's Euclidean distance
transform. The profile is that distance in optical depths at full density, divided
by `profile_depth`, so it means the same at any scale and extinction. Following Nubis
Cubed, it scales the multiply scattered sunlight by lerp(1 - strength, 1, profile),
and the sky's light by lerp(1, sqrt(1 - profile), strength). The earth's shell has
no shape to take a profile from and passes 1.

With the other constants as fitted to the earth, strength 1, and a depth of 10, the
multiple-scattering mean falls from 0.246 to 0.053 of the reference. The view sees
only the first few optical depths below the surface, where the profile is still
near 0.

## Transfer probes

The design is `docs/outlines/20261023-cloud-transfer-probes.md`. The first version
on the bench:
- 64^3 probes over the volume's box, about 65 x 44 x 80 m apart.
- One backward random walk per probe per frame. It has no similarity switch and the
  reference's bounce limit.
- Order-2 SH transfer. Walks that escape without a collision are skipped, so a probe
  holds light scattered once or more (K = 1).
- The probes are lit every frame with the sun as a delta and the sky's table
  projected to SH.
- The marcher scatters the probes' P1 radiance through the first moment of the
  droplet phase.
- The sky's single scattering is taken from the column straight up, exp(-tau_up)
  times the sky's mean.

960 x 540, reference 512 spp, approximations to 512 frames. Coverage above 0.5;
"darkest" is a share of the reference's cloud pixels:

| run                    | mean  | dark  | mid   | bright | log-rms | darkest 20 % | darkest 5 % |
| ---------------------- | ----- | ----- | ----- | ------ | ------- | ------------ | ----------- |
| Nubis model, defaults  | 0.900 | 0.888 | 1.028 | 0.824  | 2.29    | 0.749        | 0.070       |
| probes, 512 walks      | 1.097 | 1.166 | 1.114 | 1.071  | 0.291   | 1.170        | 1.352       |

- **Occluded regions:** the probes light the shadowed lobe and the base. The model's
  darkest 5 % reach 7 % of the reference.
- **Bias:** the probes are 10 % too bright overall, and more in the darkest pixels.
  The likely causes are trilinear blur of the light scattered once or twice near lit
  surfaces, into points a mean free path deeper, and order-2 SH smoothing of the
  peaked escape directions of walks that scatter once. K > 1 and finer probes would
  separate the two.
- **Cost:** a walk per probe costs about 8 ms per frame (34 against 26 ms/spp for the
  approximation with the model). Lighting the probes is negligible.

### Sun transfer in log-SH

Linear order-2 SH sun transfer goes black in the sun's shadow: a shadowed probe takes
orders of magnitude less light from the sun's direction than from directions where
the cloud is thin, and the fit rings below zero there. Smoothing the sun's harmonics
to a non-negative kernel removes the black but makes the shadow 35 times too bright.

The sun's transfer is now baked by next-event estimation into 8 x 8 equal-area
octahedral bins of light direction: at every collision a walk draws a direction from
the phase and adds the transmittance towards it. Each frame the bins are fitted to
order-2 SH of log phi (9 values) and order-1 SH of F / phi (12 values). The sky
keeps its linear transfers. Backlit view, sun 65.6 degrees, no sky, reference 1024
spp, 960 x 540; "black" = pixels below a tenth of the reference:

| run                         | mean  | dark  | mid   | bright | log-rms | black |
| --------------------------- | ----- | ----- | ----- | ------ | ------- | ----- |
| linear SH, 512 walks        | 0.824 | 3.262 | 1.251 | 0.737  | 3.71    | 27 %  |
| raw bins, 128 walks         | 0.768 | 2.036 | 1.066 | 0.714  | 3.44    | 33 %  |
| raw bins, 512 walks         | 0.779 | 2.044 | 1.072 | 0.726  | 2.17    | 14 %  |
| raw bins, 2048 walks        | 0.791 | 2.156 | 1.104 | 0.733  | 1.46    | 3.4 % |
| log-SH, 512 walks           | 0.537 | 0.198 | 0.347 | 0.565  | 2.20    | 30 %  |
| log-SH, 2048 walks          | 0.669 | 0.945 | 0.879 | 0.641  | 1.20    | 3.7 % |

- **The remaining black is noise.** Light reaches a deep shadowed probe from the sun
  only along rare walks that wander to the lit side, so its sun bin converges slowly.
  The black falls from a third to a few percent between 128 and 2048 walks.
- **The log fit is biased low while it is noisy:** the log of a noisy bin averages
  below the log of its mean. At 2048 walks it reaches 0.67 of the reference against
  the raw bins' 0.79.
- **The bright rims are low in both,** at about 0.7: a 25-degree bin averages away
  the forward peak of light scattered a few times.
- **Cost:** the next-event estimate at every collision raises the approximation's
  frame from 73 to 99 ms at 960 x 540, so 2048 walks take about 200 s.
- **128^3 probes,** about 32 x 22 x 40 m apart: at 512 walks they score as 64^3 does
  (mean 0.546, black 30 %), as each probe's noise is its own; the frame costs 168 ms
  against 99, and the probes hold about 3 GB, 2.1 GB of it the sun's bake-only bins.

### Marched next-event estimates

Each walk's sun estimate per collision is now marched, not ratio tracked. A collision
whose optical depth towards the sun passes 8 counts as unlit. Backlit view at 64^3:

| sun estimate                          | frame  | walks | mean  | dark  | mid   | bright | black |
| ------------------------------------- | ------ | ----- | ----- | ----- | ----- | ------ | ----- |
| ratio tracking                        | 59 ms  | 512   | 0.537 | 0.198 | 0.347 | 0.565  | 30 %  |
| ratio tracking                        | 59 ms  | 2048  | 0.669 | 0.945 | 0.879 | 0.641  | 3.7 % |
| march, past depth 10 = exp(-10)       | 119 ms | 512   | 0.640 | 1.131 | 0.805 | 0.614  | 1.6 % |
| march, past depth 6 unlit             | 94 ms  | 512   | 0.640 | 0.623 | 0.689 | 0.635  | 9.6 % |
| march, past depth 8 unlit             | 104 ms | 512   | 0.643 | 0.665 | 0.708 | 0.635  | 8.3 % |
| march, past depth 10 unlit            | 115 ms | 512   | 0.644 | 0.673 | 0.712 | 0.635  | 8.1 % |
| march, past depth 8 unlit, 2-voxel steps | 107 ms | 512 | 0.644 | 0.666 | 0.710 | 0.636 | 8.3 % |
| march, past depth 8 unlit             | 105 ms | 2048  | 0.703 | 1.496 | 1.032 | 0.653  | 1.0 % |

- **The cutoff is harmless from 8 on:** 8 and 10 agree. Counting a march past the
  limit as exp(-limit) instead gives every deep collision that floor, which brightened
  the dark third from 0.67 to 1.13 at 512 walks.
- **2-voxel steps save nothing:** the march's steps already lengthen with depth.
- **Both estimators still converge at 2048 walks:** the dark third rises with walks as
  the log fit's low bias from noisy bins fades, and the march gets there sooner. At
  2048 walks the march's dark third overshoots to 1.5; where it settles is open.

### Sun fit: order 3 in the log

The order-2 log fit had only been scored backlit. Front-lit with sky it came out 1.80
times the reference, while lighting from the raw bins gave 1.04. The fit, not the
bake: dumping the bins and fitting offline (64^3, 800 walks) showed the order-2 log
projection overshooting on probes at the cloud's surface, whose log fluence steps
between directions open to the light and directions through the cloud: 1.44 times
the bins at the front sun, 1.52 with the flux term. Alternatives, against the bins:

- **Floor 1e-3, Hanning window:** front 1.49, backlit dark third 1.53; front 0.76,
  backlit dark third 2.75.
- **Order-2 log by least squares weighted by sqrt(phi / max):** the bright directions
  fit within 4 %, but sun directions under 1/1000 of a probe's brightest come out
  40 to 70 times too bright. Rendered: front 1.13, backlit dark third 8.2.
- **exp(-k tau) factored out, tau the probe's straight optical depth:** worse for
  every k > 0; multiply scattered light does not follow the straight path.
- **A tanh step in the log, 6 values:** the same residual as order 2.
- **Order 3 in the log, 16 values:** best on both views offline.

Rendered at 800 walks, order 3 in the log:

| view                     | mean  | dark  | mid   | bright | log-rms | black  |
| ------------------------ | ----- | ----- | ----- | ------ | ------- | ------ |
| front-lit, sky           | 0.987 | 1.170 | 1.073 | 0.891  | 0.315   |        |
| backlit, no sky          | 0.714 | 0.737 | 0.826 | 0.700  | 1.44    | 10.4 % |

Order 3 holds 64 values a probe against order 2's 57 and needs more walks to converge.
The order is a bench setting; judged by eye, order 3 is better, and it is the default.

### Sky single scattering and the sky's transfers

The sky's single scattering moved into the probes: walks that leave the cloud without
a collision fill a second, sky-only transfer, as the sun's single scattering is
marched exactly. Front-lit with sky, 512 walks, linear sun transfer: mean 1.079
against 1.097 with the column, darkest 5 % 1.50 against 1.35. Both sky transfers are
lit by the same sky harmonics and summed, so they were merged into one (36 values)
once the sun moved to its own fit; the merge changes nothing.

### Bench settings

The walk limit (default 800), the sun estimate per collision (march or ratio
tracking), the sun's log-SH order (2 or 3) and the first-scatter treatment below are
panel settings. A changed order refits from the stored bins without rebaking.

### Probe cache

Bakes persist under `cache/probes`, one file per combination of what the walks depend
on: the volume's grid, the probe resolution, extinction, droplet or Henyey-Greenstein
phase (its g only for the latter), bounce limit and sun estimate
(`src/render/ProbeCache.zig`). A file holds the raw sums, the sky transfer and the
sun's bins, 306 MB at 64^3, so that fits, order and light can change and the bake can
resume. It is stored, by a blocking readback, when the bake reaches its limit, its
settings change or the bench closes, if it holds more walks than the file and at
least 64, so that dragging a slider leaves no file per value.

### Frame costs

960 x 540, bake finished unless stated:

| pass                                   | front-lit | backlit |
| -------------------------------------- | --------- | ------- |
| reference only                         | 16.7 ms   | 14.3 ms |
| approximation, Nubis model             | 26.9 ms   | 37.2 ms |
| approximation, probes                  | 26.4 ms   | 36.3 ms |
| approximation, probes, while baking    | ~102 ms   | ~113 ms |

The approximation costs more than the path tracer because it marches the optical
depth towards the sun at every step. That march is nearly all of it: with the forward
peak, the probes' frame was 31 / 46 ms, 0.7 ms without the march and 0.56 ms for the
view march alone; the probe reads cost about 0.15 ms. The march's cost is the majorant
cells it crosses, not its density taps (steps of 1 to 8 voxels cost the same). Stopping
it at an optical depth of 12 rather than 100 wherever only the beam, exp(-depth), is
used gives 16.0 / 16.8 ms with the image unchanged (log-rms 0.0001); the Nubis model's
high order scattering still marches to 100. A sun at exactly 270 degrees lost the device:
its direction's components of about 1e-16 put the majorant walk's crossing distances
past what floats resolve, so the walk never ended. Components under 1e-8 now count as
zero.

### Where the light varies

From the baked sums (64^3, extinction 0.1 with 5160 walks and 0.453 with 4096), each
probe's relative error when interpolated linearly from its neighbours 2k probes apart,
for the sun's (phi, F) averaged over its directions, by the distance inside the cloud
in mean free paths at unit density. Where the error stays the same as the spacing
doubles it is the walks' noise, about 0.03 to 0.05:

| depth          | extinction 0.1: x2 / x4 / x8 | extinction 0.453: x2 / x4 / x8 | abs(F) / phi |
| -------------- | ---------------------------- | ------------------------------ | ------------ |
| air, adjacent  | 0.19 / 0.30 / 0.32           | 0.24 / 0.31 / 0.31             | 0.44-0.49    |
| 0 to 0.5       | 0.11 / 0.19 / 0.28           | 0.17 / 0.23 / 0.32             | 0.18-0.20    |
| 0.5 to 1       | 0.03 / 0.05 / 0.22           | 0.16 / 0.26 / 0.35             | 0.10         |
| 1 to 2         | 0.03 / 0.03 / 0.07           | 0.06 / 0.19 / 0.33             | 0.08-0.10    |
| 2 to 4         | 0.03 / 0.03 / 0.03           | 0.04 / 0.06 / 0.27             | 0.08-0.09    |
| 4 to 8         | 0.03 / 0.03 / 0.03           | 0.04 / 0.05 / 0.13             | 0.09         |
| 8 to 16        |                              | 0.05 / 0.05 / 0.08             | 0.10         |

- **Spacing:** linear interpolation holds to the noise wherever the spacing is under
  about 2.5 times the distance to the surface, at both extinctions. The surface layer
  and the air beside it vary fastest and are worst already at the bake's spacing.
- **Direction:** inside, the light is nearly isotropic, abs(F) / phi about 0.09 from
  half a mean free path on; in the surface layer 0.2. How much the sun's direction
  matters, the spread of ln phi over its directions, falls from 1.5 at the surface to
  0.5 four mean free paths in (extinction 0.1).
- **Count:** 73 % of the dense grid's probes lie in air not next to cloud and are never
  read. Cloud and adjacent air hold 70946; spacing each at up to 2.5 times its depth
  leaves about 34000, 8 times fewer than the dense grid.

### Adaptive layout

The layout of `docs/outlines/20261023-adaptive-cloud-probes.md`, at a budget of 2^18:
257760 probes, the finest 3.56 apart against the dense grid's 6.47, built in about
0.3 s. The lookup interpolates a leaf's corners exactly (probes lit with their own
positions return the point's to half-float precision). 960 x 540, forward peak, the
dense grid from 5160 walks, the adaptive one from 2048:

| layout   | front cell-scale | front bias | backlit cell-scale | backlit bias | backlit dark | bake frame |
| -------- | ---------------- | ---------- | ------------------ | ------------ | ------------ | ---------- |
| dense    | 0.070            | 0.116      | 0.349              | 0.577        | 1.47         | 66 ms      |
| adaptive | 0.077            | 0.130      | 0.345              | 0.472        | 1.17         | 82 ms      |

Bias is the rms of the log error over 16 pixels; the frame without the bake costs the
same, 16 ms. Backlit, the adaptive layout's large-scale error falls by a fifth and its
dark third comes closer, but its interior shows each probe's noise in blotches the
dense grid's wider cells averaged away. A bake frame costs more, not less: nearly all
its probes start their walks in cloud, where walks are long, while most of the dense
grid's start in air and leave at once.

### Bake cost and noise

A bake frame at 960 x 540, less the 16 ms the approximation costs: dense 66 ms,
adaptive 81 ms; without the march towards the sun at each collision, 10 and 12 ms.
The march is 85 % of the bake.

Two independent halves of the adaptive bake (1024 and 1435 walks) give each probe's
relative noise from one half: the sun's fluence after the log fit, over random sun
directions, median 0.16 and 0.56 at the 90th percentile; the sky's, 0.009. The sun's
bins hold it: one bin's fluence is off by 0.17, all 64 summed by 0.022, as each
collision feeds one bin of 64.

### Faster bakes

Variants of the walks' sun estimate on the adaptive layout (2^18 probes). Noise is the
median relative noise of the sun's fluence after the fit from 128 walks (two
independent halves); cost is a bake frame, one session, the GPU shared with the
desktop; mean ratio against the reference at 512 walks, front-lit and backlit:

| sun estimate                                     | noise | bake frame | mean ratio    |
| ------------------------------------------------ | ----- | ---------- | ------------- |
| march towards one drawn direction                | 0.54  | 93 ms      | 0.975 / 0.921 |
| the same, skipping collisions no direction lights | 0.54  | 68 ms      | same          |
| the same, starting directions a low-discrepancy sequence | 0.53 | 68 ms | same     |
| direction volumes alone, cells of 8 voxels       | 0.32  | ~57 ms     | 0.876 / 0.806 |
| direction volumes alone, cells of 4 voxels       |       | slower     | 0.966 / 0.868 |
| direction volumes and the march as control variate | 0.40 | 115 ms    | 0.979 / 0.946 |
| direction volumes past a march of the first 8 voxels | 0.33 | 351 ms  | 0.918 / 0.809 |

- **Direction volumes:** per direction of the 64 bins, the optical depth from each
  cell's centre, built in a fraction of a second. Every collision then feeds every
  bin, but a bin stays noisy: each direction is lit by different collisions, near
  different lit surfaces. Interpolating the depth across a lit surface, where it
  starts to rise, overestimates it just inside, and the probes come out darker;
  interpolating the transmittance instead overshoots, a mean ratio of 1.38.
- **Skipping:** a collision whose least depth over every direction, in its cell and
  the cells around it, lies past 12 cannot see the sun; most collisions are such.
- **Where the volumes go dark:** not at the lit surface itself, since marching the
  first cell exactly leaves them as dark, nor in the bins' width, since marching
  towards bin centres comes out brighter. Light reaches a collision through gaps
  between billows finer than a cell, and a depth interpolated across such a shadow
  edge loses more light than it gains: the volumes cannot be made unbiased short of
  the voxels' resolution.
- Cost times variance against the march: skipping 1.37 times better, the control
  variate 1.46, the volumes alone 4.6 but biased. The bake keeps the march and the
  skip; the skip's least depths are marched per cell directly, 3 MB, and the
  direction volumes are gone.

### Photons

The walks' adjoint: light enters the grid's box from a uniform direction, and each
path adds its length to every probe it passes, by the probe's trilinear weight over
the integral of that weight, a kernel estimate of the fluence; past its first collision
it feeds the sun bin of its entry direction, all of it the sky's transfer. One round
traces as many paths as there are probes. Adaptive layout, 2^18 probes, noise from 128
rounds as above, mean ratio at 512:

| bake                  | round | of which the trace | sun after fit | per bin | summed over bins | sky   | mean ratio    |
| --------------------- | ----- | ------------------ | ------------- | ------- | ---------------- | ----- | ------------- |
| walks, march and skip | 70 ms | 57 ms              | 0.54          | 0.57    | 0.068            | 0.028 | 0.976 / 0.924 |
| photons               | 27 ms | 14 ms              | 0.32          | 0.21    | 0.022            | 0.022 | 0.971 / 0.964 |

- **Why:** a walk informs one probe; a path every probe it crosses, most of them near
  the surface, where light enters and the probes are finest.
- **Cost times variance of the trace,** against the walks: 11 times better for the sun
  after the fit, 29 per bin, 38 summed over bins. The walks' 2048 are matched in 200
  to 700 rounds, 3 to 10 s of tracing.
- **The rest of a round** is the fit, lighting and atlas fill, rerun every round while
  baking, 13 ms. Of the trace, the sky's atomic adds took 5 ms, the sun's 2.
- **The sky from the bins:** the sky's light past its first collision is the bins'
  light under the sky's radiance at each bin's direction, so the photons add the
  sky's transfer only before their first collision: a round 27 to 24 ms, the images'
  ratios unchanged to the third digit.
- **Where a round goes,** at 8 rounds a frame: finding collisions, mostly in the dense
  core, 65 % of the trace; the atomic adds 18 %; stepping and leaf lookups 17 %. The
  fit and lighting take under 3 ms a frame, so refitting only every 4 frames while
  baking saves little; the rest of a frame is the image itself.
- **Bins stratified,** each taking an equal share of the photons: no change in noise.
  A bin's photon count varies by about 1.6 % a round, far below where its photons
  happen to go.
- **Similarity deep inside** (isotropic scattering at (1 - g) the extinction once a
  photon lies a margin of 1 to 4 transport mean free paths inside, its flights stopped
  short of the margin): no faster. A photon collides about 30 times, half of them
  deeper than 8 cloud units, but the cloud's median depth is 10 units, one to two
  transport mean free paths at its mean density; the stopped flights cost about as many
  events as the collisions saved, 25 against 30. The probes agreed with the exact
  bake within 0.3 % at every margin.
- **Path lengths per leaf** instead of points a finest cell apart: the integral of each
  corner's trilinear weight along the chord, exact by Simpson's rule, one corner drawn
  in proportion. A round costs 1.3 times as much, as fine leaves are crossed more
  often than they were sampled; the sun's noise after the fit falls from 0.32 to 0.23,
  1.4 times as efficient, per bin from 0.21 to 0.20, 0.86 times. The fluence agrees
  with the points' within 0.5 % per decile.
- **Backlit,** the photons come closer to the reference than the walks, the dark
  fifth at 1.15 against 0.63.

### Grid: noise or interpolation

The cell-scale error is the rms of the log error band-passed to 2 to 16 pixels, the
scale of a probe cell. Between 800 and 3200 walks the image's content at that scale
changes by 0.07 against 0.30 front-lit: the visible grid is interpolation, not the
probes' noise.

### Reconstruction from diffusion theory

The derivation is in the outline. With the probes' light less the part locked to the
sun's delta-scaled beam, and divided by the distance to the extrapolated boundary,
800 walks, order 3:

| variant                    | front mean | front cell-scale | backlit bright | backlit cell-scale |
| -------------------------- | ---------- | ---------------- | -------------- | ------------------ |
| plain trilinear            | 0.987      | 0.131            | 0.706          | 0.492              |
| collimated part only       | 0.993      | 0.131            | 1.103          | 0.508              |
| beam-locked part           | 1.015      | 0.158            | 1.223          | 0.496              |
| boundary ramp              | 1.022      | 0.142            | 0.723          | 0.498              |
| both                       | 1.123      | 0.186            | 1.292          | 0.502              |

The beam's diffuse term drew dark creases along lit bumps; by eye both corrections
were worse, and they were removed.

### First scatter

Scored with the bake finished and loaded from the cache (800 walks, 800 frames; runs
that bake while they accumulate keep only the frames after the last doubling of the
walks, and their noise is not comparable):

| first scatter                          | front cell-scale | backlit cell-scale | backlit bright | frame, front / back |
| -------------------------------------- | ---------------- | ------------------ | -------------- | ------------------- |
| probes                                 | 0.095            | 0.392              | 0.677          | 27 / 37 ms          |
| gather, peak lobe only                 | 0.082            | 0.364              | 0.766          | 27 / 37 ms          |
| gather, broad lobe only                | 0.087            | 0.369              | 0.704          | 27 / 37 ms          |
| gather                                 | 0.077            | 0.348              | 0.794          | 27 / 37 ms          |
| forward peak, 256 frames               | 0.074            | 0.350              | 0.901          | 32 / 47 ms          |

- **Gather:** one sampled scatter per pixel and frame, at a step chosen in proportion
  to its weight; unbiased, but noisy.
- **Lobes:** the droplet phase's forward peak, its Henyey-Greenstein lobe of 53 %,
  carries about two thirds of the gather's gain and most of the rims'.
- **Forward peak:** the peak taken as straight on, its light carried along the view;
  as good as the gather without its noise, and the rims closest to the reference. It
  marches deeper, through (1 - f) of the extinction, so its frame costs 5 to 10 ms
  more. It is the default.

The reference's own noise at 512 and 1024 spp lies in the same band, so the
cell-scale numbers share a floor.

### Closure

With the sun behind the camera, the probe read turned clouds' sunlit edges black: the
light there flows almost as a beam away from the viewer, so P1's
(phi + 3 g F . -dir) / 4 pi with 3 g = 2.6 went negative and was clamped. Worst with
the plain probe read, halved by the gather and by the forward peak (whose broad lobe
has the smaller g), gone with the forward peak and a gathered broad lobe. The probe
read now takes a von Mises-Fisher lobe with the probes' phi and F (the M1 closure),
which the phase widens by multiplying its mean cosine by g: non-negative, and equal to
P1 where the light is barely directional. By eye it removes the black edges. In the
default front-lit view, probe read alone, 256 frames: mean 0.945 against P1's 0.979,
dark third 0.947 against 1.085, frame time unchanged. P1 stays a setting.


## Render cost

The approximation at 1024 x 1024, 32 spp, minimum of six runs, in ms per sample: view
00 with the sun overhead, and view 02 with sun 1, from `tests/validation/views`.

- **The sun's march is nearly all of it.** Without it, 8.1 and 18.0 fall to 1.1 and 1.5.
  Without the probe reads nothing changes.
- **The density fetch costs nothing at the margin.** A second fetch per step, near or
  forty voxels away, a point load instead of the filter, or half the fetches leave the
  time as it is; issuing four fetches at once, or one flat loop over fetches and cells,
  is slower.
- **The cells walked are the cost.** A sun ray walks 36 to 46 majorant cells for 22
  fetches, each cell a load that the next step waits on. Majorant cells of 16 voxels
  take the two views to 7.0 and 10.7, of 32 to 7.4 and 10.3; with the march stopping at
  optical depth 6 instead of 12 as well, 6.0 and 9.1. A march without the walk is 49
  and 37: skipping empty space is essential.

- **Skipping aligned empty blocks does not help.** Empty cells holding the size of the
  empty aligned block around them, which the walk then crossed in one step, saved
  nothing: the empty cells a sun ray crosses lie beside cloud, in blocks that are not
  empty. Crossing the block in one jump also slowed every walk by 1.7, block or not.
- **Coarser cells for the approximation's marches** (`view_march_cells`,
  `sun_march_cells`), each bounding the cells it covers, with the views' error against
  a 128 spp render at four times the view steps, which stays at 32 spp's noise:

  | view cells | sun cells | view steps | view 00 | view 02 | view 04 |
  | ---------- | --------- | ---------- | ------- | ------- | ------- |
  | 1 | 1 | 1    | 7.80 (0.0021) | 17.87 (0.0005) | 16.61 (0.0027) |
  | 1 | 2 | 1    | 6.09 (0.0021) | 11.11 (0.0005) | 11.22 (0.0027) |
  | 1 | 4 | 1    | 6.11 (0.0021) |  9.27 (0.0005) |  9.52 (0.0027) |
  | 2 | 2 | 1    | 6.99 (0.0020) | 11.09 (0.0005) | 11.32 (0.0026) |
  | 2 | 4 | 1    | 6.49 (0.0020) |  9.45 (0.0005) |  9.73 (0.0026) |
  | 4 | 4 | 1    | 7.32 (0.0020) | 10.66 (0.0005) | 12.14 (0.0026) |
  | 1 | 4 | 0.5  | 4.02 (0.0022) |  5.69 (0.0006) |  5.92 (0.0028) |
  | 1 | 4 | 0.35 | 3.44 (0.0024) |  4.64 (0.0009) |  4.95 (0.0030) |
  | 1 | 4 | 0.25 | 3.07 (0.0028) |  3.95 (0.0013) |  4.36 (0.0036) |

  Coarse cells pay towards the sun, not along the view. Fewer view steps leave the
  converged image as it is, the march being jittered; at 8 spp half the steps raise
  the error against the reference from 0.0043 to 0.0050 in view 00.

The defaults became sun cells 4 and view steps 0.4.
