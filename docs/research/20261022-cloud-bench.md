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

## Control variates for the walks

Tried for the backward walks, as the photons' rest would feed every probe downstream
rather than one. A walk adds the probes' prediction of its rest at a collision, then
goes on with probability q at weight 1 / q and takes the prediction off again: the
expectation is the plain walk's whatever the prediction. The prediction is P1 of prior
sums over the walks' first direction, averaged over the phase's next direction,
phi - 3 g F . dir, for each of the 9 harmonics and 64 bins, interpolated over the
leaf around the collision. The prior sums were a copy of the probes' sums, kept in two
buffers beside them (300 MB at 2^18 probes), their walk count in the uniforms, so that
the walks read them while adding to the live sums. The code was reverted. Against a
20000-round photon field, view 00's volume, 256 walks per probe:

| bake                                              | time   | sky error median | sun error median | sky total ratio |
| ------------------------------------------------- | ------ | ---------------- | ---------------- | --------------- |
| plain walks                                       | 13.6 s | 0.021            | 0.429            | 1.0001          |
| q 0.5 every collision, prior refreshed as walks double (128 walks) | 4.9 s | 0.116 | 0.897 | -1.97 |
| q 0.5 every collision, prior frozen after 64 plain walks, prediction clamped at 0 | 5.8 s | 0.090 | 0.621 | 1.0875 |
| q 0.25 at the first collision only, same prior    | 8.2 s  | 0.036            | 0.646            | 1.0002          |
| photons, 128 rounds                               | 3.4 s  | 0.022            | 0.381            | 1.0002          |

- **A prior refreshed from walks with control variates diverges.** Each refresh feeds
  the last one's noise into every neighbour's prediction; a few probes grow without
  bound and the totals turn negative.
- **Roulette at every collision grows the variance with depth**, as a walk's weight
  doubles per collision while its correction does not shrink.
- **Once at the first collision it is unbiased but loses**: 1.7 times faster than
  plain walks at 2.9 times their variance, about half their efficiency, and photons
  lead both. The prediction is too poor to pay: the sun's 64 bins from 64 walks are
  mostly noise, and P1 misses how a walk's value turns with its direction near the
  cloud's edge.
- **A converged prior does not save it.** With the 20000-round photon field as the
  prior, its sky transfer taking the bins' scattered sky light, 256 walks at q 0.25 take
  7.5 s and their split halves give a sun sigma of 0.999 and a sky sigma of 0.056,
  against plain walks' 0.574 and 0.030: per second, 0.60 and 0.53 times plain walks,
  where photons are 4.95 and 3.31 times. The residual is P1's error, not the prior's
  noise: two moments cannot predict the light the phase's peak picks out. And a walk
  that always stops at its first collision still pays its flight, its sun march and
  the prediction's 584 reads, so even a perfect prediction would leave walks short of
  photons.

## Photon connections

Each photon collision also sends M rays drawn from the phase, which add their path to
the probes at a finest cell's spacing out to a reach R, at the transmittance from the
collision that ratio tracking estimates against the density's bound; the photon's own
next part then adds only its path beyond R. In expectation the rays are the part of
that next part within R, so the field is the same, and every collision feeds the
probes around it in M directions. View 00's volume, 256 rounds, split-half sigma,
efficiency as 1 / (sigma^2 x seconds) against none:

| M | R (cells) | time   | sun sigma | sky sigma | efficiency sun | efficiency sky |
| - | --------- | ------ | --------- | --------- | -------------- | -------------- |
| 0 |           | 2.47 s | 0.343     | 0.0231    | 1.00           | 1.00           |
| 1 | 1         | 3.21 s | 0.295     | 0.0230    | 1.04           | 0.78           |
| 2 | 0.5       | 3.23 s | 0.284     | 0.0220    | 1.11           | 0.85           |
| 2 | 1         | 4.17 s | 0.252     | 0.0212    | 1.10           | 0.70           |
| 3 | 1         | 5.01 s | 0.232     | 0.0205    | 1.08           | 0.62           |
| 4 | 1         | 5.81 s | 0.221     | 0.0202    | 1.03           | 0.56           |
| 2 | 2         | 7.18 s | 0.217     | 0.0204    | 0.86           | 0.44           |
| 16 | 2        |        | 0.156     | 0.0179    |                |                |

- **Unbiased:** the sky's total against the 20000-round field stays within 0.05 % of 1,
  and the sun's ratio by brightness decile moves towards 1 as the noise falls.
- **The noise falls but the cost rises as fast.** Each ray costs about a third of a
  round, mostly its atomic deposits; sigma falls more slowly than the time grows, so
  efficiency peaks at 1.1 for the sun and only loses for the sky, whose noise is the
  light that never collides. The rays' light is local and correlated with the photon's
  own next part, so they add little a further round would not.
- **One deposit per ray at a uniform distance out to R**, to make the reach free,
  loses more: ratio tracking out to R then costs what the deposits saved, and one
  point per ray is noisier.
- Timings while the scratch volume was full ran 1.7 times slow; the table is from two
  later passes that agree within 2 %.
- The code was reverted. It had two bake settings, `photon_connections` (M) and
  `photon_connection_reach` (R in finest cells), and passed the volume's largest
  majorant to the shader as the ratio-tracking bound.

## Bake timings without the layout

`earth bake-probes` timed the probes' layout build with the bake until now, 0.5 to 2 s
of CPU time that varied from run to run. It now lays the probes out in a frame of its
own before the timer starts. The efficiency ratios of the two sections above carry
that time in both of their rows, so they understate the costs of the variants and
overstate their efficiency. At the corrected timing, 256 rounds of plain photons take
1.94 s.

## Guided photon emission

Photons enter the grid's box from a uniform direction at a uniform point of its
projected area. Guided emission draws a share of them instead from a distribution
learned over emission cells: a sun bin, a box face, and a cell of a 16 x 16 grid on
that face, 49152 cells. Each photon is weighted by uniform over the mixture's density.
This stays unbiased, as the uniform share covers every cell.

- **Learning.** Every photon scores its emission cell during the first 64 rounds. Its
  score is the sum over its deposits of the square of the deposit, at the uniform
  emission's weight, relative to the probe's sum per round in that bin. That relative
  amount is capped at one round's worth.
- **The level.** A bin's level is floored at a tenth of the probe's mean over bins.
  For light that has not collided, the level is the probe's sky fluence.
- **The distribution.** As the rounds double, each cell gets its uniform measure times
  sqrt(mean score / mean cost), with cost as 1 + collisions. This minimizes the
  probes' summed relative variance per unit time.

View 00's volume, efficiency against plain photons:

| guided share | rounds | time   | sun sigma | sky sigma | efficiency sun | efficiency sky | sun error | sky total |
| ------------ | ------ | ------ | --------- | --------- | -------------- | -------------- | --------- | --------- |
| 0            | 256    | 1.94 s | 0.344     | 0.0231    | 1.00           | 1.00           | 0.228     | 0.9997    |
| 0.8          | 256    | 2.29 s | 0.283     | 0.0199    | 1.26           | 1.15           | 0.186     | 0.9998    |
| 0.95         | 256    | 2.42 s | 0.273     | 0.0202    | 1.27           | 1.06           | 0.179     | 1.0001    |
| 0            | 1024   | 7.28 s | 0.126     | 0.0116    | 1.00           | 1.00           | 0.073     | 1.0000    |
| 0.8          | 1024   | 8.79 s | 0.104     | 0.0100    | 1.21           | 1.12           | 0.062     | 1.0001    |

- **It wins, but modestly.** The variance falls by a factor of 1.5, but guided photons
  cost about a fifth more each, as they go where the cloud is thick and bounce longer.
  Learning in every round, rather than the first 64, cost 5 % more time for the same
  noise.
- **Why the gain is small.** Within a bin, nearly every photon that reaches a dark
  probe has diffused through the cloud from its lit side. Where it enters changes
  little about where it ends up.

**Sweep, with the diffusion tail below on** (stop depth 16, p 0.5), 256 rounds:

| variant                          | time   | sun sigma | sky sigma | sun error |
| -------------------------------- | ------ | --------- | --------- | --------- |
| no guiding                       | 1.82 s | 0.137     | 0.0248    | 0.101     |
| share 0.8, learning 16 rounds    | 2.19 s | 0.116     | 0.0211    | 0.089     |
| share 0.8, learning 32 rounds    | 2.19 s | 0.116     | 0.0212    | 0.089     |
| share 0.8, learning 64 rounds    | 2.22 s | 0.116     | 0.0213    | 0.089     |
| share 0.8, learning 128 rounds   | 2.17 s | 0.116     | 0.0213    | 0.089     |
| share 0.8, learning 256 rounds   | 2.20 s | 0.116     | 0.0213    | 0.089     |
| share 0.5                        | 2.04 s | 0.122     | 0.0221    | 0.092     |
| share 0.95                       | 2.36 s | 0.113     | 0.0216    | 0.088     |
| level floor 0.01 instead of 0.1  | 2.17 s | 0.116     | 0.0213    | 0.089     |
| level floor 1 instead of 0.1     | 2.20 s | 0.116     | 0.0212    | 0.089     |

- **Nothing moves it.** Letting the field converge longer before the distribution is
  learned changes nothing: 16 rounds learn as good a distribution as 256. The share
  and the level floor only trade noise for time. Every variant lands within the
  timings' noise of 1.15 times as efficient.
- **The ceiling.** The cells' learned means under uniform emission (256 rounds, share
  1e-6) bound what any guided emission can win under this score. Without the cost,
  the ideal distribution would cut the variance by 1.90. With it, E[s] E[c] /
  E[sqrt(s c)]^2 is only 1.05: the cells whose photons score most are those that bounce
  longest. Over bins alone the bound is 1.00, over bins and faces 1.01.
- The measured 1.15 beats that bound because the score, the probes' summed relative
  variance, is not the median noise that the table measures. A score closer to the
  median could gain a little more, but not beyond the variance-only 1.9.
- The learning rounds became a setting, `photon_guiding_rounds`, now 32 by default.

## Diffusion tail for deep photons

Photons truncated after K collisions show where the light goes and where the time
goes. 256 rounds:

| K   | bake   | sky total | sun ratio, brightest decile |
| --- | ------ | --------- | --------------------------- |
| 16  | 0.82 s | 0.49      | 0.17                        |
| 64  | 1.08 s | 0.72      | 0.55                        |
| 256 | 1.34 s | 0.91      | 0.91                        |
| all | 1.94 s | 1.00      | 1.00                        |

The brightest probes lie in the dense core, where light bounces thousands of times.
Orders past 16 carry five sixths of their light, and those orders take 58 % of the
trace.

**The scheme.** A photon stops at a collision once `bench_sun_reach` there, the least
optical depth towards any sun bin, exceeds a stop depth. Its weight becomes a source in
its sun bin on a grid of 16-voxel cells (63 x 43 x 77). A cell-centred diffusion solve
then gives every bin's fluence from its sources:

- D = 1 / (3 sigma_t (1 - g)), with g = 0.819, the mean cosine of the photons' phase
  past the first scatter;
- harmonic-mean coupling between cells;
- Marshak faces where a cell borders the box or a cell thinner than an optical depth
  of 1.

Each probe takes phi and -D grad phi of the trilinear interpolant into its bins. This is
the transfer between probes: one stopped photon feeds every probe through a
deterministic solve, with none of the noise of its own remaining path. The solve was
first a prototype in Python (scipy's sparse LU, about 5 s on the CPU per field), left
out of the timings up to the GPU solver's table at the end of this section.

**Pure diffusion is biased low.** Stopping at depth 8, 256 rounds:

- the sun's noise falls from 0.344 to 0.109 in half the time, 19 times as efficient;
- but the sky's total falls to 0.90, and the sun's ratio by decile to 0.85 to 0.96;
- against the true tail (the reference less the stopped photons' own light), the
  diffusion tail carries 65 % of the light, 76 % even in the brightest decile.

The cloud is one to two transport mean free paths deep at its median, so leakage
through its surface sets the tail's level.

- **The extrapolation length** moves the tail from 0.65 of the truth to 0.87 when
  doubled, and to 1.28 when quadrupled, with the wrong shape over the deciles either
  way.
- **8-voxel cells** do worse, 0.36. Thin pockets inside the cloud turn into vacuum
  that light escapes into, where in truth it crosses them.
- **Stopping deeper** (12, 16) or later (16 collisions) only trades bias for gain.

**An unbiased correction cannot keep the gain.** Suppose a share of the stopped
photons goes on, and the solve is used as a control variate on their walks. Each
survivor's residual still holds the noise of its own path given its stop, and that
noise is the very noise that the solve removed. A biased correction is needed instead.

**Calibrated by survivors.** A share p of the stopped photons goes on regardless:

- each survivor adds its source to a second grid, and its fluence after the stop to a
  per-probe sum, kept per group of 4 x 4 neighbouring bins;
- a second solve gives the survivors' own diffusion response;
- each probe's tail in each group is the full solve's tail times the ratio of the
  survivors' walked fluence to their solved fluence;
- where the solve leaves a group dark that survivors did light, the survivors' own sum
  over p takes its place.

With the true tail as the ratio, the error floor that the solve's shape over bins
leaves is a sun error of 0.092 for one group per probe, 0.060 for four, and 0.033 for
sixteen. With survivors, sixteen groups are noisier than four (0.149 against 0.100 at
256 rounds), and smoothing the ratio over neighbouring cells costs more bias than it
saves noise.

Sun error is the median relative error against the 20000-round field. Stop depth 16,
p 0.5, four groups:

| bake                           | rounds | time   | sun sigma | sun error | sun error p90 | sky error | sky total |
| ------------------------------ | ------ | ------ | --------- | --------- | ------------- | --------- | --------- |
| photons                        | 64     | 0.59 s | 0.712     | 0.600     | 0.999         | 0.0313    | 0.9993    |
| diffusion tail                 | 64     | 0.68 s | 0.322     | 0.207     | 0.886         | 0.0340    | 0.9967    |
| diffusion tail, guided 0.8     | 64     | 0.68 s | 0.261     | 0.176     | 0.762         | 0.0297    | 0.9970    |
| photons                        | 256    | 2.05 s | 0.344     | 0.228     | 0.908         | 0.0157    | 0.9997    |
| diffusion tail                 | 256    | 1.84 s | 0.137     | 0.101     | 0.436         | 0.0175    | 0.9971    |
| diffusion tail, guided 0.8     | 256    | 2.10 s | 0.116     | 0.089     | 0.382         | 0.0153    | 0.9970    |
| photons                        | 512    | 3.71 s | 0.213     | 0.128     | 0.680         | 0.0112    | 0.9999    |
| photons                        | 1024   | 7.39 s | 0.126     | 0.073     | 0.402         | 0.0080    | 1.0000    |
| diffusion tail                 | 1024   | 6.41 s | 0.066     | 0.057     | 0.244         | 0.0092    | 0.9974    |
| diffusion tail, guided 0.8     | 1024   | 7.83 s | 0.057     | 0.051     | 0.224         | 0.0081    | 0.9973    |

- **The sun converges two to three times sooner.** The tail with guiding reaches at 256
  rounds the error that plain photons reach at about 800, and at 1024 rounds it beats
  them by 30 %, by 45 % at the 90th percentile.
- **Its bias is small.** The sun's ratio by decile stays within 1.1 % at 1024 rounds.
  The sky's total is 0.3 % low.
- **The sky alone gains nothing.** Its noise is mostly light that never collides, and
  the calibration ratios add a little noise of their own. Guiding makes up for that.
- **Shallower stops lose.** Stop depth 8 or 12 gains more early (0.183 at 64 rounds) but
  floors higher. A survival share of 0.25 saves time but is noisier, and loses at
  equal time.
- **The solve on the GPU.** Red-black SOR, overrelaxed by 1.8, over all bins and
  survivor groups at once. It is warm-started from the last solve, as the sources per
  round settle. Offline, from zero, the system converges to a relative error of 5e-5
  in 100 sweeps (PCG with a Jacobi preconditioner needs about the same; the
  coefficients vary by less than a factor of 7). Warm-started, 25 sweeps give the
  same field as 400 at 256 rounds, and lose 0.001 of sun error at 64 rounds. The bake
  uses 50, about 10 ms per solve, at each refit. The GPU's field matches the Python
  prototype's: sun error 0.101, p90 0.435, sky total 0.9971 at 256 rounds.

The same comparison with the GPU solve, its time included:

| bake                       | rounds | time   | sun sigma | sun error | sun error p90 | sky error | sky total |
| -------------------------- | ------ | ------ | --------- | --------- | ------------- | --------- | --------- |
| photons                    | 64     | 0.74 s | 0.712     | 0.600     | 0.999         | 0.0313    | 0.9993    |
| diffusion tail             | 64     | 0.51 s | 0.320     | 0.207     | 0.886         | 0.0340    | 0.9967    |
| diffusion tail, guided 0.8 | 64     | 0.61 s | 0.260     | 0.176     | 0.760         | 0.0298    | 0.9968    |
| photons                    | 1024   | 7.77 s | 0.126     | 0.073     | 0.402         | 0.0080    | 1.0000    |
| diffusion tail             | 1024   | 7.05 s | 0.066     | 0.057     | 0.244         | 0.0092    | 0.9975    |
| diffusion tail, guided 0.8 | 1024   | 8.49 s | 0.058     | 0.052     | 0.225         | 0.0081    | 0.9973    |

At equal time the sun is 7 to 9 times as efficient at 64 rounds, and 4 times at 1024.

**Broken in renders.** The field metrics above miss what the image shows. On
presets/wdas-cloud-0.zon (extinction 0.2), the approximation rendered from 2048 rounds
against one from 8192 rounds of plain photons:

| bake           | log2 error p90 | after a 15 px blur |
| -------------- | -------------- | ------------------ |
| photons        | 0.053          | 0.039              |
| guided 0.8     | 0.049          | 0.032              |
| diffusion tail | 0.135          | 0.120              |

- **Bands** on the shadowed side, a diffusion cell apart: the flux of the trilinear
  interpolant jumps at every cell face, and the march shows the flux.
- **Bright blobs:** each probe's calibration ratio rests on its own few survivors.
- **Bias** of up to a third in the shadowed interior, and the lower right too dark. The
  per-group calibration does not correct the solve's shape at this density.

The tail is therefore off by default, and the presets use guided emission only.

**State.** In the worktree, uncommitted, off unless `photon_stop_depth` is set:

- `trace_photons` stops the photons, keeps the survivors, and sums the sources;
- the `*_diffusion*` kernels of shaders/modes/cloud_bench_probes.slang set up and solve
  the system and add the calibrated tail into `Probes.sun_totals`, which the fit and
  the lighting read;
- a field with the tail holds it added. Its sources are not stored, so ProbeCache keeps
  no field with a tail, and its key marks one with a flag so that the file format stays
  as it is.

`python/validation/bake_sweep.py`, untracked, ran the sweeps above. For example:

```sh
pixi run -e py python python/validation/bake_sweep.py <scratch dir> <reference.eprb> 256 \
    photons: tail:photon_stop_depth=16,photon_survival=0.5 guided:photon_stop_depth=16,photon_survival=0.5,photon_guiding=0.8
```

## Splitting and Russian roulette of photons

Guided emission showed that where a photon enters says little about which dark probe
it lights; the choice that matters is made inside the cloud. A weight window now
follows each photon. At every collision its target weight is its emitted weight times
(the level of the nearest probe in its sun bin, over the level at its first collision)
to a power, `photon_splitting`. The level is the probe's sum per round, floored as the
guided score floors it.

- Above twice the target, the photon splits into copies, each scattering on its own,
  up to 8 waiting copies per thread.
- Below half, it survives Russian roulette at the target weight.
- Both keep each photon's expected deposits, so the bake stays unbiased whatever the
  levels are. The levels come from earlier deposits, independent of the photon's own
  future.

Photons thus multiply as they diffuse towards dim probes and thin out where the probes
are already bright. The exponent trades between equal absolute noise (0) and equal
relative noise (1).

View 00's volume, against the 20000-round field:

| bake                     | rounds | time    | sun sigma | sky sigma | eff. sun | eff. sky | sun error | p90   | sky total |
| ------------------------ | ------ | ------- | --------- | --------- | -------- | -------- | --------- | ----- | --------- |
| photons                  | 256    | 2.02 s  | 0.344     | 0.0231    | 1.00     | 1.00     | 0.228     | 0.908 | 0.9997    |
| splitting 0.5            | 256    | 3.27 s  | 0.195     | 0.0223    | 1.93     | 0.67     | 0.116     | 0.593 | 0.9998    |
| splitting 0.75           | 256    | 4.58 s  | 0.145     | 0.0225    | 2.48     | 0.47     | 0.092     | 0.387 | 0.9996    |
| guided 0.8               | 256    | 2.55 s  | 0.283     | 0.0198    | 1.17     | 1.08     | 0.186     | 0.836 | 0.9999    |
| guided, splitting 0.25   | 256    | 3.02 s  | 0.233     | 0.0197    | 1.45     | 0.92     | 0.142     | 0.722 | 0.9998    |
| guided, splitting 0.5    | 256    | 3.87 s  | 0.159     | 0.0191    | 2.42     | 0.76     | 0.097     | 0.483 | 0.9997    |
| guided, splitting 0.75   | 256    | 5.17 s  | 0.121     | 0.0193    | 3.17     | 0.56     | 0.078     | 0.320 | 0.9997    |
| guided, splitting 1      | 256    | 8.71 s  | 0.111     | 0.0212    | 2.24     | 0.27     | 0.075     | 0.262 | 0.9985    |
| photons                  | 1024   | 7.82 s  | 0.126     | 0.0116    | 1.00     | 1.00     | 0.073     | 0.402 | 1.0000    |
| guided, splitting 0.5    | 1024   | 13.80 s | 0.064     | 0.0096    | 2.21     | 0.83     | 0.042     | 0.172 | 0.9999    |
| guided, splitting 1      | 1024   | 28.34 s | 0.047     | 0.0097    | 1.96     | 0.39     | 0.035     | 0.117 | 0.9996    |

The guided rows' efficiencies are against plain photons, rescaled from the guided
baseline of their sweep.

- **The sun gains two to three times,** most at 0.75 with guiding, and its tail of
  dark probes most of all: the 90th percentile falls by more than the median.
- **The sky loses.** Its noise is mostly light that never collides, which splitting
  does not touch, while the split paths cost time.
- **No bias shows.** At 1024 rounds the sun's ratios by decile lie within 0.996 to
  1.003. At 256 rounds the strongest window reads 1 % low, shrinking with rounds, as
  the noise's skew does.
- **In the image** (presets/wdas-cloud-0.zon, extinction 0.2), guided splitting at 0.75
  and 1024 rounds against plain photons at 2048, both against 8192 rounds of plain
  photons: log2 error p90 0.037 against 0.053, and 0.021 against 0.039 after a 15 px
  blur, which shows the noise at low frequencies, at 38 s against 24 s. Its error is
  fine-grained, without the blotches of plain photons, and without any structure.
- A dense cloud costs splitting more: there each round took 3.2 times as long as plain
  photons, against 2.6 times on view 00.

**At 2048 rounds the gains shrink.** Efficiency is 1 / (sigma^2 x the bake's wall-clock
time), against plain photons; the errors are at equal rounds.

| bake                   | time    | sun sigma | sky sigma | eff. sun | eff. sky | sun error | p90   | sky total |
| ---------------------- | ------- | --------- | --------- | -------- | -------- | --------- | ----- | --------- |
| photons                | 15.41 s | 0.075     | 0.0082    | 1.00     | 1.00     | 0.046     | 0.225 | 1.0001    |
| guided 0.8             | 18.82 s | 0.063     | 0.0070    | 1.16     | 1.13     | 0.040     | 0.191 | 1.0000    |
| splitting 0.75         | 30.99 s | 0.041     | 0.0078    | 1.66     | 0.55     | 0.030     | 0.105 | 1.0000    |
| guided, splitting 0.5  | 26.96 s | 0.042     | 0.0067    | 1.79     | 0.85     | 0.030     | 0.113 | 0.9999    |
| guided, splitting 0.75 | 35.62 s | 0.036     | 0.0066    | 1.89     | 0.66     | 0.027     | 0.095 | 1.0000    |

- Guided splitting at 0.75 falls from 3.17 times as efficient on the sun at 256 rounds to
  1.89 at 2048, splitting alone from 2.48 to 1.66. Guiding alone holds its 1.16.
- The dark probes still gain most: the sun's 90th-percentile error at equal rounds falls
  from 0.225 to 0.095, and the decile ratios lose the skew plain photons keep in the
  dimmest tenth.
- The sky loses at equal time: guided splitting at 0.75 would reach a sky sigma of about
  0.0100 in plain photons' 15.4 s, against their 0.0082.

**State.** In the worktree, uncommitted; `photon_splitting` defaults to zero: it trades the sky for the sun (next section). The copies
wait on a per-thread stack, so a thread that splits runs longer than its warp; a queue
of copies served by later passes would even the work out.

## Cheaper rounds and the corners' noise

Splitting and guiding move photons between the sun's dim and bright bins. That cannot
help both the sun and the sky:

- **The sky's noise is scattered light.** At 2048 rounds of plain photons, light that
  never collided has a sigma of 0.0014, the scattered light 0.0070 (two thirds of the
  variance), though 77 % of the sky's light is scattered.
- **It is spread evenly.** Splitting by the probe's mean level instead of its bin's,
  which sends photons to dim probes rather than dim directions, leaves the sky's sigma
  at 0.0082. A randomly shifted Kronecker sequence over the photons' entry (direction,
  face, point) gains the sky 2 % and the sun nothing.

So the sky gains only from cheaper photons, or from less noise per photon at no cost.
Three ways were found and two kept. All figures are at 2048 rounds of view 00.

**Where a round's time goes.** At 512 rounds of plain photons, 4.0 s: without any
deposits 2.9 s, without the uncollided ones 3.6 s, without the scattered ones 3.7 s.
Tracing takes 72 %.

**Persistent threads (kept).** A round was a dispatch of one photon per probe, about
what the GPU holds at once, and photons' paths run from a few collisions to thousands.
Each round therefore ended in a long tail with most of the GPU idle, and each warp
waited on its longest photon. Now a dispatch traces all of a frame's rounds (up to the
next rebuild of the guided emission), and each thread takes photons from a counter
until they are done. The fields are bit-identical.

| plain photons                                     | time    |
| ------------------------------------------------- | ------- |
| a dispatch per round                              | 15.25 s |
| a dispatch per frame, a thread per photon (8 rounds per frame)   | 9.64 s  |
| the same, 32 rounds per frame                     | 8.46 s  |
| a dispatch per frame, persistent threads (8)      | 8.64 s  |
| persistent threads (32)                           | 7.48 s  |

Splitting gains more, as its copies no longer stall whole warps: guided splitting at
0.75 went from 35.6 s to 15.5 s.

**Scattered light to all eight corners (kept).** Each point along a path went to one
corner of its leaf, drawn by its trilinear weight. That draw was a large share of the
noise: depositing every point to all eight corners by weight cuts the sun's variance by
3.7 and the sky's by 2.5. It costs eight times the atomics, so:

- **Only scattered light** is spread over the corners. It takes four floats per corner,
  and holds most of the sky's noise. Light that has not collided feeds nine harmonics,
  36 floats per corner, and keeps its single drawn corner.
- **The corners' sums wait in registers** until the photon leaves the leaf, across its
  collisions: in the dense core a path's parts between collisions are mostly shorter
  than the points' spacing.

| bake (32 rounds per frame)                        | time    | sun sigma | sky sigma | sun error | p90   |
| ------------------------------------------------- | ------- | --------- | --------- | --------- | ----- |
| one drawn corner                                  | 8.04 s  | 0.075     | 0.0082    | 0.046     | 0.225 |
| all corners, every point flushed at once          | 21.62 s | 0.039     | 0.0052    | 0.028     | 0.102 |
| all corners for scattered light only              | 15.56 s | 0.039     | 0.0059    | 0.028     | 0.102 |
| the same, summed per leaf across collisions       | 14.45 s | 0.039     | 0.0059    | 0.028     | 0.102 |

- **Wider spacing loses.** Points two finest cells apart take 11.36 s but raise the
  sigmas to 0.046 and 0.0071; four apart, 9.14 s for 0.066 and 0.0097.
- **Exact lengths per leaf lose.** `leaf_lengths` with all corners takes 40.10 s, as the
  paths cross mostly fine leaves.

**The similarity relation (not kept).** Past K collisions, photons scattered
isotropically through the extinction times (1 - g), keeping the diffusion coefficient:
1.6 times faster, but at K = 32 the bright probes come out 2 % too bright, at K = 8 the
dense core 8 % too dark, at K = 4 20 %. Dropping only the phase's forward peak (delta-M)
leaves the core 7 to 12 % too dark at any K, as the peak's mean cosine is 0.91, not 1.

**Together.** The kept bake against the one before, plain photons, 2048 rounds:

| bake                                  | time    | sun sigma | sky sigma | eff. sun | eff. sky | sun error | p90   | sky total |
| ------------------------------------- | ------- | --------- | --------- | -------- | -------- | --------- | ----- | --------- |
| before                                | 15.25 s | 0.075     | 0.0082    | 1.00     | 1.00     | 0.046     | 0.225 | 1.0001    |
| now, 8 rounds per frame               | 15.85 s | 0.039     | 0.0059    | 3.56     | 1.86     | 0.028     | 0.102 | 1.0001    |
| now, 32 rounds per frame              | 14.52 s | 0.039     | 0.0059    | 3.88     | 2.03     | 0.028     | 0.102 | 1.0001    |
| now, guided 0.8                       | 19.34 s | 0.034     | 0.0052    | 3.84     | 1.96     | 0.026     | 0.093 | 1.0000    |
| now, splitting 0.5                    | 17.17 s | 0.029     | 0.0058    | 5.94     | 1.78     | 0.023     | 0.079 | 1.0001    |

The sun's ratios by decile lie within 0.997 to 1.005. Guiding no longer pays for its
longer photons, and splitting still trades the sky for the sun; both stay off.

In the image (presets/wdas-cloud-0.zon, extinction 0.2, 2048 rounds against 8192 of the
old bake): log2 error p90 0.037 against 0.053, after a 15 px blur 0.021 against 0.039,
in 19.8 s against 24.0 s, with fine-grained noise and none of the old bake's blotches.

**Code kept.** Only the persistent threads and the eight-corner deposits of scattered
light remain in the code, always on. The diffusion tail, the guided emission and
splitting were removed; the sections above record what they did. At 2048 rounds the
bake as kept takes 15.48 s at 8 rounds per frame, 14.21 s at 32, with a sun sigma of
0.039 and a sky sigma of 0.0059.

## Where the kept bake spends its time

View 00, 512 rounds, 8 per frame: 4.04 s. Without the eight-corner deposits 1.81 s,
without the uncollided sky's deposits 3.84 s, with no deposits at all 1.69 s. The
corners' deposits took 55 % of the bake.

**What the corners cost.** Over 8 rounds the photons deposit 35.6 million points and
change leaf at 22.6 million of them, 70 % in finest leaves: in the dense core a path
steps from one finest cell to the next. Each change adds about five corners to the
probes, 114 million atomic additions of four floats.

- **Not the number of atomics.** Spreading only phi over the corners and adding the
  flux at one drawn corner halves them, but saves 5 % (3.86 s). Keeping the sums of the
  corners the next leaf shares saves 2.5 % (3.94 s).
- **Not registers.** A per-thread cache of 16 to 64 probes, which a diffusing photon
  might revisit, spills to local memory: 6.2 to 16.5 s.
- **Where they land.** A probe's 64 bins lay in one kilobyte, so a leaf's eight corners
  fell on eight cache lines. Probes are numbered in the octree's depth-first order, so
  neighbouring probes have nearby numbers. Laid out bin-major, a photon's deposits,
  all in one bin, fall on neighbouring addresses.

**Kept: bin-major sun bins.** The GPU's buffer is bin-major; ProbeCache transposes it
to the files' probe-major order on save and back on load, so the files are unchanged.
At 2048 rounds the bake takes 12.68 s against 15.48 s, with the same field.

**Kept: longer dispatches offline.** `bake-probes` traces at least 32 rounds per
dispatch, as no viewer waits on it: 2048 rounds take 11.38 s at 32, 11.25 s at 64,
11.21 s at 128.

**The tracing.** Over 8 rounds: 63.5 million segments between collisions, 61.5 million
real collisions out of 88.1 million tentative ones (30 % null), and 299.5 million steps
through majorant cells of 8 voxels, 4.7 per segment. Walking coarser majorant cells,
16 or 32 voxels, is slower (3.06 s, 3.38 s, against 2.99 s at 512 rounds): the null
collisions in the cloud cost more than the steps saved. Only a walk that steps coarsely
through empty space alone could gain, at most about a tenth.

**Together.** 2048 rounds of view 00 now take 11.4 s against 15.25 s before the
persistent threads, with a sun sigma of 0.039 against 0.075 and a sky sigma of 0.0059
against 0.0082: the sun 5.0 times as efficient, the sky 2.6 times.

## The probe layout on the GPU

Before a bake can start, the probes are laid out. On the CPU, opening the bench built
the volume's occupancy grid in 3.5 s, one thread over all 823 million voxels, and the
adaptive layout's search took 314 ms: three trial builds of 60 to 90 ms of cell depths
(three passes over the occupancy grid), 3 ms of pyramid and 35 ms of tree each.

src/render/LayoutDepths.zig now marks the occupancy on the GPU from the density
already there, and measures each trial's cell depths there too. The host still decides
which occupancy texels each cell overlaps, so the depths, and the layouts, are
bit-identical to the CPU's: a test compares both at four cell sizes, and the fields
match the 20000-round reference baked on the old layout. The tree numbers its probes
through an array over the lattice instead of a hash map, in the same order.

The layout now takes 55 ms (per trial: 10 ms of depths, 3 ms of pyramid, 10 ms of
tree), and `earth bake-probes` starts its bake 3.5 s sooner: the whole process takes
3.5 s for 8 rounds, against 7.5 s.
