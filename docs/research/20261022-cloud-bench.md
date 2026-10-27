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
cloud's base sat about 1 km up then, and it scatters by the droplet phase.
`presets/wdas-cloud.zon` keeps the Mitsuba camera and sun. Scored at 333 x 180,
512 spp:

| run                 | mean  | dark  | mid   | bright | log-rms |
| ------------------- | ----- | ----- | ----- | ------ | ------- |
| single scattering   | 1.015 | 1.065 | 1.014 | 1.014  | 0.496   |
| multiple scattering | 0.215 | 0.153 | 0.209 | 0.229  | 4.28    |

The single-scattering log-rms is dominated by the reference's noise. Light scattered
once is a faint backscatter in this view.

## Approximation, real cumulus

Since then the cloud is 10 m per volume unit: 2.8 km tall, 4 x 5 km across. Every
grid's base lies where the WDAS half cloud's centre would be 1.8 km up (since
2026-10-26; 1 km base before), with the presets' cameras 1.6 km up. Its extinction is set per metre, 0.1 /m at unit density by default,
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

**After the bin-major bins.** View 00, 512 rounds, 32 per dispatch: 2.90 s. Without
the corners' deposits 1.38 s, without the single scattering sky's 2.64 s, with no
deposits at all 1.16 s: the tracing is 40 % of the bake, the corners' deposits still
half.

- **Their atomics** take 0.78 s: with the flushes' additions skipped, 2.12 s. Adding
  only phi, one float of four, 2.41 s.
- **Not registers.** The trace kernel holds 120 to 128 registers (its CUDA build, a
  proxy for the driver's), a third of the SM's threads. With the corners' sums in
  groupshared memory it holds 88, and the bake takes the same 2.90 s.
- **Not the flux's floats.** Adding phi to all eight corners and the flux vector to one
  drawn by its weight, unbiased, takes 3.17 s: a corner's cache line, once touched,
  takes its four floats at little more cost, and the draw and its branch cost more.

What the corners cost is a scattered line in L2 per corner each time a photon changes
leaf, which in the finest leaves is at nearly every point.

**Photons per sun bin by the bins' noise.** At 2048 rounds of view 00 the bins'
median relative sigma lies between 0.039 and 0.061. Photons per bin in proportion to
it would lower the sun's variance by 1 % (8 % by the mean); the sky's best allocation
depends on the sky, which the bake does not know, and costs the sun 18 %.

## Denoising the probe field by its own noise

An offline prototype averaged each probe with the probes it shares a leaf with,
weighted by how far apart their 64 bins lie in units of their noise: a Poisson
variance per bin, its scale per probe from the two halves' disagreement. The sky's
harmonics and the flux took the same weights. View 00, 2048 rounds, against the
20000-round field; h is the weight's width in units of the expected distance:

| h, passes     | sun error | p90    | sky error | flux error | sun ratio by decile |
| ------------- | --------- | ------ | --------- | ---------- | ------------------- |
| none          | 0.0280    | 0.1012 | 0.0042    | 0.0047     | 0.997 to 1.004      |
| 1, 1          | 0.0244    | 0.0839 | 0.0036    | 0.0044     | 0.997 to 1.001      |
| 2, 1          | 0.0244    | 0.0796 | 0.0037    | 0.0059     | 0.997 to 1.005      |
| 4, 1          | 0.0272    | 0.0884 | 0.0043    | 0.0087     | 0.992 to 1.017      |
| plain mean, 1 | 0.0336    | 0.1180 | 0.0048    | 0.0161     | 0.960 to 1.072      |
| 2, 3          | 0.0318    | 0.0999 | 0.0053    | 0.0092     | 0.993 to 1.008      |

- **Neighbours differ by more than the noise.** A plain mean over the leaves' corners
  is worse than none: at 2048 rounds the field already varies between neighbouring
  probes more than its noise, so there is little left to average.
- **The renders show bias first.** wdas-cloud-0 against the converged 8192-walk
  render, |log2| error p90 (after a 15-pixel blur, which bias survives): none 0.0383
  (0.0219), h 1 0.0373 (0.0217), h 2 0.0394 (0.0270). The setting best on the field
  shifts areas' brightness; the mild one gains 3 %.
- No code was kept.

## Where the picture's error comes from

The bake's noise against the approximation's own error, on wdas-cloud-0 (extinction
0.2) at 512 x 512: the approximation at 64 spp from fields of 256 to 8192 rounds,
against the path-traced reference at 2048 spp and against the 8192-round field's
render. |log2| error over the pixels the reference lights, raw and after a 15-pixel
blur, which bake noise and render noise survive less than bias:

| rounds | vs reference, raw p90 | blurred p90 | vs 8192 rounds, raw p90 | blurred p90 |
| ------ | --------------------- | ----------- | ----------------------- | ----------- |
| 256    | 0.383                 | 0.291       | 0.127                   | 0.094       |
| 512    | 0.368                 | 0.279       | 0.080                   | 0.056       |
| 1024   | 0.360                 | 0.270       | 0.051                   | 0.030       |
| 2048   | 0.359                 | 0.271       | 0.038                   | 0.022       |
| 8192   | 0.357                 | 0.269       |                         |             |

Over the cloud's pixels alone (where the field's noise moves the picture, 127k of
262k), the 8192-round render is off the reference by a median of 0.159 (0.100
blurred) and a p90 of 0.521 (0.413): the sunlit face too dark (brightest quintile
-0.137 blurred), the shadowed body too bright (+0.07), the rim both ways; the mean
over the cloud 2.8 % dark. **The bake's noise at 1024 rounds is a tenth of the
approximation's error, at 2048 a twentieth.** Rounds past 1024 change nothing the
reference would show; the approximation's error is in its model, not its photons.

**The sun's fit loses the bins' accuracy.** The order-3 log-SH fit of the converged
field's 64 bins, against those bins at each probe's brightest bin: |error| median
0.109, p90 0.273, signed p10 -0.233 to p90 +0.168. Its noise at 2048 rounds is as
large as the single brightest bin's (0.025 against 0.026), where a fit over 64 bins
could have half: the dim bins' large log noise spreads through the kernel. A fit
weighted by each bin's photon count (relative weight of the bin's phi, floor 0.01),
exact real harmonics, same converged bins:

| fit               | brightest bin, median | p90   | signed p10 / p90 | bins >= 0.1 of brightest | noise at 2048 rounds |
| ----------------- | --------------------- | ----- | ---------------- | ------------------------ | -------------------- |
| order 3 plain     | 0.109                 | 0.273 | -0.233 / +0.168  | 0.105                    | 0.025                |
| order 3 weighted  | 0.044                 | 0.115 | -0.107 / +0.058  | 0.061                    | 0.015                |
| order 4 plain     | 0.070                 | 0.205 | -0.174 / +0.113  | 0.074                    |                      |
| order 4 weighted  | 0.028                 | 0.068 | -0.059 / +0.038  | 0.039                    |                      |
| order 5 plain     | 0.081                 | 0.256 | -0.103 / +0.249  | 0.068                    |                      |
| order 5 weighted  | 0.018                 | 0.055 | -0.035 / +0.045  | 0.027                    |                      |

The plain fits are the shader's quadrature of log phi over the 64 bin centres, which
aliases past order 4; the weighted ones solve the normal equations per probe. The
weighted order-3 fit keeps the 16 floats the renderer reads, matches the bright
directions 2.5 times as well, and has 2.7 times less noise variance from the same
bake, at the cost of the dim directions, which it fits worse. 
**Built as a `probe_sun_weighted` option, then removed: it renders worse.** wdas-cloud-0
from the same 2048-round field, blurred |log2| error against the reference over the
cloud's pixels: plain fit median 0.101, p90 0.416; weighted 0.144, 0.469. By the
reference's brightness quintile, dark to bright, plain -0.037 +0.009 +0.070 +0.078
-0.131, weighted +0.095 +0.094 +0.143 +0.059 -0.219: the weighted fit brightens the
shadowed body and darkens the sunlit face further. Offline on that field at the bench's
sun direction, the plain fit sits at 1.00 of the sun's bin for the brightest fifth of
the probes and at 0.73 for the dimmest; the weighted fit at 0.97 and 1.15. So the fit
is not what darkens the sunlit face, and in the shadow the plain fit's under-estimate
had been cancelling an over-estimate from elsewhere in the approximation: the closure,
the forward peak, the probes' interpolation, or the sky's harmonics. A fit truer to the
bins exposes that error rather than fixing it. Attributing it needs renders of the
approximation's variants (closure, first scatter, sun only) against references of
their own. The option was taken out again (a weighted least squares solve per probe
in `fit_probes`, a flag through the config and panel); this record stays.

## Reciprocal photon paths

A photon's path, reversed, is a path of light entering where the photon left, with the
same probability density but for the ratio of the box's projected areas at the two
directions, which the weight absorbs: the reversed path's sample weight comes out
equal to the forward one's. So each path can add its deposits twice, half to the bin
of its entry direction as it goes and half to the bin of its exit direction once it has
left, with its flux turned around; the segment before its first collision is scattered
light in reverse, the segment after its last the sky's. The exit direction is unknown
until the end, so each thread kept records of its leaves' corner sums (slot, collisions
before the leaf's first point, eight float4s) in a buffer of its own, and replayed
them at the path's end; the leaf of the last collision holds light from both sides of
it, whose later side is walked again from the state saved at the collision and moved to
the sky. A path with more leaves than the records hold, or that ran out of bounces,
adds its forward deposits again instead, which is unbiased as reversal keeps a path's
length. It was unbiased: sun ratio by decile 0.997 to 1.004, sky total 1.0000.

View 00, 2048 rounds, against the kept bake's 11.37 s, sun sigma 0.039, sky 0.0059:

| records per thread | time    | sun sigma | sky sigma | efficiency sun | sky  |
| ------------------ | ------- | --------- | --------- | -------------- | ---- |
| 32                 | 18.70 s | 0.037     | 0.0055    | 0.68           | 0.70 |
| 128                | 21.92 s | 0.031     | 0.0053    | 0.82           | 0.64 |

- **The light is in long paths.** 32 records reverse a tenth of it, 128 three quarters:
  the photons that deposit most diffuse through the fine shell changing leaf at nearly
  every point. Full coverage would need thousands of records per thread.
- **The reverse costs what the deposits cost.** Its replay repeats every atomic of the
  forward pass, and the deposits are over half the bake; the tracing it would share is
  the smaller part. Even with every path reversed for free, halving the variance for
  1.55 times the time gains 1.3.
- The first version replayed overflowing records inside the hot loop and indexed the
  corner sums by the drawn corner, which put them on the stack: 3.2 times slower. Keeping
  the sums in registers and all replays at the path's end left 1.63 times.
- Reverted; the patch is kept outside the repository.

**The sky alone** could keep the cheap part: the last segment reversed feeds the sky's
transfer from the exit direction, and only the first and last segments' leaves need
records. That halves the sky's single scattering noise, a third of its total, for a few
records per path; a direct estimate of that transfer removes it instead.

## The sky's single scattering transfer, estimated directly

The photons' light before its first collision fed the sky's transfer through one drawn
corner per point, 36 atomic adds each, and was a third of the sky's noise. `bake_sky_transfer`
now estimates that transfer per probe: each round, the transmittance from the probe along
one direction by ratio tracking, 4 pi times its harmonics into the sums, with no
atomics as a probe owns its sums. The directions are a Fibonacci set of 1024 in a
scrambled order, rotated at random per probe and set, so that consecutive rounds
stratify the sphere while any count of them is unbiased. The photons add nothing
before their first collision.

View 00, 2048 rounds: 10.56 s against 11.37 s, sun sigma 0.039 as before, sky sigma
0.0048 against 0.0059, so the sky 1.6 times as efficient and the sun 1.07. The sky's
error against the 20000-round photon field rose from 0.0042 to 0.0093 with the total
at 1.0033: the photons' deposits estimate the fluence averaged over the probe's
trilinear kernel, the transmittance is the value at the probe. Against the path-traced
reference of wdas-cloud-0 the render is slightly closer (blurred |log2| p90 over the
cloud's pixels 0.391 against 0.416), so the point value is at least as good a thing to
interpolate, and it is kept.

## What the approximation's error is made of

wdas-cloud-0 at extinction 0.2, the approximation at 64 spp from a 2048-round field
against path-traced references at 2048 spp, |log2| error over the cloud's pixels after
a 15-pixel blur, and its sign by the reference's brightness quintile, darkest first.

**The sun's light carries the error.** Sun only, against a sun-only reference: median
0.381, p90 1.116, quintiles +0.33 +0.29 +0.17 +0.22 -0.13; the shadowed body is a
quarter too bright, the sunlit bulge a tenth too dark. Sky only, with the sun set,
against its own reference: p90 0.150, the total 13 % bright. With both, the sky's light
dilutes the sun's error to the 0.39 p90 of the earlier section.

| sun only, variant                                   | blurred median | p90   | quintiles                     |
| --------------------------------------------------- | -------------- | ----- | ----------------------------- |
| as kept                                             | 0.381          | 1.116 | +0.33 +0.29 +0.17 +0.22 -0.13 |
| probe depth ratio 1                                 | 0.383          | 1.122 | +0.35 +0.30 +0.18 +0.22 -0.14 |
| probe budget 1M                                     | 0.372          | 1.120 | +0.35 +0.26 +0.12 +0.17 -0.10 |
| view steps 0.1                                      | 0.384          | 1.118 | +0.33 +0.30 +0.18 +0.23 -0.13 |
| HG lobe closure                                     | 0.399          | 1.130 | +0.30 +0.27 +0.17 +0.24 -0.08 |
| photons from the exact sun direction, read straight | 0.175          | 0.324 | +0.11 +0.18 +0.20 +0.18 -0.12 |
| the same with the HG lobe                           | 0.187          | 0.339 | +0.05 +0.17 +0.20 +0.19 -0.06 |

With the sky, the variants of the first-scatter mode: probes 0.426, forward peak 0.391,
gather 0.375, forward peak and gather 0.347; P1 against the von Mises-Fisher closure
0.389 against 0.391; sun fit order 2 against 3: 0.595 against 0.391, its sunlit quintile
-0.38 against -0.13.

- **The sun transfer's angular resolution is most of it.** A bin averages the sun over
  about 14 degrees and the order-3 fit sees only those averages; photons sent from the
  exact sun direction into one bin, read without the fit, take the p90 from 1.12 to
  0.32 and the shadow's excess from +0.33 to +0.11. The step from order 2 to 3 had cut
  the sunlit face's deficit from -0.38 to -0.13, the same lever.
- **The closure's tails are the sunlit face's remaining deficit.** A von Mises-Fisher
  lobe with HG(0.6)'s mean cosine has a quarter of its backscatter; an HG-shaped lobe of
  the same mean cosine halves the deficit, -0.13 to -0.08, and with the exact direction
  -0.12 to -0.06, and leaves the rest as it was. P1 clamps that light to zero and is
  worse there.
- **Not the probes' resolution nor the march.** Finer or more probes and finer view steps
  move the quintiles by 0.05 at most.
- A residual of +0.1 to +0.2 in the half-lit quintiles survives the exact direction and
  the HG lobe: a single lobe about the flux for light that is partly a beam from the lit
  side and partly diffuse, or the forward peak's treatment, which the gather modes ease.

### Finer bins, bins read directly, the lobe's shape and the gather

The same measure, on the levers the attribution pointed at. Finer bins were a build with
16 x 16 bins (a 1.09 GB field for the 262144-probe budget, the bake 3 % slower per
round); the direct reads interpolate the four bins around the sun direction in
`light_probes` instead of evaluating the fit; the exact-direction fields send every
photon, or every walk's next-event estimate, from the sun's direction.

| sun only, variant                                       | blurred median | p90   | quintiles                     |
| ------------------------------------------------------- | -------------- | ----- | ----------------------------- |
| as kept, 8 x 8 bins, order-3 fit                        | 0.381          | 1.116 | +0.33 +0.29 +0.17 +0.22 -0.13 |
| 16 x 16 bins, 2048 rounds                               | 0.344          | 1.365 | -0.55 -0.02 +0.05 +0.16 -0.14 |
| 16 x 16 bins, 8192 rounds                               | 0.332          | 1.074 | +0.06 +0.12 +0.08 +0.17 -0.19 |
| 8 x 8 bins read bilinearly, linear                      | 0.419          | 2.021 | +1.02 +0.59 +0.43 +0.38 -0.21 |
| 8 x 8 bins read bilinearly, in the log                  | 0.311          | 0.990 | +0.52 +0.44 +0.29 +0.17 -0.29 |
| photons from the exact direction                        | 0.174          | 0.322 | +0.10 +0.18 +0.20 +0.18 -0.12 |
| the same, gather                                        | 0.154          | 0.299 | +0.08 +0.16 +0.18 +0.16 -0.08 |
| the same, forward peak and gather                       | 0.139          | 0.256 | +0.09 +0.14 +0.16 +0.14 -0.10 |
| the same, forward peak and gather, HG lobe              | 0.142          | 0.269 | +0.07 +0.14 +0.17 +0.14 -0.07 |
| walks with next-event estimates to the exact direction  | 0.158          | 0.399 | +0.08 +0.11 +0.13 +0.13 -0.08 |

With the sky: 16 x 16 bins at 8192 rounds 0.393 against 0.391; the HG lobe 0.395, and
0.350 with the forward peak and gather against 0.347; the log read of the bins 0.345,
but its sunlit quintile -0.27 against -0.13.

- **Finer bins trade the shadow for the lit face.** At equal rounds a bin holds 8
  photons, the fit's floor catches the empty ones and the shadow goes dark by half; at
  four times the rounds the shadow's excess falls from +0.33 to +0.06, but the order-3
  fit undershoots the sharper peak and the lit face loses another 6 %. The whole picture
  does not move, for four times the memory and the rounds.
- **The bins read directly are worse than the fit.** A shadowed probe's bin averages
  lit directions in with the sun's, which a linear read passes on whole (+1.0 in the
  shadow); in the log the read is sharper than the fit but 32 photons' log is biased low
  on the lit face (-0.29) and the picture darkens by 11 %.
- **The HG lobe moves the lit face only**, -0.13 to -0.10 with the sky, and the picture
  as a whole not at all.
- **The gather is worth 0.05 to 0.07 of p90 wherever it is applied**, with the bins
  (0.39 to 0.35) and without them (0.32 to 0.26): it reads the light one scatter further
  on, where the closure's and the interpolation's errors are smaller, and is not what
  stands in for the bins' angular resolution.
- **The splat is a third of the residual.** Walks, which estimate the probe's own point
  without the deposits' trilinear kernel, leave +0.08 to +0.13 in the half-lit quintiles
  against the photons' +0.10 to +0.20; the rest is the closure and the march's
  interpolation.
- A bake aware of the sun, which the exact-direction rows measure, would also spend no
  photons on the 63 bins the sun is not in: its sun field at 32 rounds is as converged as
  the kept one at 2048.

## The sun fit past order 3

The fit projects the bins' log fluence onto the harmonics by their quadrature, so any
order evaluates at the sun direction as the bins' log fluence weighted by the truncated
reproducing kernel, sum over l of (2l + 1) / 4 pi P_l(cos) to the order, with no basis
functions and no stored coefficients. A variant of `light_probes` did that on the kept
field; order 3 reproduced the kept picture exactly. Sun only and with the sky, blurred
|log2| error and the sun-only quintiles:

| order, 8 x 8 bins, 2048 rounds | sun median | sun p90 | quintiles                     | with sky p90 |
| ------------------------------ | ---------- | ------- | ----------------------------- | ------------ |
| 3 (kept)                       | 0.381      | 1.116   | +0.33 +0.29 +0.17 +0.22 -0.13 | 0.391        |
| 4                              | 0.349      | 1.037   | +0.13 +0.25 +0.18 +0.17 -0.28 | 0.390        |
| 5                              | 0.264      | 1.004   | +0.09 +0.14 +0.17 +0.24 -0.12 | 0.328        |
| 6                              | 0.249      | 1.087   | +0.21 +0.15 +0.20 +0.27 -0.07 | 0.311        |
| 7                              | 0.396      | 1.427   | +0.77 +0.60 +0.49 +0.40 -0.13 | 0.433        |
| 3, 512 rounds                  | 0.383      | 1.153   | +0.23 +0.28 +0.16 +0.21 -0.12 | 0.398        |
| 5, 512 rounds                  | 0.277      | 1.046   | +0.09 +0.14 +0.16 +0.24 -0.12 | 0.337        |
| 6, 512 rounds                  | 0.263      | 1.148   | +0.27 +0.17 +0.20 +0.28 -0.07 | 0.326        |
| 16 x 16 bins, 8192 rounds, 3   | 0.332      | 1.074   | +0.06 +0.12 +0.08 +0.17 -0.19 | 0.393        |
| 16 x 16 bins, 8192 rounds, 6   | 0.259      | 0.928   | +0.29 +0.28 +0.26 +0.24 -0.05 | 0.290        |
| 16 x 16 bins, 8192 rounds, 10  | 0.259      | 0.813   | +0.27 +0.28 +0.27 +0.28 -0.15 | 0.286        |

- **Order 5 is the step.** It takes the shadow's excess from +0.33 to +0.09 and the
  picture's p90 from 0.39 to 0.33, past the gather's 0.35, and holds at 512 rounds,
  where it still beats order 3 at 2048. Order 6 gains the lit face and loses the
  shadow to the bins' noise; order 7 is 64 coefficients from 64 bins and falls apart.
  Even orders overshoot the lit face (Gibbs on the log).
- 16 x 16 bins carry order 10 and the best sky picture, 0.29, but no better shadow
  than 8 x 8 at order 5, for four times the memory and the rounds.
- The kernel form needs neither a basis nor the fit's storage: `light_probes` can take
  the order from the bins directly, once per lighting change.

Applied: `fit_probes` fits order 5 and stores it in 52 bytes a probe, the constant
harmonic a float and the 35 others, with the flux ratio's 12, a byte each over the
field's range of that coefficient (`range_probes`, a wave reduction and two atomics per
coefficient per wave, before the fit; the next section measures the quantisation as
lossless). The pictures match the float variant to the third digit; a 2048-round bake,
which refits as it goes, takes the same time as before within its noise, and the render
reads the atlas as before. The order setting offers 3 and 5, a refit from the same
bins, in the same bytes.

## Bases other than harmonics for the sun transfer

What the sun transfer needs is one function on the sphere per probe, the log fluence
against the sun's direction, with a terminator, evaluated at one direction per lighting
change and stored lighting-agnostic. The recent literature on directional bases was
read against that need, and the candidates that fit it were measured on the field: each
probe's 64 bins fitted offline, scored at the sun's direction against the exact-direction
field (fluence-weighted RMS of the log error, 2048 rounds; 512 in brackets), and the
promising ones rendered.

| encoding of the sun's log fluence                 | bytes | wRMS          | p90           |
| ------------------------------------------------- | ----- | ------------- | ------------- |
| SH order 3, as kept                               | 64    | 0.228 (0.276) | 0.554 (0.685) |
| SH order 5                                        | 144   | 0.144 (0.205) | 0.381 (0.584) |
| SH order 5, coefficients at 8 bits, DC a float    | 40    | 0.144 (0.205) | 0.380 (0.584) |
| SH order 5 at 6 bits                              | 31    | 0.149 (0.210) | 0.385 (0.588) |
| SH order 5 at 4 bits                              | 22    | 0.211 (0.279) | 0.462 (0.662) |
| SH 3 + zonal l = 4..6 on the gradient axis (ZH3)  | 76    | 0.230 (0.284) | 0.530 (0.679) |
| PCA of the order-5 coefficients, 16 per probe     | 64    | 0.154 (0.217) | 0.434 (0.606) |
| clustered PCA of the bins, 32 clusters, 12        | 52    | 0.145 (0.196) | 0.412 (0.535) |
| the 64 log bins at 8 bits, read bilinearly        | 64    | 0.106 (0.128) | 0.374 (0.506) |

Rendered, sun only, blurred p90 and quintiles (with the sky, p90): order 5 1.004,
+0.09 +0.14 +0.17 +0.24 -0.12 (0.328); PCA 16 of order 5 1.034, +0.05 -0.00 +0.05
+0.18 -0.10 (0.406, a heavier tail than the kept 0.391); the bins read bilinearly with
the fitted flux 1.183, +0.63 +0.43 +0.28 +0.22 -0.26 (0.347).

- **Order 5 at 8 bits is the answer to "we cannot afford the order".** The quantisation
  costs nothing measurable at 8 bits and 2 % at 6, and 40 bytes is less than the kept
  16 floats. The ranges come from the field's own coefficients, stored once per field.
- **ZH3's zonal extension does nothing here**: the transfer is not zonal about its
  gradient, so the extra coefficients along that axis carry no energy.
- **A learned basis works offline and in the sun's picture but not in the whole picture.**
  16 PCA coefficients of the order-5 fit reach order 5's RMS, and the sun-only
  quintiles, but a global basis misses a few probes' shapes and the picture's p90 tail
  grows past the kept fit. Clustered bases might mend that; untested in the render.
- **The bins read directly are the best per byte and the worst on the faces.** Per probe
  they beat every fit, but a bin averages the sun over 22 degrees, which the fits
  extrapolate through and a read cannot: +0.6 in the shadow and -0.26 on the lit face.
  The offline fluence-weighted score, dominated by interior probes, does not see the
  surface probes that make the picture, so it misranks this; the render decides.
- Lobe families (spherical Gaussians and their anisotropic, Beta, Gabor and Voronoi
  variants, Beyond Spherical Harmonics, arXiv 2606.09794; Spherical Voronoi, arXiv
  2512.14180) are built for peaks, fitted by gradient descent, and overfit sparse data;
  the transfer is a terminator in the log, not a peak, and the fit would run per probe
  per bake. Not measured.
- Maximum-entropy moment encodings (arXiv 2608.20429) and ZH3 (I3D 2024) target
  positive irradiance from l <= 2 moments; their counterpart here is the closure, not
  the sun transfer.
- Neural volumes (Neural Irradiance Volume, arXiv 2602.12949; Neural Radiance Transfer
  Fields, 2022) beat SH2 probes tenfold per byte, but decode in 0.2 to 1.4 ms per frame
  of queries, bake in minutes, and lose capacity to every lighting parameter they take as
  input; the sun's direction would be one.
- Not a basis but a domain: on Earth the sun of a day moves along one arc, so bins along
  that arc, 64 of them over 180 degrees, would be 2.8 degrees wide at the memory of the
  8 x 8 sphere. That gives up the agnostic sphere for the agnostic day.

### The sky's order and its storage

The sky's light is linear in its harmonics, so the single scattering sums and the bins'
read contract into one SH_COUNT x 4 matrix per probe at fit time, and the bins leave the
runtime. Stored as the sun's fit is, the fluence's constant term in float and every
other entry as a byte, the matrix needs no field ranges: the transfer is a positive
kernel with |F| <= phi, so an entry over the constant term lies within
sqrt(2l + 1). Measured at 2048 rounds as variants of `light_probes` (order 3 through a
build with 16 sky harmonics in `bake_sun`, `bake_sky_transfer` and the sums):

| sky | storage | sky only p90 | brightest quintile | sun + sky p90 |
| --- | --- | --- | --- | --- |
| order 2 | float, 144 B + bins | 0.150 | +0.108 | 0.328 |
| order 2 | 8 bit, 40 B | 0.149 | +0.107 | 0.328 |
| order 3 | float, 256 B + bins | 0.116 | +0.089 | 0.324 |
| order 3 | 8 bit, 68 B | 0.116 | +0.089 | 0.324 |

The bytes cost nothing at either order. Order 3 takes the sky-only error from 0.150 to
0.116 and the overshoot of the brightest pixels from 11% to 9%: at the sunset of the
test, the horizon's warm band lights the cloud's underside, which order 2 smears into
the sky's mean and order 3 resolves (the picture shows the underside's colour). The
bake time is unchanged (13.2 s both), and the sun+sky picture moves little since the
sun's error dominates it.

Applied: the sky's radiance, its transfer bake and the sums carry 16 harmonics
(`sh_basis<3>`, one recurrence for both orders), and `fit_probes` contracts the sums
and bins into a `SkyTransfer` of 68 bytes beside the sun's 52, so that `light_probes`
reads 120 bytes a probe and the bins and sums only at the fit. The cache's format is
4. The pictures match the variant to the third digit.

The march scatters the probes' light (phi, F) towards the camera through the phase's
broad part by a von Mises-Fisher lobe about F whose mean cosine is g |F| / phi: the
light is taken as a vMF lobe, and its convolution with the phase as another. Both halves
were tested apart, on the exact-direction field so that the bins' smear does not hide
them, sun only, blurred |log2| error against the path-traced reference and its sign by
brightness quintile. The exact convolution of a lobe with the phase is the Funk-Hecke
product of their Legendre coefficients; the phase's were tabulated to order 48, the
lobe's are ratios of modified Bessel functions by Gauss's continued fraction, and the
series matched brute-force quadrature to 0.03 %.

| closure, exact-direction field, sun only                   | median | p90   | quintiles                     |
| ---------------------------------------------------------- | ------ | ----- | ----------------------------- |
| vMF lobe of mean cosine g a (kept)                         | 0.174  | 0.322 | +0.10 +0.18 +0.20 +0.18 -0.12 |
| HG lobe of mean cosine g a                                 | 0.187  | 0.337 | +0.04 +0.16 +0.20 +0.18 -0.06 |
| vMF light convolved exactly with the broad lobe            | 0.205  | 0.357 | +0.19 +0.23 +0.23 +0.21 -0.09 |
| HG light convolved exactly                                 | 0.209  | 0.385 | +0.18 +0.24 +0.25 +0.23 -0.03 |
| delta + isotropic light convolved exactly                  | 0.253  | 0.743 | +0.06 +0.22 +0.25 +0.27 +0.39 |
| the sun's single scattering split off, beam lobe for it    | 0.174  | 0.324 | +0.10 +0.18 +0.20 +0.18 -0.11 |
| second moment about the beam, isotropic + HG fit           | 0.211  | 0.389 | +0.19 +0.24 +0.25 +0.23 +0.03 |
| second-moment tensor, P1 to HG to isotropic + HG about F   | 0.179  | 0.333 | +0.16 +0.19 +0.20 +0.17 -0.06 |
| quadratic harmonics from the tensor, clamped at zero       | 0.133  | 0.284 | +0.12 +0.13 +0.14 +0.14 -0.07 |
| the same with the HG tail where sharper than P1            | 0.122  | 0.272 | +0.08 +0.10 +0.13 +0.13 -0.05 |
| for comparison: forward peak and gather, vMF               | 0.139  | 0.256 | +0.09 +0.14 +0.16 +0.14 -0.10 |
| for comparison: 1M probes, vMF                             | 0.130  | 0.244 | +0.06 +0.13 +0.15 +0.13 -0.06 |
| 1M probes, quadratic harmonics, clamped                    | 0.083  | 0.189 | +0.07 +0.08 +0.09 +0.08 -0.02 |
| 1M probes, quadratic harmonics and the HG tail             | 0.077  | 0.179 | +0.04 +0.06 +0.07 +0.07 -0.00 |

Log-space interpolation of the atlas (log phi and F / phi in the texels) changed nothing:
0.315 against 0.322. The 1M probes' gain is the splat's kernel shrinking with the cells.

- **The kept lobe is wrong in shape, in both directions at once.** Against the exact
  convolution a vMF of mean cosine 0.9 is 22 % low forward, twice too high sideways and
  five times too low backward; yet the exact convolution renders worse, +0.19 in the
  shadow, because the light itself is not a vMF lobe. Every single-moment lobe that
  brightens the lit face (HG, HG convolved, delta + isotropic) brightens the shadow with
  it: no one shape serves both.
- **The sun's single scattering is not the mixed light.** Splitting it off and giving it
  the phase's own lobe about the beam changes nothing: deep inside, where the probes
  matter, the light has scattered many times.
- **The field says what the light is.** With the second-moment tensor baked (the
  deposits' dir dir^T), the second moment about F weighted by fluence is 0.335 at the
  median, P1's 1/3: 69 % of the light sits in probes of mean cosine below 0.2, the
  mid-range probes are P1-like or flatter (46 % of those between 0.2 and 0.4 are
  oblate), and only the lit shell above 0.6, 4 % of the light, is HG-like. A lobe fit
  cannot express the oblate light, and the axial moment about the beam misses the
  flux's axis.
- **The quadratic harmonics do.** The traceless tensor's part, scattered by the lobe's
  second coefficient (Draine's is 0.57, large), is exact to that order and not axial;
  clamped at zero, and with the HG lobe's tail past order 2 where the light is sharper
  than P1, it takes the lit face from -0.12 to -0.05 and the dark quintiles from +0.10
  to +0.20 down to +0.08 to +0.13, as good as the gather without sampling. What remains
  is the field's: with 1M probes the same closure leaves every quintile within 7 % and
  the lit face exact, p90 0.18 against the kept 0.32.
- **The forward peak stays as it is.** The delta-M split keeps the peak's light on the
  view, where the march's own radiance is exact; the peak's Legendre coefficients fall
  as 0.98^l and no closure of a few moments can hold it, while the broad lobe's fall
  below 1e-6 by order 24. Scattering the probes' light by the whole phase through one
  lobe (`first_scatter = probes`) was 0.426 against 0.391 with the sky.

The price of the tensor: six more floats per deposit, 2.5 times the deposits' atomics,
which are over half the bake, and 2.5 times the bins' memory, 670 MB at the 262144
budget; the sun's fit would carry the tensor's ratios beside the flux's, and the atlas
three slices per channel. With a sun-aware bake the bins go and the tensor costs the
same as the sky's transfer does now.

## Clouds besides the WDAS cloud

Candidates for the bench at the WDAS half resolution (994 x 676 x 1225 voxels of
0.42 m, 1.6 GB as f16), found 2026-10-26:

- **Intel Volumetric Cloud Library** (dpel.aswf.io/intel-cloud-library, ASWF Asset
  License): 5 cumulus shapes, each dense and sparse, at three tiers of 4x: small
  (2 m voxels, ~160 across), medium (0.5 m, ~640 x 370 x 540, 100 to 125 MB each),
  large (0.125 m by the tiers' ratio, ~2600 across, ~3.4 GB each, 33.8 GB the
  package). The clouds are 320 x 190 x 270 m, two thirds of the WDAS cloud's. The
  medium tier converts and renders in the bench as it is (three of the five carry
  rounding noise of -7e-9 that `convert_vdb.py` rejects; clamped at zero for the
  previews); the large tier resampled by 2 would match the WDAS half's resolution.
  Rendered previews: proper cumulus with lobes, less fine detail than WDAS at the
  medium tier.
- **CGHEVEN hero cumulus and congestus VDBs** (cgheven.com/assets/vdbs, CC0, about
  twenty clouds): single `density` float grids in a 0.02 unit frame, 4 to 11 M
  active voxels; cumulus 01 is 597 x 165 x 361 (the widest found), cumulus 20
  414 x 136 x 251, congestus 21 326 x 291 x 313. Rendered in the bench at 0.65 units
  a voxel, they are rounded cumulus with lobes and a flat base, with density like
  the Intel clouds'. Downloads are RAR5 archives on cdn.cgheven.com under
  `VFX/VDBS/Clouds/{cumulus_clouds,Congestus_Cloud}/Hero_Clouds/<range>/v<n>/`, which
  `unar` opens (7z does not); the archives for cumulus 32 and 43 and congestus 50
  hold the files of 20 and 21 again, so those three are unavailable.
- **JangaFX free cloud pack** (jangafx.com, CC0): 10 EmberGen clouds of ~300 x 500
  x 170 voxels at 1 unit, 4 to 6 M active, z up in the file (swapped to y for the
  bench). Rendered at 0.75 units a voxel they are plausible but soft cumulus with
  about half the Intel clouds' density and a third of their voxels: the lowest
  detail of the three sources.
- **VFX Assets free high-altitude clouds**: cirrus, non-commercial; a different look.

The bench places a volume by its own x and z and its minimum y at the base
altitude, so a cloud whose grid starts at positive x and z sits off the frame's axis;
`convert_vdb.py` gained `--centre`, `--voxel`, `--z-up` and `--density` for this.
The bench scales every cloud's density by one extinction, so a file's values set its
optical depth; by the median vertical column over the cloud's footprint, in density
times units, the sources differ by an order of magnitude:

| cloud | max density | median active density | median column | p90 column |
| --- | --- | --- | --- | --- |
| WDAS half | 1.00 | 0.50 | 52 | 112 |
| CGHEVEN cumulus 01 at 1.2 | 1.00 | 0.42 | 30 | 55 |
| Intel dense 1 | 0.50 | 0.19 | 13 | 29 |
| JangaFX cloud 02 at 1.0 | 0.73 | 0.09 | 5 | 13 |

They were kept for a while, each with the density factor that brings its median
column to the WDAS cloud's (`intel_dense_1` at its own 0.5 voxels, `--density 4`;
`cgheven_cumulus_01` at `--voxel 1.2 --centre --density 1.75`; `jangafx_cloud_02` at
`--voxel 1.0 --centre --z-up --density 11`), rendering as the WDAS cloud does with
bright tops over a dark base, and replaced on 2026-10-27 by the four below.

The viewer's cloud bench panel lists the volumes of `cache/clouds` and the config
names the active one as `cloud_volume`, which the presets `wdas-cloud*` set;
`--cloud-volume` outranks it. Switching rebuilds the bench mode around the new
volume, and the probe cache tells volumes apart by their grid alone, so a volume
converted again onto the same grid must have its probe cache deleted.

### Resolution by resampling and erosion

A "4 Free Clouds" VDB pack (Altocumulus 787 x 177 x 783 voxels, Cumulus Cloud 1
542 x 86 x 495, Cumulus Congestus 378 x 282 x 273, Stratocumulus 1 471 x 26 x 462,
single `density` grids peaking at 4 to 33) brought two layer clouds and a tower,
at a quarter of the WDAS half's resolution per axis. `convert_vdb.py` gained
`--resample`, a cubic resampling by a factor per axis, and `--erode`, the game
clouds' detail erosion (Schneider 2015): a tiled inverted Worley noise of three
octaves, its largest feature `--erode-period` voxels (32), carves the resampled
density by (d - s n) / (1 - s n) with d over the 99th percentile of the active
voxels, so that the core keeps its shape and the thin edges break into wisps and
holes at the new voxels' scale. `--match-column` then scales the density to a given
median column, the WDAS cloud's 52, in place of a guessed factor. The converter now
works a slab at a time, as the altocumulus at 2x is 873 M voxels.

Rendered at strengths 0, 0.25 and 0.5 on the stratocumulus and the congestus: 0
is the resampled blob, smooth and featureless; 0.25 gives both cauliflower
surfaces; 0.5 breaks the thin layer into fragments while the congestus only gains
texture. All four were converted at 2x, erosion 0.35 and the WDAS half's 0.4167
units a voxel (`--voxel 0.4167 --centre --resample 2 --erode 0.35 --match-column
52`), viewed, and rejected: the erosion is invented surface detail that reads as
such, not the simulated structure the WDAS cloud has, and on the layers it shapes
the cloud rather than decorating it. The options stay in the converter.

### Altocumulus fields

Nine altocumulus VDBs (`Altocumulus.vdb` and `Altocumulus 2` to `9`, single
`density` grids peaking near 1 with voxels of 5.96, 9.23, 11.7 and 16.9 m, so
satellite-derived fields rather than modelled clouds) are kept at their native
resolution as `altocumulus_1` to `altocumulus_9`, converted with `--voxel` at a
tenth of the file's metres, the bench's units, and `--centre`. They span 7 to 38 km
and 85 to 206 voxels in height; the ninth is 2277 x 206 x 2198 voxels, 2.3 GB as
f16, past the loader's old 2 GB limit, now 4 GB, and the others 120 to 340 M
voxels. Their peaks match the WDAS cloud's but their cloudlets are made of lighter
stuff, a median active voxel of 0.13 to 0.27 against 0.50, and are small, so a
median column through the layer carries a hundredth of the WDAS cloud's 52 and the
thickest tenth a twentieth. Kept at the WDAS cloud's material: `--match-density
0.5` scales each so that its median active voxel is 0.5, which `convert_vdb.py`
gained beside `--match-column`; a layer's column then stays its own, an optical
depth of a few through a typical cloudlet, which is what an altocumulus is.

### Simulated hero clouds

An "HD clouds" VDB pack of five simulated cumuli (`cloud1` to `cloud5`, density and
temperature grids of 0.015 or 0.02 units, 716 to 918 voxels wide, 287 to 384 tall,
30 to 76 M active, densities peaking at 60 to 200) is kept as `hd_cloud_1` to
`hd_cloud_5` at native resolution, the WDAS half's 0.4167 units a voxel so that
they stand 1.2 to 1.6 km, centred and at `--match-density 0.5`: their median
columns then come to 16 to 40 against the WDAS cloud's 52. Per axis they are 0.7 to
0.9 of the WDAS half, the nearest to it of any pack found, and 200 to 640 MB each.

### A cumulonimbus field

`Cumulonimbus_Field_8` (an `f216` archive of full, half, quarter and eighth tiers,
13.3 m voxels at full, 3503 x 829 x 4720 of them, 925 M active, a field 47 x 63 km
and 11 km tall, satellite-derived like the altocumulus) is far past what a 3D
texture holds at full, 13.7 G voxels dense, and twice the WDAS half at half, 1.72 G.
The converter now reads the VDB a slab at a time through `copyToArray` on sub-boxes
and resamples by any factor, so the half tier downsampled by 0.782 lands at the
WDAS half's count, 822 M voxels of 34 m, kept as `cumulonimbus_field_8` with
`--resample 0.782 --voxel 3.4066 --centre --match-density 0.5`. Its material is
light, a median active voxel of 0.04 before the match, and the quarter tier halved
reproduces the eighth tier's grid to within a voxel.

Cumulonimbus and anvils: nothing free at useful resolution was found. The free
CGHEVEN congestus clouds are the largest towering ones (congestus 20 is 333 x 258 x
329 voxels, a dense tall blob with little surface detail), and their CDN serves the
file of congestus 20 for 25, 27 and 28 too. Paid options: screenglow's "VDB Clouds
Volume 3 – Cumulonimbus" (ArtStation/Cubebrush, USD 29, 0.01 unit voxels plus
coarser tiers; the pages block fetching, so the voxel counts are unknown), The Pixel
Lab's "Static Anvil" (75 clouds in three tiers, 6.9 GB in all, USD 69, so a hero
tier of tens of MB each), and Akna 3D's "VDB Cumulonimbus Cloud Field" series on
CGTrader and TurboSquid (USD 10 each, ~6300 x 6200 x 1040 at full resolution with
half, quarter and eighth tiers, derived from Landsat imagery, so fields that are
wide and flat rather than single towers; a free eighth-resolution sample pack of
all 13 needs a CGTrader account).

## The density's filter

The bench reads the density grid through the hardware's trilinear sampler, whose
slope breaks at every voxel face. Tried on 2026-10-26 and dropped: warping the voxel
fraction by a cubic smoothstep before the filtered fetch (Quilez's smooth
filtering), slope-continuous and still a convex blend of the same eight voxels so
the majorants hold. It cost 20% per reference sample and 9% per approximation
sample on the WDAS half at a 5 degree field and did not read as less blocky, so
`extinction_at` is the plain trilinear fetch again. The probe cache format had gone
to 5 for it and stays there.

Kept instead, for the approximation's view march alone: `extinction_cubic`, the
tricubic B-spline, which is smooth in slope across voxel faces, as the eight
trilinear fetches Sigg and Hadwiger split it into. A jittered point read in place
of the trilinear fetch was tried first and dropped, the same filter with noise; then
one of the eight lobes drawn by its weight, the B-spline in expectation, which on
the WDAS preset at 512 x 512 took the approximation's 16 spp relative RMS over the
cloud from 0.22% to 0.25% at 4% more per sample, and gave way to the eight fetches
on 2026-10-26 to leave the march no random draw but its phase. The B-spline's and
trilinear's 256 spp means agree to 0.01%, and their difference lies along the voxel
grid's lines, the trilinear creases. The B-spline's support reaches two voxels, so it can exceed the majorant, which the
march only paces by; the reference, the sun beams and the probes' walks keep the
trilinear read that the majorant bounds. Catmull-Rom would keep the smoothness
without the B-spline's half-voxel softening, but its negative lobes would need a
signed draw and a floor at zero.

## The view march's steps at the cloud's edge

The march sizes its steps by the cell's majorant optical depth, so a thin shell cell
gets one jittered sample where the wisps vary by the voxel. Since 2026-10-26 the
view march's step is also capped at `view_step_max_voxels` (panel slider; 64, the
default, is none), for judgement by eye: the user found two voxels no help. By the numbers, on the WDAS preset at 512 x 512,
caps of two and one voxel move the converged picture by over 1% in 12% of the
pixels, leave the 16 spp noise over the shell at 0.44% against 0.43% and the error
against a 1024 spp reference at 9.5% of the shell's mean and 8.0% of the core's,
and cost 50% and 120% more per sample. The shell's noise is half the sub-pixel
jitter of the camera ray, 0.21% with rays pinned to pixel centres, over wisps finer
than a pixel, and half the march's random phase and lobe draws.

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
