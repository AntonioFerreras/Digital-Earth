# Ray-march step placement: measurements

The `raymarch` mode used 32 uniform steps per segment. It now takes 12 to 32
steps per segment, depending on the segment's length, and places them by
optical depth (`StepPlacement` in `shaders/modes/raymarch.slang`). This note
records how both schemes were measured against the path tracer and against a
4096-step march.

## Scheme

- **Count.** The step count is `round(lerp(12, 32, saturate(length / 1000 km)))`.
  It follows Hillaire 2020 ("A Scalable and Production Ready Sky and Atmosphere
  Rendering Technique", EGSR). The reference code is `IntegrateScatteredLuminance`
  in `Resources/RenderSkyRayMarching.hlsl` of sebh/UnrealEngineSkyAtmosphere,
  which computes `SampleCount = lerp(RayMarchMinMaxSPP.x, RayMarchMinMaxSPP.y,
  saturate(tMax * 0.01))` with tMax in km. Its defaults are 4 to 14 samples
  (`Application/Game.h`). Our ranges are wider because a ray from orbit can
  cross 2400 km of atmosphere.
- **Placement.** Hillaire's nonlinear placement squares the step
  boundaries, `t = tMax * (s / N)^2`, and puts the sample at 0.3 of each step.
  That crowds steps at the ray origin. This is right for a ground camera but
  wrong from orbit, where the dense air sits at a limb ray's tangent point or
  at the ground end of the ray. Our placement is not his:
  1. A coarse pass reads the extinction at the centres of 16 equal cells.
  2. Each cell's share mixes its fraction of the segment's optical depth with
     a 25 % uniform share.
  3. Step boundaries are uniform in the cumulative share, which is piecewise
     linear in t and inverted with a forward-only cell cursor.
- **Unchanged.**
  - Each sample sits at the frame's jitter within its step.
  - Transmittance uses the sample's extinction over the step's own length
    (`exp(-sigma_t dt)`, a product over the chosen steps).
  - In-scattering uses the energy-conserving step integral.
  - The last step ends exactly at the segment end.

Variants tried, each against the 4096-step limit with SS only. Error is the
mean relative error over lit 3x3 bins:

- Hillaire's t^2 placement: 4 to 14 steps over 100 km gave errors 3-12x larger
  than 32 uniform steps on space and limb views (limb ~3.8 %). 8 to 32 steps
  was still ~3x worse there, though better for the low camera.
- Optical-depth placement with 32 fixed steps: about 8x lower error than uniform.
  With 16 fixed steps it beat uniform-32 everywhere except the night-side low
  camera (sun -5 degrees), where the light comes from thin, high, sunlit air.
- Raising the uniform share to 50 %, or down-weighting cells in the planet's
  shadow by 10x, did not fix that case. Driving the count by optical depth
  (8 to 32, 4 steps per unit) was worse than driving it by length.
- Chosen: count by length from 12 to 32, optical-depth placement, 25 % uniform
  share.

## Method

- **Scene.** Symmetric preset (no maps, no clouds), 480x270, linear PFM
  output, luminance Y from Rec.709.
- **Views.** Views are built by `make_configs.py`, which is kept with the
  renders. The sun elevation is measured above the horizon of the camera's
  sub-point.
  - Space: full disk from 4 planet radii, sun azimuth 90 degrees.
  - Limb: from 400 km, looking 2 degrees above the horizon dip, sun azimuth
    45 degrees.
  - Low: from 2 km, level, sun azimuth 45 degrees.
- **SS-only.** Uses `earth render --single-scattering`, so both modes see the
  same light paths:
  - The path tracer gets one bounce: next-event single scattering at the first
    collision, plus direct sun on the surface; any surface hit ends the path.
  - The ray marcher gets one segment without the multiple-scattering table,
    plus the same direct surface term.
  - No view has the sun disk in frame.
- **References.**
  - Path tracer: 65536 spp. Its noise floor comes from comparing the two
    halves of the samples.
  - Ray marcher: 1024 spp.
  - 4096-step limit: 4096 uniform jittered steps.
- **Isolating the quadrature.** RNG streams are keyed by pixel and frame, so
  the ray marcher and its 4096-step limit draw the same wavelengths and
  jitters. Their difference is therefore the step quadrature alone.
- **Agreement with the path tracer.** The 4096-step SS march matches the path
  tracer's SS luminance to 0.02 % or better on every view. The single-scattering
  model has no other bias at this precision.
- **Metrics.**
  - ratio: sum of Y over lit pixels (Y_ref > 1e-3 x its 99.5th percentile).
  - rel: mean |dY| / Y over lit 3x3 bins.
  - p99: the 99th percentile of the same per-bin relative difference.
  - limb: the same metrics over grazing rays. From space these are rays that
    miss the planet below 110 km tangent altitude, or hit it at incidence
    cosine < 0.25. From 2 km they are rays within 3 degrees of the horizon.
- **Noise.** Against the path tracer, rel is dominated by noise: about
  0.7 %, about 2.5 % for low/-5, from 1024-spp spectral noise plus the path
  tracer's floor of 0.1 to 0.3 %. Ratios are not.

## Single scattering: step error (vs the 4096-step limit)

| view | sun | ratio before | ratio after | rel % before | rel % after | p99 % before | p99 % after | limb rel % before | limb rel % after | limb ratio before | limb ratio after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| space | 60 | 0.9993 | 0.9998 | 0.08 | 0.03 | 0.33 | 0.16 | 0.21 | 0.06 | 0.9980 | 0.9996 |
| space | 30 | 0.9996 | 0.9999 | 0.06 | 0.04 | 0.29 | 0.29 | 0.12 | 0.03 | 0.9990 | 0.9998 |
| space | 10 | 0.9995 | 0.9998 | 0.07 | 0.05 | 0.28 | 0.34 | 0.11 | 0.03 | 0.9991 | 0.9999 |
| space | 3 | 0.9993 | 0.9998 | 0.08 | 0.05 | 0.29 | 0.35 | 0.13 | 0.03 | 0.9989 | 0.9998 |
| space | 0 | 0.9992 | 0.9998 | 0.09 | 0.05 | 0.29 | 0.38 | 0.14 | 0.03 | 0.9988 | 0.9998 |
| space | -5 | 0.9990 | 0.9997 | 0.10 | 0.06 | 0.31 | 0.38 | 0.16 | 0.03 | 0.9986 | 0.9998 |
| limb | 60 | 0.9999 | 1.0000 | 0.01 | 0.01 | 0.04 | 0.05 | 0.01 | 0.01 | 0.9999 | 1.0001 |
| limb | 30 | 0.9986 | 0.9998 | 0.13 | 0.03 | 0.24 | 0.07 | 0.13 | 0.03 | 0.9985 | 0.9998 |
| limb | 10 | 0.9972 | 0.9995 | 0.28 | 0.06 | 0.50 | 0.16 | 0.26 | 0.06 | 0.9973 | 0.9995 |
| limb | 3 | 0.9968 | 0.9993 | 0.32 | 0.07 | 0.57 | 0.20 | 0.30 | 0.07 | 0.9970 | 0.9994 |
| limb | 0 | 0.9968 | 0.9993 | 0.31 | 0.08 | 0.58 | 0.23 | 0.30 | 0.07 | 0.9969 | 0.9994 |
| limb | -5 | 0.9974 | 0.9993 | 0.23 | 0.09 | 0.52 | 0.32 | 0.25 | 0.08 | 0.9973 | 0.9994 |
| low | 60 | 0.9992 | 0.9997 | 0.09 | 0.03 | 0.49 | 0.17 | 0.07 | 0.02 | 0.9993 | 0.9998 |
| low | 30 | 0.9993 | 0.9997 | 0.07 | 0.03 | 0.39 | 0.17 | 0.05 | 0.02 | 0.9995 | 0.9998 |
| low | 10 | 1.0001 | 0.9997 | 0.05 | 0.03 | 0.33 | 0.16 | 0.08 | 0.02 | 1.0003 | 0.9999 |
| low | 3 | 1.0021 | 0.9996 | 0.13 | 0.04 | 0.49 | 0.17 | 0.19 | 0.02 | 1.0021 | 1.0000 |
| low | 0 | 1.0046 | 0.9994 | 0.25 | 0.06 | 0.92 | 0.21 | 0.36 | 0.04 | 1.0064 | 1.0002 |
| low | -5 | 1.0045 | 0.9981 | 0.37 | 0.25 | 0.99 | 0.72 | 0.72 | 0.09 | 1.0072 | 0.9998 |

## Single scattering vs the path tracer

| view | sun | ratio before | ratio after | rel % before | rel % after | p99 % before | p99 % after | limb rel % before | limb rel % after | limb ratio before | limb ratio after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| space | 60 | 0.9993 | 0.9998 | 0.72 | 0.71 | 2.31 | 2.31 | 0.78 | 0.78 | 0.9993 | 1.0009 |
| space | 30 | 0.9995 | 0.9999 | 0.74 | 0.74 | 2.47 | 2.47 | 0.76 | 0.76 | 1.0003 | 1.0011 |
| space | 10 | 0.9994 | 0.9998 | 0.77 | 0.76 | 2.74 | 2.75 | 0.76 | 0.76 | 1.0003 | 1.0011 |
| space | 3 | 0.9993 | 0.9998 | 0.78 | 0.78 | 2.75 | 2.75 | 0.79 | 0.79 | 1.0001 | 1.0010 |
| space | 0 | 0.9992 | 0.9998 | 0.79 | 0.79 | 2.95 | 2.86 | 0.78 | 0.78 | 1.0000 | 1.0010 |
| space | -5 | 0.9991 | 0.9998 | 0.80 | 0.80 | 2.97 | 2.93 | 0.76 | 0.76 | 0.9998 | 1.0009 |
| limb | 60 | 1.0001 | 1.0002 | 0.73 | 0.73 | 2.62 | 2.61 | 0.78 | 0.78 | 1.0000 | 1.0001 |
| limb | 30 | 0.9988 | 0.9999 | 0.75 | 0.74 | 2.64 | 2.62 | 0.80 | 0.79 | 0.9986 | 0.9998 |
| limb | 10 | 0.9973 | 0.9996 | 0.80 | 0.77 | 2.75 | 2.64 | 0.84 | 0.81 | 0.9973 | 0.9996 |
| limb | 3 | 0.9970 | 0.9995 | 0.82 | 0.78 | 2.81 | 2.64 | 0.86 | 0.81 | 0.9970 | 0.9995 |
| limb | 0 | 0.9970 | 0.9994 | 0.83 | 0.78 | 2.81 | 2.66 | 0.86 | 0.82 | 0.9970 | 0.9994 |
| limb | -5 | 0.9975 | 0.9994 | 0.83 | 0.81 | 2.87 | 2.85 | 0.85 | 0.82 | 0.9974 | 0.9994 |
| low | 60 | 0.9993 | 0.9998 | 0.73 | 0.72 | 2.33 | 2.32 | 0.72 | 0.71 | 0.9993 | 0.9998 |
| low | 30 | 0.9994 | 0.9998 | 0.73 | 0.72 | 2.33 | 2.33 | 0.72 | 0.71 | 0.9996 | 0.9998 |
| low | 10 | 1.0003 | 0.9998 | 0.73 | 0.73 | 2.39 | 2.38 | 0.72 | 0.71 | 1.0003 | 0.9999 |
| low | 3 | 1.0024 | 0.9999 | 0.76 | 0.74 | 2.50 | 2.41 | 0.73 | 0.70 | 1.0022 | 1.0001 |
| low | 0 | 1.0053 | 1.0002 | 1.01 | 0.95 | 3.46 | 3.43 | 0.95 | 0.86 | 1.0067 | 1.0005 |
| low | -5 | 1.0068 | 1.0004 | 2.62 | 2.53 | 10.23 | 9.69 | 2.75 | 2.63 | 1.0086 | 1.0012 |

## Full comparison and the error split

"Rest" is the 4096-step march (no step error) against the path tracer. It
covers everything except stepping: the multiple-scattering table, ground
bounces, and table resolution. Stepping now leaves less than 0.1 % of the
luminance, 0.16 % at low/-5. The remaining 5 to 17 % deficit, 7 to 26 % at
the limb, is not quadrature.

| view | sun | full ratio before | full ratio after | 4096-step limit vs PT | stepping after (ratio - 1) | rest (limit - 1) | limb: limit vs PT | limb: stepping after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| space | 60 | 0.9466 | 0.9474 | 0.9477 | -0.03% | -5.23% | 0.9089 | -0.05% |
| space | 30 | 0.9452 | 0.9460 | 0.9462 | -0.02% | -5.38% | 0.8937 | -0.03% |
| space | 10 | 0.9415 | 0.9424 | 0.9426 | -0.03% | -5.74% | 0.8911 | -0.02% |
| space | 3 | 0.9394 | 0.9404 | 0.9407 | -0.03% | -5.93% | 0.8926 | -0.03% |
| space | 0 | 0.9383 | 0.9393 | 0.9397 | -0.03% | -6.03% | 0.8932 | -0.03% |
| space | -5 | 0.9365 | 0.9376 | 0.9379 | -0.04% | -6.21% | 0.8949 | -0.03% |
| limb | 60 | 0.8887 | 0.8895 | 0.8896 | -0.01% | -11.04% | 0.8727 | -0.01% |
| limb | 30 | 0.9012 | 0.9028 | 0.9031 | -0.03% | -9.69% | 0.8912 | -0.03% |
| limb | 10 | 0.9127 | 0.9150 | 0.9156 | -0.06% | -8.44% | 0.9075 | -0.05% |
| limb | 3 | 0.9146 | 0.9171 | 0.9177 | -0.06% | -8.23% | 0.9109 | -0.06% |
| limb | 0 | 0.9154 | 0.9178 | 0.9184 | -0.07% | -8.16% | 0.9122 | -0.06% |
| limb | -5 | 0.9180 | 0.9198 | 0.9204 | -0.07% | -7.96% | 0.9156 | -0.06% |
| low | 60 | 0.8680 | 0.8685 | 0.8688 | -0.03% | -13.12% | 0.8157 | -0.02% |
| low | 30 | 0.8815 | 0.8819 | 0.8822 | -0.03% | -11.78% | 0.8358 | -0.02% |
| low | 10 | 0.8773 | 0.8771 | 0.8774 | -0.04% | -12.26% | 0.8295 | -0.01% |
| low | 3 | 0.8712 | 0.8694 | 0.8698 | -0.04% | -13.02% | 0.8128 | -0.00% |
| low | 0 | 0.8615 | 0.8577 | 0.8581 | -0.05% | -14.19% | 0.7729 | +0.02% |
| low | -5 | 0.8377 | 0.8327 | 0.8340 | -0.16% | -16.60% | 0.7433 | -0.01% |

Before the change, the uniform march's bias partly cancelled the deficit at
low/-5 and low/0 (+0.5 %). Removing the bias makes those two views about
0.4 % darker against the path tracer.

## Performance

Ray marcher at 1920x1080, symmetric preset, 128 spp per run. The baseline and
new builds ran interleaved, six runs each, and the table gives the median
ms/spp.

| view | before | after |
| --- | ---: | ---: |
| default camera | 5.02 | 2.64 |
| sunset-hurricane camera | 7.88 | 4.28 |

Optical-depth placement with a fixed 32 steps measured 3.76 and 4.31
ms/spp. Length-scaled uniform steps measured 2.77 and 4.57. Both came from
single runs that were not interleaved.

- **Default camera.** The saving comes mostly from fewer steps on short
  segments: near-vertical rays and the ground-bounce segments.
- **Sunset camera.** The saving is not a step count, because the fixed
  32-step variant is just as fast. A likely cause is register pressure in the
  baseline's constant-count loop. The compiler probably unrolls that loop,
  with two LUT fetches per step. No profiler confirmed this.

## Open issues

- The terminator from orbit keeps a per-bin p99 error about equal to the
  baseline's (space/10: 0.34 % vs 0.28 %). Sunlit thin air over shadowed ground
  draws few steps, and the 25 % uniform share is all that covers it.
- The night-side low camera (sun -5 degrees) shows the same effect in its
  upward rays. A placement that weights cells by sun visibility, from the
  transmittance table rather than a binary shadow test, could close it, at 16
  extra texture fetches per segment.
