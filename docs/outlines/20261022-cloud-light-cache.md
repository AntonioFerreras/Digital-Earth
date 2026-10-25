# Cloud light cache: outline

Shelved: the code lives on the local branch `shelf/cloud-light-cache`. The ray
marcher models the clouds' multiple scattering as Nubis does instead.

The ray marcher's clouds get their multiply scattered light from a world-space
cache that the path tracer itself fills, instead of from a closed-form
approximation. The march keeps its exact single scattering of the sun and adds,
at every cloud step, the light the cache holds for that point.

## Why

Two closed forms failed on the clouds-only scene, where multiple scattering is
70 to 95% of the clouds' radiance:

- **Octaves** (Wrenninge 2013) compress the contrast, and no one set of
  attenuation, contribution, and eccentricity factors fits both test presets.
- **A slab table** (multiple scattering of a sunlit slab, indexed by the optical
  depth towards the sun and beyond the point) matches dense decks, but thin and
  broken cloud comes out about twice too bright and bright tops too dim. A slab
  is infinite sideways: light that escapes through a cloud's sides, the cause of
  the dark gaps and the bright tops, cannot leave it.

What a cloud scatters again depends on the shape around it, on the sky above it,
and on the ground and ocean below it. Only transport through the actual scene
captures all three. The transport mean free path in these clouds is about
2.5 km, so the multiply scattered field is smooth at kilometre scale and a grid
of that spacing can hold it.

Production engines (Unreal, Frostbite, Nubis, Red Dead Redemption 2) stop at
octaves plus an ambient term; precomputed radiance transfer and neural
estimators (Deep Scattering, MRPNN) are either tied to one cloud or need
training. Caching path-traced light is the method that generalizes to multiple
layers and 3D noise without new approximations.

## Decided

- **Gather backwards with `path_radiance`, not forward photons.** Each cache
  node traces paths from its own position with the path tracer's estimator
  (`modes/path_radiance.slang`), starting at bounce 1. That reuses the path
  tracer's transport unchanged: delta tracking, its phase conventions (the
  reduced peak, the deep isotropic bounces), spectral MIS, the sun's next-event
  estimation, sky, ground, ocean glint, and terrain shadows. Every node owns its
  accumulators, so no atomics are needed. Forward photons would need a second
  transport implementation and atomic splatting, and they would deposit nothing
  in thin cloud.
- **What a node holds:** the radiance arriving at it from every direction,
  except the sun's direct beam (path_radiance never returns that beam). The
  march's single scattering already covers it. Stored per unit sun irradiance,
  like the march's other in-scattering, as:
  - the mean incoming radiance per wavelength-bundle bin;
  - the normalized first moment of the incoming direction (grey), D = E1 / E0.
- **Lookup:** the incoming light is taken as a Henyey-Greenstein distribution of
  mean cosine |D| around D. It keeps the stored moments and stays positive, and
  it convolves with each HG lobe of the droplet phase in closed form: the mean
  cosines multiply, and Draine's lobe counts as the HG lobe of its mean cosine.
  The bundle bins are interpolated in the bundle's u, as the atmosphere's
  multiple-scattering table does. The linear moment form M (1 + 3 g D·d) was
  tried first and lost half the second order at sunset: there the droplet phase
  and the incoming light both peak forward.
- **Grid:** nodes spread over a gnomonic tangent plane anchored under the view,
  times the cloud shell's height. That gives N x N x 16 nodes, trilinearly
  filtered in hardware.
  - **Extent:** a fit kernel sizes the plane to the frustum's footprint on the
    shell, clamped.
  - **Where nodes gather:** only nodes whose neighbourhood can hold cloud (the
    bound map's coverage and its column) trace paths.
- **Accumulation:** the cache's running mean restarts with the image's
  accumulation, on frame 0, so any change that resets the image (camera, sun,
  scene) resets the cache too. No keys, no CPU readback.
- **The slab table and its walks go.** The cache replaces them.

## Shape

- `shaders/earth/cloud_cache.slang`: the grid's frame (anchor, tangents,
  extent), node positions, the position-to-texture mapping, the lookup, and the
  fit.
- `shaders/modes/cloud_cache_bake.slang`: three kernels:
  - **fit:** one group; writes the frame to a small buffer on frame 0;
  - **gather:** one thread per node; one path per frame into the node's sums;
  - **resolve:** writes means and directions into an rgba16f 3D texture with one
    slab per bin plus one for the direction, which the march samples.
- `path_radiance` takes the bounce its path starts at.
- `src/render/RayMarcher.zig` owns the buffers and texture and records fit,
  gather, and resolve before the march.

## Risks

- **Cost.** Every gathering node traces one full path per frame. A path through
  dense cloud bounces dozens of times. If a frame of gathers costs more than the
  march, gather a rotating subset of nodes per frame.
- **Resolution.** From orbit the plane spans thousands of kilometres, so the
  cells are several kilometres wide and cloud edges soften in the multiply
  scattered term. The single scattering is unaffected. Camera-centred clipmap
  levels are the follow-up.
- **Noise.** The cache converges at the image's own rate. Early frames carry
  node noise, which the trilinear filter spreads into kilometre-sized blotches.
- **Angular detail.** The first moment cannot hold the bright forward lobe of
  thin cloud seen towards the sun. If that shows, add a second lobe or the l = 2
  band.

## Validation

Clouds-only renders of the florida and sunset-hurricane presets against the
path tracer, scored as before (mean, thin-third, and thick-third luminance
ratios), then the full scene and the user's low view. Success means the thin
and thick ratios both land near 1, where the slab table gave about 2 and 0.8.

## First results

Clouds-only, 480 x 270, 512 frames, cache 256 x 256 x 32. Ratios are of luminance
against the path tracer:

| preset           | method      | mean | thin | thick |
| ---------------- | ----------- | ---- | ---- | ----- |
| florida          | slab table  | 1.01 | 1.92 | 0.81  |
| florida          | cache       | 0.85 | 1.14 | 0.78  |
| sunset-hurricane | slab table  | 1.10 | 2.23 | 0.88  |
| sunset-hurricane | cache       | 0.80 | 0.76 | 0.77  |

- **Contrast:** the gap between thin and thick cloud mostly closes.
- **Level:** what remains is about 20% too dark. Making the first scatter
  isotropic in both renderers isolates the stored light level, and that reaches
  only 0.76 of the second order at 128 x 128 x 64 nodes and 0.83 at
  512 x 512 x 16. The rest of the loss is in the angular model. The light near
  sunlit surfaces varies faster than a kilometre-scale grid resolves.
- **Resolution:** at the fit's 3000 km reach, cells are about 23 km wide. Small
  cumulus near the camera come out too bright, and the cache's noise shows as
  kilometre-scale blotches.
- **Cost:** about 6.5 ns per node path, so 2M nodes cost about 13 ms per frame,
  whatever the image size.
