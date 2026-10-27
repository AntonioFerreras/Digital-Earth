# Cloud transfer probes: outline

Probes inside the clouds store how the cloud carries light from any direction to
the probe: the sky's light all of it, the sun's only what scattered on the way, as
the march takes the sun's single scattering exactly. They depend only on the density
field, so the sun and sky can change without a rebake, and at runtime the ray
marcher combines them with the current light. They replace the model's guesses from
straight-line optical depths. Those cannot light a lobe in another part's shadow or
a cloud's base; there the light flows around the cloud, as the cloud bench shows
(`docs/research/20261022-cloud-bench.md`).

## Why

On the bench, the worst fifth of the cloud, a lobe in the tower's shadow and the
base, gets 13 times too little light from the Nubis-style model. The path tracer
puts 9.7 units of multiply scattered sunlight there and the model 0.9; the sky adds
2.9 there and the model 0.03. Both terms take one straight optical depth, towards
the sun or straight up, and diffused light does not travel straight. The
dimensional profile cannot help: it only multiplies the term, and it knows the
cloud's shape, not where the light is.

## What a probe holds

The light arriving at the probe from distant light, in the P1 form of the diffusion
approximation, (phi + 3 F . w) / 4 pi along w, as functions of the light's direction:

- **Sky:** linear in its radiance. Per spherical harmonic j of order 3 of the sky,
  phi_j and F_j: 64 values, phi_0 a float and every other a byte over
  +-sqrt(2 l + 1) phi_0, which bounds it. The sky is smooth over directions, and
  integrating a smooth light needs only the transfer's low orders; order 3 resolves
  the horizon's band at sunset, which order 2 smears into the sky's mean.
- **Sun:** one direction, which a probe in its shadow receives orders of magnitude
  more weakly than directions where the cloud is thin; a linear fit of few harmonics
  rings below zero there. So log phi(s) is fitted, through order 5 (36 values; a
  setting fits order 3 from the same bins), and
  F / phi, which lies within the unit ball, through order 1 (12 values); the constant
  harmonic is a float, every other value a byte over the field's range of that
  coefficient. The sun's single scattering is marched exactly, so this transfer holds
  sunlight scattered on its way only.

120 bytes a probe, 52 the sun's and 68 the sky's, fitted once from the bake's sums
and bins, which then leave the runtime. At every change of the light the lighting
pass contracts them with the present sun and the sky's harmonics into phi and F per
colour channel.
The march scatters them towards the viewer through a phase of first moment g by the
M1 closure: the light taken as a von Mises-Fisher lobe about F with the same phi and
F, whose mean cosine the phase multiplies by g. P1, (phi + 3 g F . -dir) / 4 pi, goes
negative once g |F| / phi exceeds 1/3 against the view, which with the sun behind the
camera blackened clouds' sunlit edges.

## Baking

A backward random walk from each probe, the adjoint of the reference's transport, one
per probe per frame:
- a uniform look direction, then delta tracking and phase sampling through the cloud;
- leaving the cloud, the walk adds 4 pi Y_j(escape) to the sky's transfer, with the
  look's opposite for F;
- at every collision, for a sun it does not know: a direction drawn from the phase,
  so that the phase cancels, and the transmittance towards it, marched, into that
  direction's bin of an 8 x 8 equal-area octahedral map. A march's exact
  transmittance converges the bins far sooner than ratio tracking's: a walk's light
  hides in its few collisions near a lit surface. Collisions past an optical depth of
  8 towards the sun count as unlit, and those past 12 towards every direction, by a
  coarse volume of least depths, skip the march.

The bins are bake-only (268 MB at 64^3); the fit to log-SH runs from them each time
the walks or the order change. Bakes persist per setting (`src/render/ProbeCache.zig`).
The similarity switch to cut deep walks is not built.

## Spacing and the hard part

The diffuse field is smooth over a transport mean free path, 1 / (sigma (1 - g)):

- bench cumulus at 0.1 /m: about 55 m (g = 0.82, the walks' phase);
- the earth's present clouds at 0.0029 /m: about 2 km.

The bench's probes lie 44 to 80 m apart (64^3). Light scattered once or twice before
the marched point changes within a few mean free paths of a surface, far finer: that
is the grid the probes show. Neither diffusion corrections nor finer probes were the
answer; following the first scatter at full resolution was (Reconstruction, One-
scatter gather, Forward peak, below).

## Scaling to the earth

The bake depends only on density, which separates residency from lighting:

1. **Static clouds** (today's coverage map): bake offline per tile and stream with
   the terrain tiles' machinery.
   - At today's transport mean free path of about 2 km, half the planet clouded in
     three layers is about 2e8 probes, 23 GB at 57 f16 values.
   - Levels of detail double the spacing per level away from the camera.
2. **Real-scale or moving clouds:** a camera-centred clipmap of probe bricks.
   - Only bricks the bound map marks as cloudy hold probes.
   - Bricks bake progressively under a per-frame walk budget.
   - The sun moving never re-bakes; a brick re-bakes only when its density changes
     or it enters the clipmap.

The probe format, the bake kernel, and the runtime lookup are the same in both; only
the brick allocator differs.

## Plan

1. **Bench:** done in its first form: 64^3 probes, marched bake, log-SH sun, forward
   peak in the march. Scores in `docs/research/20261022-cloud-bench.md`.
2. **Earth, static:** probes in the cloud shell, baked offline per tile, streamed.
3. **Clipmap,** when clouds move or reach real scale. At the bench's density a sparse
   global cloud layer would need about 3e12 probes, 250 to 350 TB, so the clipmap
   must coarsen with distance, and coarse levels must still place probes inside
   small clouds.

## Risks

- **Leaks across thin gaps:** the trilinear blend has no visibility; a cloud thinner
  than the spacing takes light from its neighbour's probes.
- **Bake cost in dense cores:** walks through 100 optical depths scatter thousands of
  times, each with a march towards a sun direction; deep probes converge slowest.
- **Coarse levels:** the forward peak and the march carry detail at full resolution,
  but how far probe spacing can grow before the light inside small clouds goes wrong
  is untested.

## Reconstruction

Trilinear interpolation of the probes limits the light to the probe spacing, 44 to 80
m on the bench. At unit density the mean free path 1/sigma is 10 m and the transport
mean free path l* = 1/(sigma (1 - g)) about 55 m (g = 0.82, the walks' phase). Diffusion
theory says the field varies slowly, over l* or the cloud's size, but in two layers:
within a few mean free paths of a sunlit surface, where it is locked to the sun's
beam, and within about l* of any surface, where it falls towards the vacuum boundary.
The march knows both at full resolution: it marches tau, the optical depth towards the
sun, at every step for the single scattering, and the edge distance d is a texture.

**Beam-locked part.** Delta-scaling (Joseph, Wiscombe and Weinman 1976) treats the
phase's forward peak, a share f = <P2> of it, as unscattered, with extinction
sigma' = (1 - f) sigma and asymmetry g' = (g - f) / (1 - f); for the walks' droplet
phase f = 0.707. Sunlight then runs on as a beam E exp(-tau'), tau' = (1 - f) tau. The
exact single scattering takes E exp(-tau), so the probes hold a collimated part

    Lambda_c = E (exp(-tau') - exp(-tau)), travelling along -s.

The beam feeds the diffuse light. P1 without absorption:

    div F = Q0,   grad phi / 3 + sigma' (1 - g') F = Q1,
    Q0 = sigma' E exp(-tau'),   Q1 = g' Q0 (-s).

Where tau' falls at sigma' towards the sun (s . grad tau' = -sigma', the plane-parallel
case), phi_p = alpha E exp(-tau') solves it with

    alpha sigma'^2 / (3 sigma' (1 - g')) = -g' sigma'^2 / (sigma' (1 - g')) - sigma'
    => alpha = -3, for any g',

the Eddington result for conservative scattering, with flux
F_p = (Q1 - grad phi_p / 3) / (sigma' (1 - g')) = E exp(-tau') s, which cancels the
collimated part's. The remainder H satisfies the source-free diffusion equation: it
is harmonic, without interior extrema.

**Boundary ramp.** At a vacuum boundary phi = 2 D d phi / dn, so near any surface H
rises linearly with the distance d inside from zero at an extrapolated distance
z_e = 2 D = (2/3) l* outside. H / (d + z_e) varies slowly.

**Reconstruction.** The lighting pass marches tau_i from each probe, once per sun, and
stores (L_i - Lambda_i) / (d_i + z_e), Lambda the collimated and the particular part
together. At a step:

    L(x) = Lambda(x) + (d(x) + z_e) sum_i w_i (L_i - Lambda_i) / (d_i + z_e),

the collimated part scattered towards the viewer by the exact phase, the rest through
P1. Nothing more is stored, the probes stay independent of the light, and every
constant follows from the phase and the extinction.

**Assumptions to check on the bench.** The plane-parallel gradient fails around bumps,
where tau' also varies across the beam; diffusion is approximate within l* of a
surface, where both corrections act; probes in air obey no diffusion law and are taken
as surface values, d = 0; leaks across gaps thinner than the spacing remain.

**Result.** Neither correction reduced the grid on the bench (64^3, 800 walks); by
eye both looked worse, and they are removed. The diffuse beam term drew dark creases
along lit bumps, and the boundary ramp changed nothing measurable: both rest on
diffusion within a transport mean free path of the surface, about the probe spacing,
where diffusion does not hold. Only the collimated part, which needs delta-scaling
alone, helped: backlit rims from 0.71 to 1.10 of the reference.

## One-scatter gather

The grid is the light scattered once or twice before the marched point, which varies
over a few mean free paths near surfaces. The march follows one scatter exactly
instead of interpolating it: each pixel chooses one of its steps in proportion to its
weight (extinction x transmittance x step), draws a direction there from the phase,
and finds the first collision. There the sun's single scattering is marched and the
rest read from the probes; a direction leaving the cloud takes the sky's radiance.
Times the steps' total weight, the estimate is unbiased for the probes' part. The
probes then supply only light scattered twice before the marched point. Cost: one
collision search and one march to the sun per pixel and frame.

## Forward peak

The gather's gain comes mostly from the phase's forward peak: for cloud droplets
diffraction holds about half the scattered energy within a few degrees. A delta-M
split (Wiscombe 1977) writes the phase of the view's scatter as

    p = f delta(forward) + (1 - f) p_broad,

f the peak's share: the Henyey-Greenstein lobe of the droplet fit (0.53), g^2 for a
Henyey-Greenstein phase, the truncated diffraction peak for a tabulated Mie phase. The
peak's light keeps to the view, so the radiance R arriving along it satisfies

    dR/dt = (1 - f) sigma R - sigma (S + (1 - f) P_broad),
    R = integral of exp(-(1 - f) tau) sigma (S + (1 - f) P_broad) + exp(-(1 - f) tau_all) sky,

S the sun's single scattering through the whole phase, which keeps the peak's halo
exact, and P_broad the probes' light through the broad lobe's first moment. The march
attenuates at (1 - f) sigma and scatters at sigma: deterministic, no extra rays. The
error is the peak's width, about 4 degrees, over the few tens of metres a pixel's
light comes from: a couple of metres. Optionally the broad lobe is gathered instead
of read from the probes.

