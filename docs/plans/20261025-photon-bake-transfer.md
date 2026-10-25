# Plan: guided photon emission, then a diffusion tail for the photon bake

Question: can the photons' bake (shaders/modes/cloud_bench_probes.slang,
`trace_photons`) converge faster by moving information between probes? It must beat
plain photons at equal bake time on view 00's volume, against the 20000-round photon
field. Two local schemes have already failed: control variates and photon connections
(docs/research/20261022-cloud-bench.md).

## Measurement

- `earth bake-probes` at 256 rounds with `--half`. Its timer now starts after a frame
  that only lays the probes out. Before, every timing carried the layout's build,
  2 to 4 s on a bake of about 2 s.
- Noise is the split-half sigma of the sun's fitted fluence and of the sky's fluence.
  Efficiency is 1 / (sigma^2 x seconds) against plain photons.
- For a biased scheme, the error against the reference matters more: the sun's and
  the sky's median relative error, the sky's total, and the sun's ratio by brightness
  decile.

## What the photons spend their time on

Orders past 16 collisions carry most of the dense core's light and take most of the
trace; see "Diffusion tail for deep photons" in docs/research/20261022-cloud-bench.md.

## 1. Guided emission (unbiased)

Photons enter from a uniform direction at a uniform point of the box's projected area.
Instead, draw the entry from a learned distribution over cells of (sun bin, box face,
16 x 16 cell of that face), 49152 cells, and weight each photon by uniform over drawn
density. Unbiased, as long as every cell keeps a defensive share of the uniform
distribution.

- **Score:** a photon's squared contributions relative to what each probe already
  holds, sum over deposits of (deposit / level)^2. The level is the probe's sum in
  that bin, floored at a tenth of the probe's mean over bins; for light that has not
  collided, it is the probe's sky fluence. Each cell keeps its photons' count, summed
  score and summed collisions.
- **Distribution:** in proportion to the cell's measure times sqrt(mean score / mean
  cost), with cost as 1 + collisions. This minimizes the sum of the probes' relative
  variances per unit time, mixed with uniform at a share alpha.
- **Schedule:** learn during the first rounds, rebuild the distribution as rounds
  double, and keep learning throughout.
- **Measure:** a sweep over alpha against plain photons, together with the predicted
  gain, E[s^2 cost] / (E[sqrt(s^2 cost)])^2 over cells, which bounds what guiding can
  win.

## 2. Diffusion tail (biased)

Photons stop once they are deep: past K_min collisions, and where `bench_sun_reach`,
the least optical depth towards any sun direction, exceeds tau_deep. Each stopped
photon adds its weight to an isotropic source in its sun bin, on a diffusion grid of
16-voxel cells (63 x 43 x 77). A cell-centred finite-volume diffusion solve then gives
each bin's fluence from the bins' sources:

- D = 1 / (3 sigma_t (1 - g)) from the cell's mean density, with g the phase's mean
  cosine;
- harmonic-mean coupling between cloud cells;
- Marshak vacuum conditions where a cell borders the box or a cell thinner than
  tau_min, from which no light returns.

The probes take phi and F = -D grad phi, interpolated at each probe, into their bins.
This is the transfer between probes: each stopped photon feeds every probe through a
deterministic, noiseless solve.

- **Prototype offline first.** The GPU stops the photons and dumps the sources, and
  Python solves (scipy) and writes a field, so that the bias can be measured before a
  GPU solver is written. The sweep covers tau_deep and K_min.
- **If the bias is small,** a GPU solver: red-black SOR over all bins at once, bins
  innermost, warm-started from the last solve, as the sources only grow.
- **If the bias is not small,** an unbiased variant: stopped photons go on with
  probability p at weight 1 / p, and the source becomes w (1 - [goes on] / p), so that
  the solve corrects only what the survivors' own walks do not carry. It is unbiased
  for any p > 0, and as good as diffusion predicts the walks.

## Outcome

Both were built and measured; the results are under "Guided photon emission" and
"Diffusion tail for deep photons" in docs/research/20261022-cloud-bench.md. The
unbiased variant of the tail was not built, as it cannot keep the gain (see there); a
biased calibration by survivors took its place.

Each scheme is recorded in docs/research/20261022-cloud-bench.md: what was tried, the
tables, and why it won or lost. Code that does not win is reverted. Code that wins is
left uncommitted for review.

The GPU solver followed, red-black SOR as planned, warm-started, 50 sweeps per refit.
It reproduces the prototype's field; the Python solve was removed. Guided emission was
swept over its learning rounds, share and level floor, and stays at about 1.15 times as
efficient, near the bound its own score predicts.

Neither scheme was kept. The diffusion tail renders bands and a bias of up to a third
at higher density, and guiding no longer pays once the bake was made cheaper. What was
kept came after: persistent threads over a frame's rounds, and scattered light
deposited to all eight corners of its leaf (docs/research/20261022-cloud-bench.md,
"Cheaper rounds and the corners' noise").

