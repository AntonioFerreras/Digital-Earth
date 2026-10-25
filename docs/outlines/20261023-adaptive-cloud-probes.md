# Adaptive cloud probes: outline

The dense 64^3 grid spends its probes where they are not needed and lacks them where
they are (`docs/research/20261022-cloud-bench.md`, Where the light varies):

- 73 % of its probes lie in air away from cloud and are never read.
- Past about one mean free path inside, linear interpolation holds to the walks' noise
  wherever the spacing is under about 2.5 times the distance to the surface, at
  extinctions 0.1 and 0.453 alike.
- The surface layer varies fastest and is already the worst at the dense spacing; it
  is where the grid shows.

An adaptive layout puts the probes along the surface at a finer spacing than the dense
grid and thins them with depth, under the same budget or a smaller one. It stays a
lattice, so that the lookup stays one fetch and a hardware trilinear filter.

## Layout

An octree of cells over the volume's box. The finest cells have edge h0; a cell of
level L has edge h0 2^L. Probes sit at the corners of the leaf cells.

- **Which cells:** a cell is needed if it holds cloud: a density texel within the
  density's filter footprint. A march step only reads where the extinction is above
  zero, and interpolates its own cell's corners, so air cells hold no probes.
- **How coarse:** a needed cell may be of level L if h0 2^L <= k d over it, d the
  distance to the cloud's surface (the volume's distance texture, its minimum over
  the cell) and k = 2.5 from the measurement. A cell holding the surface has d = 0
  and is of level 0.
- **Building:** from the coarsest level down, a cell becomes a leaf when every needed
  finest cell inside it allows its level, and is split into eight otherwise. Probe
  positions are lattice points of the finest level, deduplicated, so that leaves
  sharing a corner share its probe.
- **Budget:** the probe count falls monotonically with h0, so h0 is found by
  bisection to meet a budget given in probes. It depends only on the density, not on
  extinction or light: built on the CPU once when the volume loads.

Counts on the bench cloud (k = 2.5):

| h0 (x dense spacing) | probes  | (x 64^3) | leaf cells | atlas texels |
| -------------------- | ------- | -------- | ---------- | ------------ |
| 1                    | 52776   | 0.20     | 39202      | 313616       |
| 1/2                  | 261380  | 1.00     | 192612     | 1540896      |
| 3/8                  | 496835  | 1.90     | 359615     | 2876920      |

The surface takes nearly all of the probes; this cloud's billows have a large area.
So the budget buys either:

- **Same budget:** h0 at half the dense spacing, the surface 2 times finer per axis.
- **A fifth of the budget:** the dense grid's surface resolution with a fifth of the
  walks. A walk costs more on average, though: nearly every probe starts in cloud,
  where most of the dense grid's start in air and leave at once.

## Storage and lookup

- **Indirection:** a 3D texture of r32uint, one texel per finest cell (159 x 88 x 129
  at h0 = 1/2, 7 MB): the leaf holding it, as its slot in the atlas and its level.
  Rays step coherently, so neighbouring threads load the same texels.
- **Atlas:** a 3D texture, rgba16f, of 2 x 2 x 2 texels per leaf, its eight corners'
  (phi, F), one slab per colour channel as today. A step loads its indirection texel,
  places itself within the leaf, and samples the atlas at that leaf's texels with the
  linear filter: exact trilinear interpolation of its corners. Cost: one load more
  than the dense grid.
- **Atlas map:** per atlas texel, the probe it holds (r32uint). After lighting, a
  pass copies each probe's light into its texels. Each probe appears in about six
  leaves.

Bricks of more probes per leaf (3^3, 4^3) shrink the indirection but tie more probes
to each surface leaf: at h0 = 1/2, 3^3 bricks take 347k probes against 261k. Single
cells waste no probes, and the indirection is small at the bench's size.

## Seams

Where a fine leaf meets a coarse one, the fine leaf's corner at the middle of the
shared face is a probe of its own, while the coarse leaf interpolates its four
corners. The jump there is the coarse leaf's interpolation error, which the level
rule bounds at about the walks' noise. First form: bake these hanging probes like any
other and measure. If the seams show, they are set from the coarse face's
interpolation instead (with a 2:1 balance between neighbours), which also removes
them from the bake.

## Bake, fit and lighting

The walk, the fit and the lighting do not change, only how they find a probe:

- One dispatch over the probe list, its positions in a buffer, instead of the grid.
- The sums, the fits and the light are indexed by probe.
- `ProbeCache.Key` adds the layout's h0 and k.

## Plan

Steps 1 to 4 are built; the measurements are in the research doc (Adaptive layout).

1. **Layout builder** (Zig, `src/render/`): cell statistics from the volume's distance
   and density, the octree, deduplicated probes, bisection to the budget; outputs the
   positions, the indirection and the atlas map. Tests: every cell holding cloud is
   covered by a leaf; the count falls with h0; leaves sharing a corner share a probe.
2. **Probe list in the bake, fit and lighting passes,** and the atlas fill.
3. **Lookup** in `probe_radiance`; the dense grid kept as a layout setting for
   comparison.
4. **Settings:** layout (dense or adaptive), probe budget, k.
5. **Measure:** bake time per round, the cell-scale error front-lit and backlit, by
   eye, at the 64^3 budget and at a fifth of it.

## Risks

- **k from one cloud:** the 2.5 holds at two extinctions on the bench; thin, sparse
  cloud may want another.
- **Noise:** finer probes average less of their neighbours' noise, so the same walks
  leave visible blotches where the dense grid's wider cells smoothed them; at the same
  budget a bake frame costs about a quarter more.
- **Distance threshold:** the distance texture's surface is a density threshold;
  wisps below it count as air at depth zero, which the level rule already treats as
  surface.
- **The earth:** the same leaves and atlas, with the indirection as a clipmap and the
  layout built per brick on the GPU as clouds stream in.
