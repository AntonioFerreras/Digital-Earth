# Plan: encoding study for the MLP atmosphere mode

Question: which representation of the 4D function uvwz -> linear Rec.709 RGB
(symmetric scene) gives the best error against noise-free references for a
given inference cost (budget: < 10 ms per frame at 2560x1440 on the RTX 4090,
including ray -> uvwz) and parameter count, when trained on Monte Carlo noisy
targets?

## Data (from `earth datagen`, new normalized brightness scale)

- `ref_clean.edgn`: 65,536 held-out uniform-recipe points, ~2^22 paths each.
- `train_16k.edgn`, `train_1k.edgn`: 2^25 samples at 16,384 / 1,024 paths.
- `lut_*.edgn`: dense 4D grid at texel centres, 65,536 paths per texel.
- Recipe: uvwz ~ U[0,1]^4, 10% with uvwz.y scaled by 0.4 (sunset).
- Axis order uvwz = (nu, mu_s, mu, r) as in shaders/atmosphere/bruneton.slang.

## Candidates (sizes are starting points; sweep small/medium/large)

Heads stay tiny: encodings carry the capacity. Decoder heads are 0-2 hidden
layers of width 16 or 32; 64-wide only as an upper reference.

| id | representation | notes |
| --- | --- | --- |
| A | small plain MLPs: 4 -> 16 -> 16 -> 3, 4 -> 32 -> 32 -> 3, 4 -> 32 -> 3 | baseline; the current 4 -> 64 x4 -> 3 is a single reference row only (too slow, no quality gain from its size) |
| B | Fourier features (per-axis sin/cos, L octaves) + MLP | NeRF-style frequency encoding; L and base frequency per axis tuned |
| C | 4D multiresolution hash grid (Instant-NGP) + tiny MLP | L levels, F=2 features, table 2^T per level, per-axis resolutions (nu coarse, mu fine), quadrilinear (16 corners) |
| D | K-Planes (6 planes over axis pairs, multiscale, Hadamard product) + tiny MLP | no hashing |
| E | dense 4D LUT fitted to noisy data (least squares + optional smoothness) | "grid with linear decoder" |
| F | dense 4D LUT baked directly at high spp | no training |

## Protocol

- Loss: relative L2 with detached denominator, `(p - y)^2 / (p.detach()^2 + eps)`,
  unbiased under zero-mean target noise; MSE reported too.
- Same wall-clock training budget per candidate; validation on held-out data,
  final metrics only on `ref_clean`: relative L2, MSE, max error, error by
  region (horizon |mu| small, terminator mu_s ~ 0, forward peak nu ~ 1).
- Noise robustness: train on `train_1k` and `train_16k`; report the gap.
- Report parameters / memory, inference ms at 2560x1440 (Slang kernels), and
  renders of default, sunset, and limb views against the path tracer.
