# MLP training

The renderer's `neural` mode evaluates a small MLP (`model.py`: uvwz in, linear
Rec.709 radiance out), or a network of the encoding study (`study/`), trained offline in PyTorch on path-traced samples. Every
command runs from the repository root in the `py` environment.

## 1. Training data

Training reads `.edgn` binary files (`datagen_format.py`, written by
`earth datagen`) and text files of `u v w z R G B` lines.

`earth datagen` path-traces the symmetric scene at uvwz points, averaging
`--paths` spectral paths per point, and streams the records to `--out`:

```sh
pixi run zig build -Doptimize=ReleaseSafe
# The original data_prep.py recipe: uniform uvwz, 10% of points with v scaled by 0.4.
./zig-out/bin/earth datagen --count 33554432 --paths 16384 --seed 1 --out cache/training/train_16k.edgn
# Texel centres of a 4D table, u fastest: record ((z * NW + w) * NV + v) * NU + u.
./zig-out/bin/earth datagen --sampling grid --grid 32,32,128,32 --paths 65536 --out cache/training/lut.edgn
# The uvwz of existing records, re-evaluated (e.g. clean references for held-out points).
./zig-out/bin/earth datagen --points held_out.edgn --paths 4194304 --seed 2 --out cache/training/held_out_clean.edgn
```

A seed fixes every output bit, whatever the GPU's timing. Its radiance is the
renderer's, which is `export_weights.TAICHI_OUTPUT_SCALE` (2^7.406, about
169.6) times the original Taichi data's; do not mix the two in one training run.
Unlike the original, targets are not clamped at zero: the Rec.709 mean of a
saturated spectrum can be slightly negative.

Text is slow to parse and memory-hungry as Python floats, so convert an
existing text dataset once:

```sh
pixi run -e py convert-training-data --source_dir ~/Digital-Earth/uvwz_rgb_data \
                                     --out_dir cache/training/uvwz_rgb
```

Each text file becomes one `<stem>.edgn` file, and a sample of each file's rows
is checked bit for bit against Python's `float()` parse of the source line.

## 2. Training

```sh
pixi run -e py train --data_dir cache/training/uvwz_rgb --out_dir model_weights
```

The defaults are the settings of the shipped weights: 120 epochs, batch 512,
learning rate `3e-4 * batch_size / 2048` with cosine decay to a tenth of that
over the whole run. A checkpoint `mlp_e<epoch>.pth` is written to `--out_dir`
every `--save_every` epochs (default every epoch). Validation draws a random
`--val_split` of all samples, which training also sees, so the validation loss
measures fit rather than generalization.

## 3. Export

```sh
pixi run -e py export-weights --checkpoint model_weights/mlp_e120.pth --training-data taichi \
                              --out cache/mlp/mlp_e120.emlp
```

`--training-data` records in the file how the renderer scales the network's
output: `datagen` for networks trained on `earth datagen` output, `taichi` for
the original text data. Files exported before this flag existed (EMLP version 1)
load as `taichi`. `--run cache/study/runs/<run>` exports an encoding-study run
instead. The export checks a NumPy forward pass of the file against torch
before writing it. `cache/mlp/mlp_e120.emlp` is what the renderer loads when no
`--weights` is given.

## 4. Render

```sh
pixi run zig build
pixi run ./zig-out/bin/earth render --mode neural --weights cache/mlp/mlp_e120.emlp --out mlp.png
```
