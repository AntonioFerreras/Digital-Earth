"""Named model specs of the encoding study, named as the inference benchmark names
its kernels (src/bench_neural.zig in the bench worktree) so rows match."""

HEADS = {"lin": (0, 0), "w16x1": (16, 1), "w32x1": (32, 1), "w16x2": (16, 2), "w32x2": (32, 2)}
# K-Planes' own initialization (uniform 0.1-0.5) multiplies six planes down to
# ~1e-3; around 1 every plane's gradient stays of order one.
KPLANES_INIT = (0.5, 1.5)
# Learning rates follow Instant-NGP and K-Planes: tables and planes 1e-2, heads alike.
LR_GRID = 1e-2
LR_HEAD = 1e-2
LR_MLP = 1e-2
# A coarser finest level than the benchmark's (32, 512, 1024, 64): 2^25 noisy samples
# give fewer than one per finest cell there, so those levels fit noise.
HASH_CELLS_MAX_COARSE = (16, 128, 256, 32)


def head_spec(head: str, activation: str) -> dict:
    width, hidden = HEADS[head]
    return {"width": width, "hidden": hidden, "activation": activation}


def specs() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for head in ("w16x2", "w32x2", "w32x1"):
        result[f"raw_{head}"] = {"candidate": "A", "encoding": {"kind": "raw"}, "head": head_spec(head, "gelu"),
                                 "lr_head": LR_MLP}
    result["raw_w64x4"] = {"candidate": "A", "encoding": {"kind": "raw"},
                           "head": {"width": 64, "hidden": 4, "activation": "gelu"}, "lr_head": LR_MLP}
    for octaves in (4, 6, 8):
        for head in ("w16x1", "w32x1", "w32x2"):
            result[f"fourier{octaves}_{head}"] = {"candidate": "B", "encoding": {"kind": "fourier", "octaves": octaves},
                                                  "head": head_spec(head, "relu"), "lr_head": LR_MLP}
    for levels in (8, 12, 16):
        for log2_table, heads in ((16, ("w16x1",)), (18, ("lin", "w16x1", "w32x1")), (20, ("w16x1",))):
            for head in heads:
                result[f"hash{levels}_{head}_t{log2_table}"] = {
                    "candidate": "C", "encoding": {"kind": "hash", "levels": levels, "log2_table": log2_table},
                    "head": head_spec(head, "relu"), "lr_grid": LR_GRID, "lr_head": LR_HEAD}
    for levels in (8, 12, 16):
        result[f"hash{levels}_w16x1_t18_coarse"] = {
            "candidate": "C",
            "encoding": {"kind": "hash", "levels": levels, "log2_table": 18, "cells_max": list(HASH_CELLS_MAX_COARSE)},
            "head": head_spec("w16x1", "relu"), "lr_grid": LR_GRID, "lr_head": LR_HEAD}
    for features in (8, 16, 32):
        for base, heads in ((16, ("lin", "w16x1", "w32x1")), (32, ("w16x1",))):
            for head in heads:
                result[f"kplanes{features}_{head}_s{base}"] = {
                    "candidate": "D",
                    "encoding": {"kind": "kplanes", "features": features, "base": base,
                                 "init_low": KPLANES_INIT[0], "init_high": KPLANES_INIT[1]},
                    "head": head_spec(head, "relu"), "lr_grid": LR_GRID, "lr_head": LR_HEAD}
    for vertices in ((32, 64, 64, 32), (8, 32, 128, 32), (16, 64, 256, 64), (32, 32, 128, 32)):
        name = "lut_" + "x".join(map(str, vertices))
        result[name] = {"candidate": "E", "encoding": {"kind": "dense", "vertices": list(vertices)}, "lr_grid": LR_GRID}
    for vertices in ((32, 32, 128, 32), (16, 64, 256, 64)):
        name = "bruneton_lut_" + "x".join(map(str, vertices))
        result[name] = {"candidate": "E", "encoding": {"kind": "bruneton", "vertices": list(vertices)}, "lr_grid": LR_GRID}
    for spec in result.values():
        spec.setdefault("output", "identity")
        spec.setdefault("loss", "rel")
        spec.setdefault("tv", 0.0)
    return result
