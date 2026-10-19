"""One training run of the encoding study: a fixed wall-clock budget on GPU-resident
data, then metrics on the clean reference points."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import datagen_format
from study import encodings, metrics, view_uvwz

DATA_FILES = {"16k": "train_16k.edgn", "1k": "train_1k.edgn", "lut": "lut_bruneton_32x32x128x32_64k.edgn"}
REFERENCE_FILE = "ref_clean.edgn"
# The tail of a training file is held out whole: neighbouring rows are not
# correlated, but a contiguous chunk is the honest choice if they ever are.
HOLDOUT_FRACTION = 0.05
VALIDATION_COUNT = 2**20
WARMUP_FRACTION = 0.02
LR_FINAL_FRACTION = 0.01
TIMING_STEPS = 30
# shaders/common/constants.slang: solar radius over the Earth-Sun distance.
SUN_ANGULAR_RADIUS = 6.95e8 / 1.4959e11


def renderable(records: np.ndarray) -> np.ndarray:
    """The rows a camera ray can produce, without those that look into the sun disk.
    A path that sees the disk returns ~1e6: a few such samples hold most of the
    loss, and no smooth model can represent the disk anyway."""
    nu = view_uvwz.scattering_params_from_uvwz(records[:, :4])[3]
    sun_disk = nu > np.cos(SUN_ANGULAR_RADIUS)
    keep = view_uvwz.reachable(records[:, :4]) & ~sun_disk
    print(f"dropped {sun_disk.sum()} sun-disk and {(~keep).sum() - sun_disk.sum()} unreachable "
          f"of {len(records):,} rows", flush=True)
    return records[keep]


def load_training(data_dir: Path, data: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(train, validation) record tensors on `device`; the LUT trains whole and
    validates on the held-out tail of train_16k."""
    records = torch.from_numpy(renderable(datagen_format.read_records(data_dir / DATA_FILES[data])))
    if data == "lut":
        holdout = torch.from_numpy(renderable(datagen_format.read_records(data_dir / DATA_FILES["16k"])))
        holdout = holdout[-int(len(holdout) * HOLDOUT_FRACTION):]
        train = records
    else:
        split = len(records) - int(len(records) * HOLDOUT_FRACTION)
        train, holdout = records[:split], records[split:]
    return train.to(device), holdout[:VALIDATION_COUNT].to(device)


def loss_function(kind: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if kind == "rel":
        # The detached denominator keeps the gradient unbiased under zero-mean target noise.
        return ((prediction - target) ** 2 / (prediction.detach() ** 2 + metrics.EPS)).mean()
    assert kind == "mse"
    return ((prediction - target) ** 2).mean()


def total_variation(grid: encodings.GridLevels) -> torch.Tensor:
    """Mean squared relative difference between neighbouring vertices of a dense RGB grid."""
    assert grid.level_count == 1
    n = [c + 1 for c in grid.cells_list[0]]
    values = grid.table.view(n[3], n[2], n[1], n[0], 3)
    scale = values.detach() ** 2 + metrics.EPS
    total = values.new_zeros(())
    for dim in range(4):
        difference = values.diff(dim=dim)
        total = total + (difference**2 / scale.narrow(dim, 0, difference.shape[dim])).mean()
    return total


@torch.no_grad()
def initialize_dense(model: encodings.Model, train: torch.Tensor, chunk: int) -> None:
    """Sets each vertex to the interpolation-weighted mean of the targets around it,
    a close start that Adam's fixed step size could not reach from zero."""
    grid = model.encoding
    assert grid.level_count == 1
    numerator = torch.zeros_like(grid.table)
    denominator = torch.zeros(grid.table.shape[0], device=train.device)
    for start in range(0, len(train), chunk):
        rows = train[start:start + chunk]
        index, weight = grid.corners(model.encoding_input(rows[:, :4]))
        denominator.index_add_(0, index.reshape(-1), weight.reshape(-1))
        numerator.index_add_(0, index.reshape(-1), (weight[:, :, None] * rows[:, None, 4:]).reshape(-1, 3))
    # A vertex no sample touches gets no gradient either; it stays zero and only
    # weighs into lookups that no training sample resembles.
    empty = denominator == 0
    print(f"{int(empty.sum()):,} of {len(empty):,} vertices without samples", flush=True)
    grid.table.copy_(numerator / denominator.clamp(min=1e-30)[:, None])


def make_optimizer(model: encodings.Model, spec: dict) -> torch.optim.Optimizer:
    groups = []
    grid_parameters = list(model.encoding.parameters())
    head_parameters = [] if model.head is None else list(model.head.parameters())
    if spec["encoding"]["kind"] in ("raw", "fourier"):
        assert not grid_parameters
    if grid_parameters:
        groups.append({"params": grid_parameters, "lr": spec["lr_grid"]})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": spec["lr_head"]})
    # Instant-NGP's Adam: beta2 0.99 adapts to the sparse table gradients, eps tiny for their small magnitudes.
    return torch.optim.Adam(groups, betas=(0.9, 0.99), eps=1e-15)


def learning_rate_factor(step: int, step_count: int) -> float:
    """Linear warmup, then cosine decay to LR_FINAL_FRACTION."""
    warmup = max(1, int(step_count * WARMUP_FRACTION))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, step_count - warmup)
    return LR_FINAL_FRACTION + (1.0 - LR_FINAL_FRACTION) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model: torch.nn.Module, uvwz: torch.Tensor, chunk: int) -> torch.Tensor:
    model.eval()
    result = torch.cat([model(uvwz[start:start + chunk]) for start in range(0, len(uvwz), chunk)])
    model.train()
    return result


def run(spec: dict, data_dir: Path, out_dir: Path, budget_s: float, batch_size: int, seed: int) -> dict:
    """Trains `spec` for `budget_s` seconds of steps and writes model.pt and result.json to `out_dir`."""
    assert budget_s > 0
    assert batch_size > 0
    torch.manual_seed(seed)
    device = torch.device("cuda")
    train, validation = load_training(data_dir, spec["data"], device)
    model = encodings.build(spec).to(device)
    if spec["encoding"]["kind"] in ("dense", "bruneton"):
        initialize_dense(model, train, chunk=2**20)
    optimizer = make_optimizer(model, spec)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    generator = torch.Generator(device=device).manual_seed(seed)

    def step(lr_factor: float) -> torch.Tensor:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * lr_factor
        rows = train[torch.randint(len(train), (batch_size,), device=device, generator=generator)]
        loss = loss_function(spec["loss"], model(rows[:, :4]), rows[:, 4:])
        if spec["tv"] > 0:
            loss = loss + spec["tv"] * total_variation(model.encoding)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.detach()

    # The step count is fixed from a timed probe, so the schedule is a function of
    # steps and every run spends the same wall clock.
    for _ in range(5):
        step(1e-3)
    torch.cuda.synchronize()
    probe_start = time.perf_counter()
    for _ in range(TIMING_STEPS):
        step(1e-3)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - probe_start) / TIMING_STEPS
    step_count = max(100, int(budget_s / step_s))
    print(f"{spec['name']}: {encodings.parameter_count(model):,} parameters, {step_s * 1e3:.2f} ms/step, "
          f"{step_count:,} steps", flush=True)

    start = time.perf_counter()
    running = torch.zeros((), device=device)
    for index in range(step_count):
        running += step(learning_rate_factor(index, step_count))
        if (index + 1) % max(1, step_count // 10) == 0:
            print(f"  step {index + 1:,}: loss {running.item() / max(1, step_count // 10):.5f}", flush=True)
            running.zero_()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - start

    validation_prediction = predict(model, validation[:, :4], chunk=2**16)
    validation_loss = float(loss_function("rel", validation_prediction, validation[:, 4:]))
    reference = datagen_format.read_records(data_dir / REFERENCE_FILE)
    prediction = predict(model, torch.from_numpy(reference[:, :4]).to(device), chunk=2**16).cpu().numpy()
    result = {
        "spec": spec,
        "parameters": encodings.parameter_count(model),
        "steps": step_count,
        "samples_seen": step_count * batch_size,
        "train_s": train_s,
        "step_ms": 1e3 * train_s / step_count,
        "validation_rel_l2_noisy": validation_loss,
        "metrics": metrics.evaluate(prediction, reference),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.save(out_dir / "ref_prediction.npy", prediction.astype(np.float32))
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    m = result["metrics"]
    print(f"  {spec['name']}: rel_l2 {m['rel_l2']:.3e}, rel_mean {m['rel_mean']:.4f}, rel_p99 {m['rel_p99']:.4f}, "
          f"val {validation_loss:.4e}, {train_s:.0f} s", flush=True)
    return result
