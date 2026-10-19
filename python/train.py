#!/usr/bin/env python3
"""
train.py
========
Train the 4 -> 3 MLP on HDR samples `u v w z R G B`: 4 inputs normalized to
0-1 and 3 unbounded HDR RGB outputs.

Every file in *--data_dir* is read: `.edgn` files are the binary format written
by `earth datagen` or `convert_text_data.py` (see `datagen_format.py`); files
with no suffix or `.txt` hold one sample per line as seven space-separated
floats. Text loads far slower, so convert large text datasets first.

Usage (most common):
    python python/train.py --data_dir cache/training/uvwz_rgb --epochs 120 --batch_size 512

The validation set is a random subset of all samples; training iterates over
every sample in contiguous blocks of *--batch_size*, so the two sets overlap.
The weights are saved every *--save_every* epochs to *--out_dir* as
`mlp_e<epoch>.pth`.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import datagen_format
from model import MLP

TEXT_SUFFIXES = {"", ".txt"}
LOG_INTERVAL_SAMPLES = 524288

################################################################################
# Dataset
################################################################################


class HDRDataset(Dataset):
    """Every binary and text data file in *root_dir*, loaded into memory as one (N, 7) tensor."""

    def __init__(self, root_dir: Path):
        if not root_dir.is_dir():
            raise FileNotFoundError(f"{root_dir} is not a directory")
        paths = sorted(path for path in root_dir.iterdir()
                       if path.is_file() and (path.suffix == datagen_format.SUFFIX or path.suffix in TEXT_SUFFIXES))
        if not paths:
            raise RuntimeError(f"No data files found in {root_dir}")
        print(f"Found {len(paths)} data files in {root_dir}")

        chunks = [datagen_format.read_records(path) if path.suffix == datagen_format.SUFFIX
                  else datagen_format.read_text(path)
                  for path in tqdm(paths, desc="Reading data files")]
        self.tensor = torch.from_numpy(np.concatenate(chunks))
        assert self.tensor.dtype == torch.float32
        print(f"Successfully loaded {self.tensor.size(0)} samples into memory")

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        row = self.tensor[idx]
        return row[:4], row[4:]


class BlockDataset(Dataset):
    """
    Contiguous [batch_size, 7] blocks of an in-memory tensor, each served as one
    batch. Call `shuffle_blocks()` once per epoch to randomise the block order;
    the samples inside a block stay contiguous.
    """

    def __init__(self, full_tensor: torch.Tensor, batch_size: int):
        assert full_tensor.dim() == 2 and full_tensor.size(1) == 7
        self.data = full_tensor
        self.batch_size = batch_size
        self.block_ids = list(range(full_tensor.size(0) // batch_size))  # drops the last partial block

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, block_idx):
        start = self.block_ids[block_idx] * self.batch_size
        # A worker sends a view's whole storage to the main process: without the
        # copy, every worker moves its own copy of the full dataset into shared memory.
        rows = self.data[start:start + self.batch_size].clone()
        return rows[:, :4], rows[:, 4:]

    def shuffle_blocks(self):
        random.shuffle(self.block_ids)


################################################################################
# Training helpers
################################################################################


def step_epoch(model, loader, optimizer, criterion, device, scheduler=None, train=True, max_grad_norm=1.0):
    """One pass over *loader*; returns the mean per-sample loss."""
    model.train(train)

    # Accumulated on the device so the loop synchronizes only when it logs.
    running = torch.zeros((), device=device)
    total_samples = 0
    samples_since_last_log = 0
    last_log_time = torch.cuda.Event(enable_timing=True)
    current_log_time = torch.cuda.Event(enable_timing=True)
    epoch_start_time = torch.cuda.Event(enable_timing=True)
    epoch_end_time = torch.cuda.Event(enable_timing=True)
    epoch_start_time.record()
    last_log_time.record()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss = criterion(model(x), y)
        running += loss.detach() * x.size(0)
        total_samples += x.size(0)
        if train:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            samples_since_last_log += x.size(0)
            if samples_since_last_log >= LOG_INTERVAL_SAMPLES:
                current_log_time.record()
                current_log_time.synchronize()
                elapsed_s = last_log_time.elapsed_time(current_log_time) / 1000.0
                print(f"  Samples {total_samples:,}: loss = {running.item() / total_samples:.6f}, "
                      f"lr = {optimizer.param_groups[0]['lr']:.8f}, "
                      f"speed = {samples_since_last_log / elapsed_s:.1f} samples/sec")
                samples_since_last_log = 0
                last_log_time.record()

    epoch_end_time.record()
    epoch_end_time.synchronize()
    epoch_elapsed_s = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
    mode = "Training" if train else "Validation"
    print(f"  {mode} complete: {total_samples:,} samples in {epoch_elapsed_s:.2f}s "
          f"({total_samples / epoch_elapsed_s:.1f} samples/sec avg)")
    return running.item() / total_samples


################################################################################
# Main
################################################################################


def main(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved in {cfg.out_dir.absolute()}")

    # ---------- dataset ----------
    full_ds = HDRDataset(cfg.data_dir)
    val_size = int(len(full_ds) * cfg.val_split)
    train_size = len(full_ds) - val_size
    _, val_ds = random_split(full_ds, [train_size, val_size])

    train_set = BlockDataset(full_ds.tensor, batch_size=cfg.batch_size)

    train_loader = DataLoader(
        train_set,
        batch_size=None,  # the dataset already serves whole batches
        shuffle=False,  # the dataset shuffles its blocks
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ---------- model + optim ----------
    model = torch.compile(MLP().to(device))

    lr = cfg.lr * (cfg.batch_size / 2048)
    min_lr = lr * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    total_batches = len(train_loader) * cfg.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=min_lr)
    criterion = nn.MSELoss()

    print("Training with cosine annealing LR scheduler:")
    print(f"  Initial LR: {lr}")
    print(f"  Min LR: {min_lr}")
    print(f"  LR will decay over {total_batches} batches ({cfg.epochs} epochs)")
    print("  Gradient clipping enabled with max_norm=1.0")
    print(f"  Models will be saved every {cfg.save_every} epochs in {cfg.out_dir}")

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_set.shuffle_blocks()
        train_loss = step_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler, train=True, max_grad_norm=1.0)
        val_loss = step_epoch(model, val_loader, optimizer, criterion, device, train=False)
        print(f"[Epoch {epoch:3d}/{cfg.epochs}] train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.8f}")

        if epoch % cfg.save_every == 0:
            save_path = cfg.out_dir / f"mlp_e{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Model saved at epoch {epoch} → {save_path}")

        best_val = min(best_val, val_loss)

    print("Training complete. Best validation loss:", best_val)


################################################################################
# CLI
################################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train an MLP on HDR uvwz -> RGB data")
    p.add_argument("--data_dir", type=Path, default=Path("cache/training/uvwz_rgb"), help="Directory with .edgn or .txt data files")
    p.add_argument("--out_dir", type=Path, default=Path("model_weights"), help="Directory the mlp_e<epoch>.pth checkpoints are saved to")
    p.add_argument("--save_every", type=int, default=1, help="Save a checkpoint every this many epochs")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate at batch size 2048, scaled linearly with --batch_size")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()

    main(cfg)
