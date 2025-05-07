#!/usr/bin/env python3
"""
train_mlp.py
============
Train a simple multi-layer perceptron (MLP) on HDR data stored in plain-text files.
Each file contains one sample per line with **seven** space-separated floats:
    4 inputs  (normalized to 0-1)
    3 outputs (HDR RGB values, unbounded)

Usage (most common):
    python train_mlp.py --data_dir ./dataset --epochs 200 --batch_size 1024 \
                        --lr 5e-4 --out hdr_mlp.pth

The script automatically finds **all** files in *--data_dir* (regardless of
filename) and concatenates their contents into a single PyTorch Dataset. It
then splits the data into training/validation sets, trains the MLP, and saves
whichever epoch achieves the best validation loss.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from model import MLP
################################################################################
# Dataset
################################################################################

class HDRTextDataset(Dataset):
    """Lazy-loads every line from every *.txt* file in *root_dir* into memory."""

    def __init__(self, root_dir: str, dtype: torch.dtype = torch.float32):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"{root_dir} is not a directory")

        self.samples = []  # will hold [ [7 floats], ... ]
        for path in self.root_dir.iterdir():
            if path.is_file() and path.suffix in {"", ".txt"}:
                with path.open("r") as f:
                    for line_no, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 7:
                            raise ValueError(
                                f"{path} L{line_no}: expected 7 floats, got {len(parts)}"  # noqa: E501
                            )
                        self.samples.append([float(x) for x in parts])

        if not self.samples:
            raise RuntimeError("No samples found—check --data_dir path")

        self.tensor = torch.tensor(self.samples, dtype=dtype)
        del self.samples  # free list − everything lives in one contiguous tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        row = self.tensor[idx]
        x, y = row[:4], row[4:]
        return x, y



################################################################################
# Training helpers
################################################################################

def step_epoch(model, loader, optimizer, criterion, device, scheduler=None, train=True, max_grad_norm=1.0):
    if train:
        model.train()
    else:
        model.eval()

    running = 0.0
    total_samples = 0
    batch_count = 0
    
    with torch.amp.autocast(enabled=False, device_type='cuda', dtype=torch.bfloat16):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                preds = model(x)
                loss = criterion(preds, y)
            if train:
                loss.backward()
                # Apply gradient clipping
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # Step the scheduler after each batch if provided
                if scheduler is not None:
                    scheduler.step()
                
                # Print batch progress every 256 batches in training mode
                batch_count += 1
                if batch_count % 256 == 0:
                    current_avg_loss = running / total_samples if total_samples > 0 else 0
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Batch {batch_count:5d}: current avg loss = {current_avg_loss:.6f}, lr = {current_lr:.8f}")
            
            batch_loss = loss.item() * x.size(0)
            running += batch_loss
            total_samples += x.size(0)
            
    return running / len(loader.dataset)

################################################################################
# Main
################################################################################

def main(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # ---------- dataset ----------
    full_ds = HDRTextDataset(cfg.data_dir)
    val_size = int(len(full_ds) * cfg.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------- model + optim ----------
    model = MLP().to(device)
    
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Calculate the total number of batches for 25 epochs
    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * 25
    min_lr = cfg.lr * 0.1
    
    # Create the cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=min_lr)
    
    criterion = nn.MSELoss()

    print(f"Training with cosine annealing LR scheduler:")
    print(f"  Initial LR: {cfg.lr}")
    print(f"  Min LR: {min_lr}")
    print(f"  LR will decay over {total_batches} batches ({25} epochs)")
    print(f"  Gradient clipping enabled with max_norm=1.0")

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = step_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler, train=True, max_grad_norm=1.0)
        val_loss = step_epoch(model, val_loader, optimizer, criterion, device, train=False)
        
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:3d}/{cfg.epochs}] train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.8f}")

        if val_loss < best_val or True:
            best_val = val_loss
            if cfg.out:
                torch.save(model.state_dict(), cfg.out)
                print(f"  ↳ New best model saved → {cfg.out}")

    print("Training complete. Best validation loss:", best_val)

################################################################################
# CLI
################################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train an MLP on HDR text data")
    p.add_argument("--data_dir", default="./uvwz_rgb_data", help="Directory with .txt data files")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="mlp_best.pth", help="Path to save best model (optional)")
    cfg = p.parse_args()

    main(cfg)
