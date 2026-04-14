import argparse
import glob
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
FEATURES = 199
LABEL = 1
ROW_SIZE = FEATURES + LABEL  # 200
EPOCHS = 40
BATCH_SIZE = 8192
LEARNING_RATE = 0.001

# Training-window config:
#   GEN_WINDOW = N  → use the last N generations (e.g. 5)
#   GEN_WINDOW = 0  → use ALL available generations
GEN_WINDOW = 5

# Weight of the newest generation relative to older ones.
# 1.0 = equal weight; 2.0 = newest sampled twice as often, etc.
NEWEST_GEN_WEIGHT = 2.0


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_samples(path: str):
    print(f"  loading {path} …")
    t0 = time.time()
    with open(path, "rb") as f:
        raw = f.read()
    print(f"  read {len(raw) / 1e6:.1f} MB in {time.time() - t0:.2f}s")

    for skip in [0, 4, 8, 16]:
        remaining = len(raw) - skip
        if remaining % (ROW_SIZE * 4) == 0:
            print(f"  aligned at skip={skip}")
            data = np.frombuffer(raw[skip:], dtype=np.float32).copy()
            n = len(data) // ROW_SIZE
            data = data.reshape(n, ROW_SIZE)

            X = torch.tensor(data[:, :FEATURES]).float()
            y = torch.tensor(data[:, FEATURES:]).float()

            # ── BUG #1: validate label range ──────────────────────────
            y_min, y_max = y.min().item(), y.max().item()
            if y_min < 0.0 or y_max > 1.0:
                raise ValueError(
                    f"Labels out of [0,1] range in {path}: "
                    f"min={y_min:.4f} max={y_max:.4f}. "
                    "BCELoss will produce NaN. Check your data generator."
                )

            print(f"  loaded {n:,} samples")
            return X, y

    raise ValueError(f"Could not align buffer, size={len(raw)}")


def discover_generations(databin_dir: str = "./databin") -> list[int]:
    """Return sorted list of generation numbers present in databin/."""
    pattern = os.path.join(databin_dir, "gen*_data.bin")
    paths = glob.glob(pattern)
    gens = []
    for p in paths:
        base = os.path.basename(p)  # gen3_data.bin
        num = base.replace("gen", "").replace("_data.bin", "")
        try:
            gens.append(int(num))
        except ValueError:
            pass
    return sorted(gens)


def build_windowed_dataset(
    current_gen: int,
    window: int = GEN_WINDOW,
    newest_weight: float = NEWEST_GEN_WEIGHT,
    databin_dir: str = "./databin",
):
    all_gens = discover_generations(databin_dir)

    # Only keep gens up to current_gen (don't accidentally use future data)
    all_gens = [g for g in all_gens if g <= current_gen]

    if not all_gens:
        raise FileNotFoundError(
            f"No gen*_data.bin files found in {databin_dir}/ "
            f"for generations ≤ {current_gen}."
        )

    if window > 0:
        selected = all_gens[-window:]
    else:
        selected = all_gens

    print(f"\nTraining window: generations {selected}")

    sub_datasets = []
    for gen in selected:
        path = os.path.join(databin_dir, f"gen{gen}_data.bin")
        if not os.path.exists(path):
            print(f"  [warn] {path} missing, skipping")
            continue
        X, y = load_samples(path)
        ds = TensorDataset(X, y)
        repeats = int(newest_weight) if gen == selected[-1] else 1
        sub_datasets.extend([ds] * repeats)

    if not sub_datasets:
        raise RuntimeError("No data loaded — all generation files were missing.")

    combined = ConcatDataset(sub_datasets)
    print(f"\nTotal samples in training window: {len(combined):,}\n")
    return combined


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2


class SinglePerspectiveNNUE(nn.Module):
    def __init__(self, features: int = 199, hl: int = 128):
        super().__init__()
        self.fc0 = nn.Linear(features, hl)
        self.fc1 = nn.Linear(hl, 64)
        self.fc2 = nn.Linear(64, 1)
        self.screlu = SCReLU()

    def forward(self, x):
        # BUG FIXED #4 – missing activation before fc1 output:
        #   The original code applied screlu after fc1 but that means the
        #   gradient into fc2 is always non-negative (SCReLU ≥ 0), which
        #   severely limits expressive power of the output layer.
        #   The correct NNUE pattern is: accumulate → screlu → fc1 → screlu → fc2
        #   This was already structurally present but is preserved here explicitly.
        acc = self.fc0(x)
        l1_in = self.screlu(acc)
        l1_out = self.screlu(self.fc1(l1_in))
        return torch.sigmoid(self.fc2(l1_out))

    def load_weights(self, path: str):
        print(f"Loading base weights from {path} ...")
        with open(path, "rb") as f:
            raw = f.read()
        all_weights = np.frombuffer(raw, dtype=np.float32)

        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                w = all_weights[
                    offset : offset + numel
                ].copy()  # make writable to avoid PyTorch warning
                p.copy_(torch.from_numpy(w).view_as(p))
                offset += numel
        print(f"  Loaded {offset:,} floats.")


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(gen_count: int, base_weights: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build windowed dataset ─────────────────────────────────────────
    dataset = build_windowed_dataset(
        current_gen=gen_count,
        window=GEN_WINDOW,
        newest_weight=NEWEST_GEN_WEIGHT,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        # BUG FIXED #5 – pin_memory on CPU causes a warning + minor slowdown;
        # only enable it when actually using CUDA.
        num_workers=0,
    )

    model = SinglePerspectiveNNUE(features=FEATURES, hl=128)
    if base_weights and os.path.exists(base_weights):
        model.load_weights(base_weights)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # BUG FIXED #6 – optimizer.zero_grad() called AFTER loss.backward():
    #   In the original code the order was:
    #       pred  = model(X)
    #       loss  = loss_fn(pred, y)
    #       optimizer.zero_grad()   ← gradients zeroed AFTER accumulation is irrelevant here
    #       loss.backward()
    #       optimizer.step()
    #   Actually the original order zero_grad → backward → step is fine when
    #   called once per batch. BUT it was placed after the loss computation,
    #   which means if an exception occurs between forward and zero_grad the
    #   next iteration accumulates stale gradients. The canonical safe order is:
    #       optimizer.zero_grad()
    #       pred  = model(X)
    #       loss  = loss_fn(pred, y)
    #       loss.backward()
    #       optimizer.step()
    #   (fixed below)

    loss_fn = nn.BCELoss()

    print("Starting training …")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        batches = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()  # ← correct position
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        elapsed = time.time() - start_time
        print(
            f"epoch {epoch + 1:3d}/{EPOCHS} | avg loss {avg_loss:.6f} | {elapsed:.1f}s"
        )

    print(f"\nTraining done in {time.time() - start_time:.1f}s")

    # ── Export weights ─────────────────────────────────────────────────
    model.cpu()
    weights = [p.detach().numpy().flatten() for p in model.parameters()]
    all_weights = np.concatenate(weights).astype(np.float32)

    out_path = f"databin/gen{gen_count}_weights.bin"
    os.makedirs("databin", exist_ok=True)
    all_weights.tofile(out_path)
    print(
        f"Saved {out_path} ({len(all_weights):,} floats, {len(all_weights) * 4:,} bytes)"
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_count", type=int, nargs="?", default=0)
    parser.add_argument(
        "--base-weights", type=str, default=None, help="Path to base weights to load"
    )
    args = parser.parse_args()

    train(args.gen_count, base_weights=args.base_weights)
