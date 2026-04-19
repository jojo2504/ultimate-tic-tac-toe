import argparse
import glob
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
FEATURES = 199
SCORE = 1
LABEL = 1
PLY = 1
ROW_SIZE = FEATURES + SCORE + LABEL + PLY  # 202
EPOCHS = 5
BATCH_SIZE = 8192
LEARNING_RATE = 0.0005

# Training-window config:
#   GEN_WINDOW = N  → use the last N generations (e.g. 5)
#   GEN_WINDOW = 0  → use ALL available generations
GEN_WINDOW = 35

# Weight of the newest generation relative to older ones.
# 1.0 = equal weight; 2.0 = newest sampled twice as often, etc.
NEWEST_GEN_WEIGHT = 2.0

N_BUCKETS = 4


def get_bucket(ply: int) -> int:
    return min(ply * N_BUCKETS // 82, N_BUCKETS - 1)


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
            search_scores = torch.tensor(data[:, FEATURES : FEATURES + 1]).float()
            outcomes = torch.tensor(data[:, FEATURES + 1 : FEATURES + 2]).float()
            plies = data[:, FEATURES + 2].astype(np.int64)
            buckets = torch.tensor(
                np.clip(plies * N_BUCKETS // 82, 0, N_BUCKETS - 1), dtype=torch.long
            )
            y = (0.8 * search_scores + 0.2 * outcomes).squeeze(1)

            # ── BUG #1: validate label range ──────────────────────────
            y_min, y_max = y.min().item(), y.max().item()
            if y_min < 0.0 or y_max > 1.0:
                raise ValueError(
                    f"Labels out of [0,1] range in {path}: "
                    f"min={y_min:.4f} max={y_max:.4f}. "
                    "BCELoss will produce NaN. Check your data generator."
                )

            print(f"  loaded {n:,} samples")
            return X, y, buckets

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
        selected_set = set(all_gens[-window:])
        for g in all_gens:
            if g % 10 == 0:
                selected_set.add(g)
        selected = sorted(list(selected_set))
    else:
        selected = all_gens

    print(f"\nTraining window: generations {selected}")

    X_list = []
    y_list = []
    b_list = []
    for gen in selected:
        path = os.path.join(databin_dir, f"gen{gen}_data.bin")
        if not os.path.exists(path):
            print(f"  [warn] {path} missing, skipping")
            continue
        X, y, buckets = load_samples(path)
        repeats = int(newest_weight) if gen == selected[-1] else 1
        for _ in range(repeats):
            X_list.append(X)
            y_list.append(y)
            b_list.append(buckets)

    if not X_list:
        raise RuntimeError("No data loaded — all generation files were missing.")

    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    b_all = torch.cat(b_list, dim=0)

    print(f"\nTotal samples in training window: {len(X_all):,}\n")
    return X_all, y_all, b_all


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
        self.fc2 = nn.Linear(64, N_BUCKETS)
        self.screlu = SCReLU()

    def forward(self, x, buckets: torch.Tensor):
        # buckets: [batch] int64 tensor with values 0..N_BUCKETS-1
        l1 = self.screlu(self.fc0(x))
        l2 = self.screlu(self.fc1(l1))
        all_out = self.fc2(l2)  # [batch, N_BUCKETS]

        # Pick the right bucket output per sample
        out = all_out.gather(1, buckets.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(out)

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

    # ── Build windowed tensors and move to VRAM ────────────────────────
    X_all, y_all, b_all = build_windowed_dataset(
        current_gen=gen_count,
        window=GEN_WINDOW,
        newest_weight=NEWEST_GEN_WEIGHT,
    )

    print("Moving dataset to GPU VRAM...")
    t0 = time.time()
    X_all = X_all.to(device)
    y_all = y_all.to(device)
    b_all = b_all.to(device)
    print(f"Dataset moved in {time.time() - t0:.2f}s")

    n_samples = len(X_all)

    model = SinglePerspectiveNNUE(features=FEATURES, hl=128)
    if base_weights and os.path.exists(base_weights):
        model.load_weights(base_weights)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_fn = nn.BCELoss()

    print("Starting training …")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        batches = 0

        # Fast in-VRAM shuffle
        indices = torch.randperm(n_samples, device=device)
        X_shuffled = X_all[indices]
        y_shuffled = y_all[indices]
        b_shuffled = b_all[indices]

        for start_idx in range(0, n_samples, BATCH_SIZE):
            batch_X = X_shuffled[start_idx : start_idx + BATCH_SIZE]
            batch_y = y_shuffled[start_idx : start_idx + BATCH_SIZE]
            batch_b = b_shuffled[start_idx : start_idx + BATCH_SIZE]

            optimizer.zero_grad()  # ← correct position
            pred = model(batch_X, batch_b)
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
