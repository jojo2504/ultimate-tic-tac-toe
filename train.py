import argparse
import glob
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
FEATURES = 217
POLICY = 81
SCORE = 1
LABEL = 1
PLY = 1
ROW_SIZE = FEATURES + POLICY + SCORE + LABEL + PLY  # 301

EPOCHS = 15  # raised ceiling — early stopping will cut this short when needed
BATCH_SIZE = 8192
LEARNING_RATE = 0.0005
SCORE_RATIO = 0.8
OUTCOME_RATIO = 0.2

# Minimum samples-to-parameters ratio. If we fall below this we warn loudly.
# Source: community best-practice — 10× params is the lower bound before loss plateaus.
MIN_SAMPLES_PER_PARAM = 10

# Early stopping: halt if val loss does not improve for this many epochs.
EARLY_STOP_PATIENCE = 3

# LR scheduler: reduce LR by this factor when val loss plateaus for patience//2 epochs.
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5
LR_MIN = 1e-6

# Training-window config:
#   GEN_WINDOW = N  → use the last N generations (e.g. 35)
#   GEN_WINDOW = 0  → use ALL available generations
GEN_WINDOW = 35

# Weight of the newest generation relative to older ones.
NEWEST_GEN_WEIGHT = 2.0

# Fraction of samples held out as the validation set.
VAL_SPLIT = 0.10

N_BUCKETS = 4

# Loss weight for policy. Can be tuned.
POLICY_LOSS_WEIGHT = 1.0


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
            policies = torch.tensor(data[:, FEATURES : FEATURES + POLICY]).float()
            search_scores = torch.tensor(
                data[:, FEATURES + POLICY : FEATURES + POLICY + 1]
            ).float()
            outcomes = torch.tensor(
                data[:, FEATURES + POLICY + 1 : FEATURES + POLICY + 2]
            ).float()
            plies = data[:, FEATURES + POLICY + 2].astype(np.int64)
            buckets = torch.tensor(
                np.clip(plies * N_BUCKETS // 82, 0, N_BUCKETS - 1), dtype=torch.long
            )
            y = (SCORE_RATIO * search_scores + OUTCOME_RATIO * outcomes).squeeze(1)

            y_min, y_max = y.min().item(), y.max().item()
            if y_min < 0.0 or y_max > 1.0:
                raise ValueError(
                    f"Labels out of [0,1] range in {path}: "
                    f"min={y_min:.4f} max={y_max:.4f}. "
                    "BCELoss will produce NaN. Check your data generator."
                )

            print(f"  loaded {n:,} samples")
            return X, policies, y, buckets

    raise ValueError(f"Could not align buffer, size={len(raw)}")


def discover_generations(databin_dir: str = "./databin") -> list[int]:
    pattern = os.path.join(databin_dir, "gen*_data.bin")
    paths = glob.glob(pattern)
    gens = []
    for p in paths:
        base = os.path.basename(p)
        num = base.replace("gen", "").replace("_data.bin", "")
        try:
            gens.append(int(num))
        except ValueError:
            pass
    return sorted(gens)


def build_windowed_dataset(
    current_gen: int,
    lineage: Optional[list[int]] = None,
    window: int = GEN_WINDOW,
    newest_weight: float = NEWEST_GEN_WEIGHT,
    databin_dir: str = "./databin",
):
    all_gens = discover_generations(databin_dir)
    all_gens = [g for g in all_gens if g <= current_gen]

    if lineage is not None:
        lineage_set = set(lineage)
        filtered = [g for g in all_gens if g in lineage_set or g == 0]
        if not filtered:
            print("[warn] lineage filter removed all gens, falling back to full window")
            filtered = all_gens
        all_gens = sorted(set(filtered))
        print(f"Lineage filter applied — eligible generations: {all_gens}")

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
        selected = sorted(selected_set)
    else:
        selected = all_gens

    print(f"\nTraining window: generations {selected}")

    X_list, p_list, y_list, b_list = [], [], [], []
    for gen in selected:
        path = os.path.join(databin_dir, f"gen{gen}_data.bin")
        if not os.path.exists(path):
            print(f"  [warn] {path} missing, skipping")
            continue
        X, p, y, buckets = load_samples(path)
        repeats = int(newest_weight) if gen == selected[-1] else 1
        for _ in range(repeats):
            X_list.append(X)
            p_list.append(p)
            y_list.append(y)
            b_list.append(buckets)

    if not X_list:
        raise RuntimeError("No data loaded — all generation files were missing.")

    X_all = torch.cat(X_list, dim=0)
    p_all = torch.cat(p_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    b_all = torch.cat(b_list, dim=0)

    print(f"\nTotal samples in training window: {len(X_all):,}\n")
    return X_all, p_all, y_all, b_all


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2


class SinglePerspectiveNNUE(nn.Module):
    def __init__(self, features: int = FEATURES, hl: int = 128):
        super().__init__()
        self.fc0 = nn.Linear(features, hl)
        self.fc1 = nn.Linear(hl, 64)
        self.fc2 = nn.Linear(64, N_BUCKETS)
        self.policy_head = nn.Linear(64, 81)
        self.screlu = SCReLU()

    def forward(self, x, buckets: torch.Tensor):
        l1 = self.screlu(self.fc0(x))
        l2 = self.screlu(self.fc1(l1))

        # Value out
        all_out = self.fc2(l2)  # [batch, N_BUCKETS]
        out = all_out.gather(1, buckets.unsqueeze(1)).squeeze(1)

        # Policy out (raw logits)
        pol_out = self.policy_head(l2)

        return torch.sigmoid(out), pol_out

    def load_weights(self, path: str):
        print(f"Loading base weights from {path} ...")
        with open(path, "rb") as f:
            raw = f.read()
        all_weights = np.frombuffer(raw, dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                if offset + numel > len(all_weights):
                    print(
                        "  Legacy weights detected, skipping policy head init (leaving as random/zero)."
                    )
                    p.zero_()
                    continue
                w = all_weights[offset : offset + numel].copy()
                p.copy_(torch.from_numpy(w).view_as(p))
                offset += numel
        print(f"  Loaded {offset:,} floats.")

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# Train / val split helpers
# ─────────────────────────────────────────────
def split_dataset(X, p, y, b, val_fraction: float = VAL_SPLIT):
    n = len(X)
    perm = torch.randperm(n)
    X, p, y, b = X[perm], p[perm], y[perm], b[perm]

    n_val = max(1, int(n * val_fraction))
    n_tr = n - n_val

    return (
        X[:n_tr],
        p[:n_tr],
        y[:n_tr],
        b[:n_tr],  # train
        X[n_tr:],
        p[n_tr:],
        y[n_tr:],
        b[n_tr:],  # val
    )


def run_epoch(model, X, p, y, b, batch_size, device, optimizer=None, training=True):
    model.train(training)
    total_loss = 0.0
    n = len(X)
    indices = (
        torch.randperm(n, device=device) if training else torch.arange(n, device=device)
    )
    X_s, p_s, y_s, b_s = X[indices], p[indices], y[indices], b[indices]

    val_loss_fn = nn.BCELoss()

    with torch.set_grad_enabled(training):
        for start in range(0, n, batch_size):
            bX = X_s[start : start + batch_size]
            bp = p_s[start : start + batch_size]
            by = y_s[start : start + batch_size]
            bb = b_s[start : start + batch_size]

            if training:
                optimizer.zero_grad()

            val_pred, pol_pred = model(bX, bb)

            # Value loss
            v_loss = val_loss_fn(val_pred, by)

            # Policy loss
            # Only calculate policy loss for samples that have policy targets (i.e. not bootstrap)
            # The bootstrap samples have policy array set to 0.
            p_mask = (bp.sum(dim=1) > 0.0).float()

            # CrossEntropyLoss expects logits and probabilities
            # reduction='none' allows us to apply the mask
            p_loss_raw = F.cross_entropy(pol_pred, bp, reduction="none")

            mask_sum = p_mask.sum()
            if mask_sum > 0:
                p_loss = (p_loss_raw * p_mask).sum() / mask_sum
            else:
                p_loss = 0.0

            loss = v_loss + POLICY_LOSS_WEIGHT * p_loss

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(1, (n + batch_size - 1) // batch_size)


# ─────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────
def train(
    gen_count: int,
    base_weights: Optional[str] = None,
    depth: int = 3,
    lineage: Optional[list[int]] = None,
):
    global BATCH_SIZE, LEARNING_RATE, SCORE_RATIO, OUTCOME_RATIO, GEN_WINDOW, EPOCHS

    try:
        with open("config/config.json", "r") as f:
            config = json.load(f)
            depth_str = str(depth)
            if "depth" in config and depth_str in config["depth"]:
                d_conf = config["depth"][depth_str]
                BATCH_SIZE = d_conf.get("batch_size", BATCH_SIZE)
                LEARNING_RATE = d_conf.get("learning_rate", LEARNING_RATE)
                SCORE_RATIO = d_conf.get("score_ratio", 80) / 100.0
                OUTCOME_RATIO = d_conf.get("outcome_ratio", 20) / 100.0
                GEN_WINDOW = d_conf.get("window_length", GEN_WINDOW)
                EPOCHS = d_conf.get("epochs", EPOCHS)
                print(f"Loaded config for depth {depth}")
    except Exception as e:
        print(f"Could not load config: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_all, p_all, y_all, b_all = build_windowed_dataset(
        current_gen=gen_count,
        lineage=lineage,
        window=GEN_WINDOW,
        newest_weight=NEWEST_GEN_WEIGHT,
    )

    model = SinglePerspectiveNNUE(features=FEATURES, hl=128)
    total_params = model.count_params()
    n_samples = len(X_all)
    ratio = n_samples / total_params

    print(f"\nModel parameters : {total_params:,}")
    print(f"Training samples : {n_samples:,}")
    print(
        f"Samples / params : {ratio:.1f}× (minimum recommended: {MIN_SAMPLES_PER_PARAM}×)"
    )

    if ratio < MIN_SAMPLES_PER_PARAM:
        print(
            f"\n[WARN] Only {ratio:.1f}× samples-per-param "
            f"(need ≥{MIN_SAMPLES_PER_PARAM}×). "
            "Loss will likely plateau early. "
            "Consider increasing games_per_generation.\n"
        )
    else:
        print(f"Dataset size OK ✓\n")

    X_tr, p_tr, y_tr, b_tr, X_val, p_val, y_val, b_val = split_dataset(
        X_all, p_all, y_all, b_all
    )

    print(f"Train samples : {len(X_tr):,}  |  Val samples : {len(X_val):,}")
    print("Moving dataset to device …")
    t0 = time.time()
    X_tr, p_tr, y_tr, b_tr = (
        X_tr.to(device),
        p_tr.to(device),
        y_tr.to(device),
        b_tr.to(device),
    )
    X_val, p_val, y_val, b_val = (
        X_val.to(device),
        p_val.to(device),
        y_val.to(device),
        b_val.to(device),
    )
    print(f"Done in {time.time() - t0:.2f}s")

    if base_weights and os.path.exists(base_weights):
        model.load_weights(base_weights)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_MIN,
    )

    print(
        f"\nStarting training (max {EPOCHS} epochs, early-stop patience={EARLY_STOP_PATIENCE}) …\n"
    )
    start_time = time.time()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(EPOCHS):
        t_ep = time.time()

        tr_loss = run_epoch(
            model,
            X_tr,
            p_tr,
            y_tr,
            b_tr,
            BATCH_SIZE,
            device,
            optimizer=optimizer,
            training=True,
        )
        val_loss = run_epoch(
            model,
            X_val,
            p_val,
            y_val,
            b_val,
            BATCH_SIZE,
            device,
            optimizer=optimizer,
            training=False,
        )

        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss - 1e-6
        marker = " ✓" if improved else ""
        print(
            f"epoch {epoch + 1:3d}/{EPOCHS} | "
            f"train {tr_loss:.6f} | val {val_loss:.6f}{marker} | "
            f"lr {lr_now:.2e} | {time.time() - t_ep:.1f}s / {elapsed:.0f}s total"
        )

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping: val loss did not improve for "
                    f"{EARLY_STOP_PATIENCE} epochs. "
                    f"Best val loss = {best_val_loss:.6f}"
                )
                break

    total_time = time.time() - start_time
    print(
        f"\nTraining finished in {total_time:.1f}s | best val loss = {best_val_loss:.6f}"
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Restored best checkpoint weights.")

    model.cpu()
    weights = [p.detach().numpy().flatten() for p in model.parameters()]
    all_weights = np.concatenate(weights).astype(np.float32)

    out_path = f"databin/gen{gen_count}_weights.bin"
    os.makedirs("databin", exist_ok=True)
    all_weights.tofile(out_path)
    print(
        f"Saved {out_path} "
        f"({len(all_weights):,} floats, {len(all_weights) * 4:,} bytes)"
    )

    stats = {
        "gen": gen_count,
        "depth": depth,
        "best_val_loss": float(best_val_loss),
        "total_samples": n_samples,
        "params": total_params,
        "samples_per_param": float(ratio),
        "epochs_run": epoch + 1,
        "training_seconds": float(total_time),
    }
    stats_path = f"databin/gen{gen_count}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_count", type=int, nargs="?", default=0)
    parser.add_argument("--base-weights", type=str, default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--lineage",
        type=str,
        default=None,
        help="Comma-separated list of promoted generation numbers "
        "(e.g. '0,3,7,12'). Only these gens will be used for training. "
        "Rejected gens are excluded even if their data file exists.",
    )
    args = parser.parse_args()

    lineage = None
    if args.lineage:
        try:
            lineage = [int(x.strip()) for x in args.lineage.split(",") if x.strip()]
        except ValueError:
            print(f"[warn] Could not parse --lineage='{args.lineage}', ignoring.")

    train(
        args.gen_count,
        base_weights=args.base_weights,
        depth=args.depth,
        lineage=lineage,
    )
