import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = 199  # single perspective
LABEL = 1
ROW_SIZE = FEATURES + LABEL  # 200

EPOCHS = 15
BATCH_SIZE = 8192
LEARNING_RATE = 0.001


def load_samples(path):
    print(f"loading {path}...")
    start = time.time()
    with open(path, "rb") as f:
        raw = f.read()
    print(f"read {len(raw) / 1e6:.1f} MB in {time.time() - start:.2f}s")

    for skip in [0, 4, 8, 16]:
        remaining = len(raw) - skip
        if remaining % (ROW_SIZE * 4) == 0:
            print(f"aligned at skip={skip}")
            data = np.frombuffer(raw[skip:], dtype=np.float32)
            n = len(data) // ROW_SIZE
            data = data.reshape(n, ROW_SIZE)
            X = torch.tensor(data[:, :FEATURES])  # 199 features
            y = torch.tensor(data[:, FEATURES:])  # 1 label
            print(f"loaded {n} samples ({n * ROW_SIZE * 4 / 1e6:.1f} MB)")
            return X, y

    raise ValueError(f"could not align buffer, size={len(raw)}")


class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2


class SinglePerspectiveNNUE(nn.Module):
    def __init__(self, features=199, hl=128):
        super().__init__()
        self.features = features
        self.hl = hl

        # Accumulator: single perspective
        self.fc0 = nn.Linear(features, hl)

        # Layer 1: Takes SCReLU'd hidden layer
        self.fc1 = nn.Linear(hl, 64)

        # Output layer
        self.fc2 = nn.Linear(64, 1)

        self.screlu = SCReLU()

    def forward(self, x):
        acc = self.fc0(x)
        l1_in = self.screlu(acc)

        l1_out = self.fc1(l1_in)
        l1_out = self.screlu(l1_out)

        out = self.fc2(l1_out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    gen_count = sys.argv[1] if len(sys.argv) > 1 else "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    data_path = f"./databin/gen{gen_count}_data.bin"
    X, y = load_samples(data_path)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = SinglePerspectiveNNUE(features=FEATURES, hl=128).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Binary Cross Entropy is mathematically required for Sigmoid outputs!
    loss_fn = nn.BCELoss()

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        batches = 0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        elapsed = time.time() - start_time
        print(
            f"epoch {epoch + 1:3d}/{EPOCHS} | avg loss {avg_loss:.6f} | {elapsed:.1f}s elapsed"
        )

    print(f"training done in {time.time() - start_time:.1f}s")

    # export
    model.cpu()
    weights = [p.detach().numpy().flatten() for p in model.parameters()]
    all_weights = np.concatenate(weights).astype(np.float32)

    out_path = f"databin/gen{gen_count}_weights.bin"
    all_weights.tofile(out_path)
    print(f"saved {out_path} ({len(all_weights)} floats, {len(all_weights) * 4} bytes)")
