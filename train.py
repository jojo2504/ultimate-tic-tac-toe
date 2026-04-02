import time

import numpy as np
import torch
import torch.nn as nn


def load_samples(path):
    print(f"loading {path}...")
    start = time.time()
    with open(path, "rb") as f:
        raw = f.read()
    print(f"read {len(raw) / 1e6:.1f} MB in {time.time() - start:.2f}s")

    for skip in [0, 4, 8, 16]:
        remaining = len(raw) - skip
        if remaining % (201 * 4) == 0:
            print(f"aligned at skip={skip}")
            data = np.frombuffer(raw[skip:], dtype=np.float32)
            n = len(data) // 201
            data = data.reshape(n, 201)
            X = torch.tensor(data[:, :200])
            y = torch.tensor(data[:, 200:])
            print(f"loaded {n} samples ({n * 201 * 4 / 1e6:.1f} MB)")
            return X, y

    raise ValueError(f"could not align buffer, size={len(raw)}")


# load data
X, y = load_samples("./databin/gen0_data.bin")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# tiny network
model = nn.Sequential(
    nn.Linear(200, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)
total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params}")

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

print("training...")
start = time.time()
for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        elapsed = time.time() - start
        print(f"epoch {epoch:3d}/100 | loss {loss.item():.6f} | {elapsed:.1f}s elapsed")

print(f"training done in {time.time() - start:.1f}s")

# export
weights = [p.detach().numpy().flatten() for p in model.parameters()]
all_weights = np.concatenate(weights).astype(np.float32)
all_weights.tofile("databin/gen1_weights.bin")
print(
    f"saved gen1_weights.bin ({len(all_weights)} floats, {len(all_weights) * 4} bytes)"
)
