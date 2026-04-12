import sys
import time

import numpy as np
import torch
import torch.nn as nn

FEATURES = 398  # dual perspective: 199 * 2
LABEL = 1
ROW_SIZE = FEATURES + LABEL  # 399


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
            X = torch.tensor(data[:, :FEATURES])  # 398 features
            y = torch.tensor(data[:, FEATURES:])  # 1 label
            print(f"loaded {n} samples ({n * ROW_SIZE * 4 / 1e6:.1f} MB)")
            return X, y

    raise ValueError(f"could not align buffer, size={len(raw)}")


# load data
X, y = load_samples("./databin/gen0_data.bin")
print(f"X shape: {X.shape}, y shape: {y.shape}")


class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2


class DualPerspectiveNNUE(nn.Module):
    def __init__(self, features=199, hl=128):
        super().__init__()
        self.features = features
        self.hl = hl

        # Accumulator: shared weights for both perspectives
        self.fc0 = nn.Linear(features, hl)

        # Layer 1: Takes concatenated STM and NSTM hidden layers
        self.fc1 = nn.Linear(hl * 2, 64)

        # Output layer
        self.fc2 = nn.Linear(64, 1)

        self.screlu = SCReLU()

    def forward(self, x):
        stm = x[:, : self.features]
        nstm = x[:, self.features :]

        acc_stm = self.fc0(stm)
        acc_nstm = self.fc0(nstm)

        l1_in = torch.cat([self.screlu(acc_stm), self.screlu(acc_nstm)], dim=1)

        l1_out = self.fc1(l1_in)
        l1_out = self.screlu(l1_out)

        out = self.fc2(l1_out)
        return torch.sigmoid(out)


model = DualPerspectiveNNUE(features=199, hl=128)

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
all_weights.tofile(f"databin/gen{sys.argv[1]}_weights.bin")
print(
    f"saved gen{sys.argv[1]}_weights.bin ({len(all_weights)} floats, {len(all_weights) * 4} bytes)"
)
