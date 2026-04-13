import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
FEATURES = 398
LABEL = 1
ROW_SIZE = FEATURES + LABEL
BATCH_SIZE = 16384  # Large batches are better for GPU utilization
EPOCHS = 100
LEARNING_RATE = 0.001

# Detect Device: AMD ROCm (Linux) or MPS (Mac) or DirectML/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # ROCm maps to "cuda" in PyTorch
    print(f"Using AMD GPU (via ROCm)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (AMD GPU not detected. Install ROCm PyTorch!)")


# --- Faster Data Loading with Memory Mapping ---
def load_data_fast(path):
    print(f"Mapping {path}...")
    # np.memmap avoids loading the whole file into RAM at once
    data = np.memmap(path, dtype=np.float32, mode="r")

    # Try to find alignment
    for skip in [0, 4, 8, 16]:
        n_elements = len(data) - skip
        if n_elements % ROW_SIZE == 0:
            n_samples = n_elements // ROW_SIZE
            # Reshape without copying memory
            data = data[skip:].reshape(n_samples, ROW_SIZE)
            X = torch.from_numpy(data[:, :FEATURES])
            y = torch.from_numpy(data[:, FEATURES:])
            print(f"Successfully aligned: {n_samples} samples found.")
            return X, y
    raise ValueError("Could not align binary data.")


# --- Architecture ---
class SCReLU(nn.Module):
    def forward(self, x):
        # Optimized SCReLU: clamp and square
        return torch.clamp(x, 0.0, 1.0).pow(2)


class DualPerspectiveNNUE(nn.Module):
    def __init__(self, features=199, hl=128):
        super().__init__()
        self.features = features
        # Shared accumulator for both perspectives
        self.fc0 = nn.Linear(features, hl)
        self.fc1 = nn.Linear(hl * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.screlu = SCReLU()

    def forward(self, x):
        # Split features into Side-To-Move and Non-Side-To-Move
        stm = x[:, : self.features]
        nstm = x[:, self.features :]

        # Perspective Pooling
        acc_stm = self.screlu(self.fc0(stm))
        acc_nstm = self.screlu(self.fc0(nstm))

        # Concatenate and pass through hidden layer
        l1_in = torch.cat([acc_stm, acc_nstm], dim=1)
        l1_out = self.screlu(self.fc1(l1_in))

        return torch.sigmoid(self.fc2(l1_out))


# --- Main Execution ---
if __name__ == "__main__":
    X_raw, y_raw = load_data_fast(f"./databin/gen{sys.argv[1]}_data.bin")

    dataset = TensorDataset(X_raw, y_raw)
    # pin_memory=True speeds up CPU -> GPU transfer
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = DualPerspectiveNNUE().to(device)

    # Optional: PyTorch 2.0+ Compilation (Linux only, adds 1-2 min startup but 30% faster training)
    if sys.platform == "linux" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled for optimized kernels.")
        except:
            print("Compilation failed, falling back to standard mode.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Use Automatic Mixed Precision (AMP) for a massive speed boost on RDNA/CDNA cards
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = (
                batch_X.to(device, non_blocking=True),
                batch_y.to(device, non_blocking=True),
            )

            # Mixed precision context
            with torch.cuda.amp.autocast():
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Time: {time.time() - start_time:.1f}s"
            )

    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Export
    model.to("cpu")  # Move back to export weights
    weights = [p.detach().numpy().flatten() for p in model.parameters()]
    all_weights = np.concatenate(weights).astype(np.float32)
    save_path = f"databin/gen{sys.argv[1] if len(sys.argv) > 1 else 0}_weights.bin"
    all_weights.tofile(save_path)
    print(f"Saved weights to {save_path}")
