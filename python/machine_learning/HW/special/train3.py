#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

SAVE_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\"
DATA_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\data\\train"  # 修改为你的路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_spectrogram(x):
    # nperseg=256 与之前保持一致 -> freq bins = 129
    f, t, Z = stft(x, nperseg=256)
    S = np.abs(Z)
    S = S.astype(np.float32)
    return S  # shape: (freq, time)

class SpectrogramDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        for fn in sorted(os.listdir(DATA_DIR)):
            if not fn.endswith(".csv"):
                continue
            try:
                label = int(fn.split("-")[-1].split(".")[0])
                arr = np.loadtxt(os.path.join(DATA_DIR, fn))
                spec = compute_spectrogram(arr)  # (F, T)
                self.data.append(spec)
                self.labels.append(label)
            except Exception as e:
                print(f"[Error] Failed: {fn} => {e}")

        if len(self.data) == 0:
            raise RuntimeError("No training files found in DATA_DIR")

        self.data = np.array(self.data)          # (N, F, T)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert to tensor with channel dim: (1, F, T)
        x = torch.tensor(self.data[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx]).float()
        return x, y

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        # feature extractor: keep spatial dims but then adaptive pool to fixed (4,4)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # halves each spatial dim (floor)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # halves again
            nn.AdaptiveAvgPool2d((4, 4)) # force output to (4,4)
        )
        # after AdaptiveAvgPool2d output shape = (batch, 32, 4, 4)
        flat_dim = 32 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)   # (B, 32, 4, 4)
        x = self.classifier(x) # (B, 1)
        return x

def main():
    ds = SpectrogramDataset()
    train_len = int(len(ds) * 0.8)
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4)

    model = CNN2D().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []

    print("Training 2D CNN on Spectrogram... (device =", DEVICE, ")")
    # quick debug: print one sample shape
    sample_x, _ = ds[0]
    print("Example spectrogram shape (1,F,T):", sample_x.shape)

    for epoch in range(15):
        model.train()
        total = 0.0
        iters = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x).squeeze()          # shape: (B,)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
            iters += 1
        train_losses.append(total / max(1, iters))

        model.eval()
        vtotal = 0.0
        viters = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze()
                vtotal += loss_fn(pred, y).item()
                viters += 1
        val_losses.append(vtotal / max(1, viters))

        print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.6f}, val_loss={val_losses[-1]:.6f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model3_spec2d.pt"))
    print("Saved model3_spec2d.pt")

    # loss图
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Model 3 - Training Curve (Spectrogram 2D CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "train3_curve.png"))
    print("Saved train3_curve.png")

if __name__ == "__main__":
    main()
