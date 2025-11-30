#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

SAVE_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\"
DATA_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\data\\train" # 我的路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TimeSeriesDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        for fn in os.listdir(DATA_DIR):
            if not fn.endswith(".csv"):
                continue
            try:
                label = int(fn.split("-")[-1].split(".")[0])
                arr = np.loadtxt(os.path.join(DATA_DIR, fn))
                self.data.append(arr.astype(np.float32))
                self.labels.append(label)
            except Exception as e:
                print(f"[Error] Failed: {fn} => {e}")
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).unsqueeze(0)   # (1, 8192)
        y = torch.tensor(self.labels[idx]).float()
        return x, y

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * 512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    ds = TimeSeriesDataset()
    train_len = int(len(ds) * 0.8)
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8)

    model = CNN1D().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []

    print("Training 1D CNN...")
    for epoch in range(15):
        model.train()
        total = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        train_losses.append(total / len(train_dl))

        # 验证
        model.eval()
        vloss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze()
                vloss += loss_fn(pred, y).item()
        val_losses.append(vloss / len(val_dl))

        print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

    torch.save(model.state_dict(), SAVE_DIR + "model2_1dcnn.pt")
    print("Saved model2_1dcnn.pt")

    # 画loss曲线
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Model 2 - Training Curve (1D CNN)")
    plt.savefig(SAVE_DIR + "train2_curve.png")
    print("Saved train2_curve.png")

if __name__ == "__main__":
    main()
