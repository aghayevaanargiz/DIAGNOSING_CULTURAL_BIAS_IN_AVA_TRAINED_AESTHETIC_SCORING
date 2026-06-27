"""
sanity_overfit.py

Day-1 sanity check. Trains a ResNet18 + 10-way softmax head with
EMD loss on a tiny fixed batch of 100 AVA images for 50 epochs.

PURPOSE: catch data-loader, label-shape, and loss-function bugs
before launching a long training run. If loss drops near zero,
the plumbing works. If it plateaus or NaNs, we have a bug.

Saves nothing. Throwaway.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR = Path.home() / "aesthetics" / "data" / "splits"

N_IMAGES = 100
N_EPOCHS = 50
LR = 1e-3
SEED = 42

VOTE_COLS = [f"vote_{i}" for i in range(1, 11)]


class AvaTinySet(Dataset):
    """Loads a small fixed subset of AVA into memory for overfitting."""

    def __init__(self, csv_path, n_images, seed):
        df = pd.read_csv(csv_path)
        df = df.sample(n=n_images, random_state=seed).reset_index(drop=True)
        self.df = df
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(row[VOTE_COLS].to_numpy(dtype=np.float32))
        return x, y


def emd_loss(pred, target):
    """Earth Mover's Distance loss for ordinal score distributions.
    Both pred and target are (batch, 10) and sum to 1 along dim 1.
    EMD = mean over batch of sqrt(mean of squared cumulative differences)."""
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    sq = (cdf_pred - cdf_target) ** 2
    return torch.sqrt(sq.mean(dim=1) + 1e-8).mean()


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 10)
    return m


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"device: {DEVICE}")
    print(f"loading {N_IMAGES} images from train.csv ...")
    ds = AvaTinySet(SPLITS_DIR / "train.csv", N_IMAGES, SEED)
    # Load everything in one batch so the "overfit" is unambiguous.
    loader = DataLoader(ds, batch_size=N_IMAGES, shuffle=False, num_workers=2)

    print("building model ...")
    model = build_model().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"overfitting {N_IMAGES} images for {N_EPOCHS} epochs ...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = F.softmax(logits, dim=1)
            loss = emd_loss(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(f"  epoch {epoch:3d}   loss = {loss.item():.5f}")

    final = loss.item()
    print(f"\nfinal loss: {final:.5f}")
    if final < 0.02:
        print("PASS: model successfully overfit the small batch. Plumbing works.")
    elif final < 0.08:
        print("MARGINAL: loss dropped but not all the way. Probably fine, worth a look.")
    else:
        print("FAIL: loss did not drop. There is a bug in the data loader, labels, or loss.")


if __name__ == "__main__":
    main()
