"""
eval_ava_test.py

Final evaluation of the frozen ResNet18 aesthetic model on the
held-out AVA test set. This is the headline instrument-validity
number for the paper.

Loads: ~/aesthetics/checkpoints/resnet18_ava_best.pt
Reads: ~/aesthetics/data/splits/test.csv  (2321 images, never seen)
Writes: ~/aesthetics/logs/ava_test_predictions.csv  (per-image preds)

Reports: test EMD loss, Spearman, Pearson, MSE on mean scores.

The model is in eval mode and weights are NOT modified.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.home() / "aesthetics"
SPLITS_DIR = ROOT / "data" / "splits"
CKPT_PATH = ROOT / "checkpoints" / "resnet18_ava_best.pt"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
NUM_WORKERS = 4

VOTE_COLS = [f"vote_{i}" for i in range(1, 11)]
SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AvaEvalSet(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(row[VOTE_COLS].to_numpy(dtype=np.float32))
        return x, y, int(row["image_num"]), float(row["mean_score"])


def emd_loss(pred, target):
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    sq = (cdf_pred - cdf_target) ** 2
    return torch.sqrt(sq.mean(dim=1) + 1e-8).mean()


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def main():
    print(f"device: {DEVICE}")
    print(f"loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    print(f"  saved at epoch {ckpt['epoch']}  val_loss {ckpt['val_loss']:.4f}  "
          f"val_spearman {ckpt['val_spearman']:.4f}")

    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = AvaEvalSet(SPLITS_DIR / "test.csv")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    print(f"test set: {len(ds)} images")

    total_loss = 0.0
    n = 0
    all_image_nums = []
    all_true_means = []
    all_pred_means = []
    all_pred_stds = []
    all_pred_probs = []

    score_levels = SCORE_LEVELS.to(DEVICE)

    with torch.no_grad():
        for x, y, image_nums, true_means in tqdm(loader, desc="eval"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                loss = emd_loss(probs, y)
            probs_f = probs.float()
            pred_mean = (probs_f * score_levels).sum(dim=1)
            pred_var = (probs_f * (score_levels ** 2)).sum(dim=1) - pred_mean ** 2
            pred_std = torch.sqrt(torch.clamp(pred_var, min=0.0) + 1e-8)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_image_nums.extend(image_nums.tolist())
            all_true_means.extend(true_means.tolist())
            all_pred_means.extend(pred_mean.cpu().numpy().tolist())
            all_pred_stds.extend(pred_std.cpu().numpy().tolist())
            all_pred_probs.append(probs_f.cpu().numpy())

    test_loss = total_loss / n
    pred_means = np.array(all_pred_means)
    true_means = np.array(all_true_means)
    pred_stds = np.array(all_pred_stds)
    pred_probs = np.concatenate(all_pred_probs, axis=0)

    spearman = spearmanr(pred_means, true_means).correlation
    pearson = pearsonr(pred_means, true_means)[0]
    mse = float(np.mean((pred_means - true_means) ** 2))
    mae = float(np.mean(np.abs(pred_means - true_means)))

    print(f"\n=== AVA test set results ({n} images) ===")
    print(f"  EMD loss:   {test_loss:.4f}")
    print(f"  Spearman:   {spearman:.4f}")
    print(f"  Pearson:    {pearson:.4f}")
    print(f"  MSE (mean): {mse:.4f}")
    print(f"  MAE (mean): {mae:.4f}")
    print(f"  predicted mean: avg {pred_means.mean():.3f}  std {pred_means.std():.3f}")
    print(f"  ground-truth mean: avg {true_means.mean():.3f}  std {true_means.std():.3f}")
    print(f"  predicted distribution std (per image): avg {pred_stds.mean():.3f}  std {pred_stds.std():.3f}")

    out_df = pd.DataFrame({
        "image_num": all_image_nums,
        "true_mean": true_means,
        "pred_mean": pred_means,
        "pred_std": pred_stds,
    })
    for i in range(10):
        out_df[f"pred_vote_{i+1}"] = pred_probs[:, i]
    out_path = LOG_DIR / "ava_test_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\npredictions saved: {out_path}")


if __name__ == "__main__":
    main()
