"""
inference_nonwestern.py

Run the frozen ResNet18 aesthetic model on the non-Western
evaluation sets (687 images across East Asia, South Asia, MENA).

No ground-truth ratings exist for these images, so we report
behavioural statistics only: predicted mean score, predicted
distribution std (per-image uncertainty), and the full predicted
score distribution.

Reads: provenance.csv (filter status == "kept")
Loads: checkpoints/resnet18_ava_best.pt
Writes: logs/nonwestern_predictions.csv
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
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.home() / "aesthetics"
CKPT_PATH = ROOT / "checkpoints" / "resnet18_ava_best.pt"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

CBD_ROOT = Path("/home/tempuser1/Desktop/AESTHETICS_RESEARCH/cultural-bias-data")
PROV_CSV = CBD_ROOT / "provenance.csv"

BATCH_SIZE = 64
NUM_WORKERS = 4

SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CulturalSet(Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True)
        self.root = root
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
        path = self.root / row["region"] / row["filename"]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, row["filename"], row["region"]


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

    print(f"reading {PROV_CSV}")
    prov = pd.read_csv(PROV_CSV)
    kept = prov[prov["status"] == "kept"].reset_index(drop=True)
    print(f"kept images: {len(kept)}")
    print(f"  per region: {kept['region'].value_counts().to_dict()}")

    ds = CulturalSet(kept, CBD_ROOT)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_filenames = []
    all_regions = []
    all_pred_means = []
    all_pred_stds = []
    all_pred_probs = []

    score_levels = SCORE_LEVELS.to(DEVICE)

    with torch.no_grad():
        for x, fnames, regions in tqdm(loader, desc="infer"):
            x = x.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
            probs_f = probs.float()
            pred_mean = (probs_f * score_levels).sum(dim=1)
            pred_var = (probs_f * (score_levels ** 2)).sum(dim=1) - pred_mean ** 2
            pred_std = torch.sqrt(torch.clamp(pred_var, min=0.0) + 1e-8)

            all_filenames.extend(list(fnames))
            all_regions.extend(list(regions))
            all_pred_means.extend(pred_mean.cpu().numpy().tolist())
            all_pred_stds.extend(pred_std.cpu().numpy().tolist())
            all_pred_probs.append(probs_f.cpu().numpy())

    pred_means = np.array(all_pred_means)
    pred_stds = np.array(all_pred_stds)
    pred_probs = np.concatenate(all_pred_probs, axis=0)

    out = pd.DataFrame({
        "filename": all_filenames,
        "region": all_regions,
        "pred_mean": pred_means,
        "pred_std": pred_stds,
    })
    for i in range(10):
        out[f"pred_vote_{i+1}"] = pred_probs[:, i]

    out_path = LOG_DIR / "nonwestern_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}  ({len(out)} rows)")

    print("\n=== per-region summary ===")
    for region in sorted(out["region"].unique()):
        sub = out[out["region"] == region]
        print(f"  {region:12s}  n={len(sub):4d}  "
              f"pred_mean: avg {sub['pred_mean'].mean():.3f}  std {sub['pred_mean'].std():.3f}  "
              f"min {sub['pred_mean'].min():.3f}  max {sub['pred_mean'].max():.3f}  "
              f"pred_dist_std (uncertainty): avg {sub['pred_std'].mean():.3f}")

    print("\n=== AVA test reference (from earlier run) ===")
    ava_path = LOG_DIR / "ava_test_predictions.csv"
    if ava_path.exists():
        ava = pd.read_csv(ava_path)
        print(f"  ava_test     n={len(ava):4d}  "
              f"pred_mean: avg {ava['pred_mean'].mean():.3f}  std {ava['pred_mean'].std():.3f}  "
              f"min {ava['pred_mean'].min():.3f}  max {ava['pred_mean'].max():.3f}  "
              f"pred_dist_std (uncertainty): avg {ava['pred_std'].mean():.3f}")
    else:
        print(f"  (ava_test_predictions.csv not found at {ava_path})")


if __name__ == "__main__":
    main()
