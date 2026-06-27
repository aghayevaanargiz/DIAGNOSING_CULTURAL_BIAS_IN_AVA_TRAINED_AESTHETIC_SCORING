"""
train_ava.py

Fine-tune a ResNet18 aesthetic regressor on the AVA subset.

- Backbone: ImageNet-pretrained ResNet18
- Head: final fc replaced with 10-unit linear -> softmax (predicts
  full score distribution 1..10, not a single number)
- Loss: Earth Mover's Distance (NIMA-style)
- Optimizer: Adam with discriminative LR (small for backbone, larger for head)
- Mixed precision (fp16) for A5000 throughput
- Validation each epoch; best-val-loss checkpoint saved
- Test set is NEVER touched here (separate script later)
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.home() / "aesthetics"
SPLITS_DIR = ROOT / "data" / "splits"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Config
N_EPOCHS = 10
BATCH_SIZE = 128
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
SEED = 42

VOTE_COLS = [f"vote_{i}" for i in range(1, 11)]
SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AvaDataset(Dataset):
    def __init__(self, csv_path, train=False):
        self.df = pd.read_csv(csv_path)
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
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
        mean_score = float(row["mean_score"])
        return x, y, mean_score


def emd_loss(pred, target):
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    sq = (cdf_pred - cdf_target) ** 2
    return torch.sqrt(sq.mean(dim=1) + 1e-8).mean()


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 10)
    return m


def predicted_mean_scores(probs):
    """probs: (batch, 10) -> mean of score distribution per image."""
    return (probs * SCORE_LEVELS.to(probs.device)).sum(dim=1)


def run_validation(model, loader):
    model.eval()
    total_loss = 0.0
    n = 0
    pred_means_all = []
    true_means_all = []
    with torch.no_grad():
        for x, y, m in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                loss = emd_loss(probs, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            pred_means_all.append(predicted_mean_scores(probs.float()).cpu().numpy())
            true_means_all.append(m.numpy())
    avg_loss = total_loss / n
    pred_means_all = np.concatenate(pred_means_all)
    true_means_all = np.concatenate(true_means_all)
    spearman = spearmanr(pred_means_all, true_means_all).correlation
    pearson = pearsonr(pred_means_all, true_means_all)[0]
    return avg_loss, spearman, pearson


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    print(f"device: {DEVICE}")
    print(f"epochs: {N_EPOCHS}  batch: {BATCH_SIZE}  lr_backbone: {LR_BACKBONE}  lr_head: {LR_HEAD}")

    train_ds = AvaDataset(SPLITS_DIR / "train.csv", train=True)
    val_ds = AvaDataset(SPLITS_DIR / "val.csv", train=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model().to(DEVICE)

    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optim = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    log_rows = []
    ckpt_best = CKPT_DIR / "resnet18_ava_best.pt"

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:2d}/{N_EPOCHS}")
        for x, y, _ in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                loss = emd_loss(probs, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            running_n += x.size(0)
            pbar.set_postfix(loss=f"{running_loss/running_n:.4f}")

        train_loss = running_loss / running_n
        val_loss, val_spearman, val_pearson = run_validation(model, val_loader)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_spearman": val_spearman,
                "val_pearson": val_pearson,
            }, ckpt_best)
            tag = "  [saved best]"
        else:
            tag = ""

        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"spearman {val_spearman:.4f}  pearson {val_pearson:.4f}  "
              f"({dt:.0f}s){tag}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_spearman": val_spearman,
            "val_pearson": val_pearson,
            "seconds": dt,
        })
        pd.DataFrame(log_rows).to_csv(LOG_DIR / "train_log.csv", index=False)

    print(f"\nbest val loss: {best_val_loss:.4f}")
    print(f"best checkpoint: {ckpt_best}")
    print(f"log: {LOG_DIR / 'train_log.csv'}")


if __name__ == "__main__":
    main()
