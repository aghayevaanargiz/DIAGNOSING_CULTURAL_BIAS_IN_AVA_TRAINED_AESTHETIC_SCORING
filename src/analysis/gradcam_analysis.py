"""
gradcam_analysis.py

Third diagnostic axis: spatial attention via Grad-CAM.

For every image in AVA-test (2321) and non-Western (687):
  - Compute Grad-CAM on the last conv block of ResNet18 (layer4)
  - Compute the border-to-centre attention ratio
    (defined below: outer-ring area vs inner-square area)
  - Save the ratio per image to a CSV

Then:
  - Compute group statistics + KS / Wasserstein vs AVA-test
  - Select 8 images by 25th/75th percentile of border ratio
    within each of {ava_test, east_asia, south_asia, mena}
  - Render Grad-CAM overlays for those 8 and save as a figure

Outputs:
  logs/gradcam_metrics.csv     -- per-image border ratios
  logs/gradcam_summary.csv     -- per-group stats + KS/W
  figures/gradcam_examples.png -- 8-panel figure
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.home() / "aesthetics"
CKPT_PATH = ROOT / "checkpoints" / "resnet18_ava_best.pt"
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
LOG_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SPLITS_DIR = ROOT / "data" / "splits"
CBD_ROOT = Path("/home/tempuser1/Desktop/AESTHETICS_RESEARCH/cultural-bias-data")
PROV_CSV = CBD_ROOT / "provenance.csv"

# We compute Grad-CAM one-at-a-time to keep memory simple and predictable.
# It's still very fast on the A5000.
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32).to(DEVICE)

# Border-centre split: inner square covers the centred area = INNER_FRAC of total area.
# INNER_FRAC = 0.5 means the inner box has side length sqrt(0.5) * 224 ~= 158px centred,
# so half the pixel area is "centre" and half is "border".
INNER_FRAC = 0.5


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])


def make_rgb_loader():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
    ])


def make_centre_mask(h, w, inner_frac=INNER_FRAC):
    """Returns (centre_mask, border_mask) of shape (h, w), each in {0, 1}."""
    side = int(round(np.sqrt(inner_frac) * min(h, w)))
    cy, cx = h // 2, w // 2
    y0 = cy - side // 2
    x0 = cx - side // 2
    centre = np.zeros((h, w), dtype=np.float32)
    centre[y0:y0 + side, x0:x0 + side] = 1.0
    border = 1.0 - centre
    return centre, border


def border_centre_ratio(cam, centre_mask, border_mask):
    """cam: (H, W) normalized in [0, 1]. Returns total_border_mass / total_centre_mass.
    Normalized by area so the comparison is per-pixel intensity, not raw mass."""
    centre_mass = (cam * centre_mask).sum() / max(centre_mask.sum(), 1)
    border_mass = (cam * border_mask).sum() / max(border_mask.sum(), 1)
    return float(border_mass / (centre_mass + 1e-8))


class MeanScoreTarget:
    """For Grad-CAM: we want gradients w.r.t. the predicted mean score,
    which is a weighted combination of the 10 softmax outputs."""
    def __init__(self, levels):
        self.levels = levels  # tensor on DEVICE, shape (10,)

    def __call__(self, model_output):
        # model_output is the raw logits, shape (10,) for one image
        probs = torch.softmax(model_output, dim=-1)
        return (probs * self.levels).sum()


def collect_image_records():
    """Returns a list of dicts: {group, identifier, image_path}."""
    records = []

    ava_test = pd.read_csv(SPLITS_DIR / "test.csv")
    for _, r in ava_test.iterrows():
        records.append({
            "group": "ava_test",
            "identifier": str(int(r["image_num"])),
            "image_path": r["image_path"],
        })

    prov = pd.read_csv(PROV_CSV)
    kept = prov[prov["status"] == "kept"]
    for _, r in kept.iterrows():
        records.append({
            "group": r["region"],
            "identifier": r["filename"],
            "image_path": str(CBD_ROOT / r["region"] / r["filename"]),
        })

    return records


def main():
    print(f"device: {DEVICE}")
    print(f"loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Grad-CAM needs gradients; eval mode is correct, but we must not block grad.
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = [model.layer4[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)
    targets = [MeanScoreTarget(SCORE_LEVELS)]

    tf = make_transform()
    rgb_tf = make_rgb_loader()
    centre_mask, border_mask = make_centre_mask(IMG_SIZE, IMG_SIZE)

    records = collect_image_records()
    print(f"images to process: {len(records)}")
    print(f"  by group: {pd.DataFrame(records)['group'].value_counts().to_dict()}")

    rows = []
    # Cache the original-resolution PIL too, for later use on the 8 chosen ones.
    for rec in tqdm(records, desc="grad-cam"):
        try:
            img_pil = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  skip {rec['image_path']}: {e}")
            continue
        x = tf(img_pil).unsqueeze(0).to(DEVICE)
        cam = cam_extractor(input_tensor=x, targets=targets)[0]  # (H, W) in [0,1]
        ratio = border_centre_ratio(cam, centre_mask, border_mask)
        rows.append({
            "group": rec["group"],
            "identifier": rec["identifier"],
            "image_path": rec["image_path"],
            "border_centre_ratio": ratio,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(LOG_DIR / "gradcam_metrics.csv", index=False)
    print(f"\nsaved: {LOG_DIR / 'gradcam_metrics.csv'} ({len(metrics)} rows)")

    # Per-group stats + KS / Wasserstein vs AVA
    ref = metrics[metrics["group"] == "ava_test"]["border_centre_ratio"].to_numpy()
    summary_rows = []
    for grp in ["ava_test", "east_asia", "south_asia", "mena"]:
        vals = metrics[metrics["group"] == grp]["border_centre_ratio"].to_numpy()
        if grp == "ava_test":
            ks, p, w = 0.0, 1.0, 0.0
        else:
            ks_stat, p_val = ks_2samp(vals, ref)
            ks, p = float(ks_stat), float(p_val)
            w = float(wasserstein_distance(vals, ref))
        summary_rows.append({
            "group": grp,
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "ks_vs_ava": ks,
            "p_vs_ava": p,
            "wasserstein_vs_ava": w,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(LOG_DIR / "gradcam_summary.csv", index=False)

    print("\n=== Border-to-centre attention ratio, vs AVA-test ===")
    print(f"{'group':<12} {'n':>5} {'mean':>7} {'std':>6} {'p25':>6} {'med':>6} {'p75':>6} "
          f"{'KS':>7} {'p':>10} {'W':>7}")
    for r in summary_rows:
        print(f"{r['group']:<12} {r['n']:>5d} {r['mean']:>7.3f} {r['std']:>6.3f} "
              f"{r['p25']:>6.3f} {r['median']:>6.3f} {r['p75']:>6.3f} "
              f"{r['ks_vs_ava']:>7.4f} {r['p_vs_ava']:>10.2e} {r['wasserstein_vs_ava']:>7.4f}")

    # ---------- Pick 8 images by percentile of border ratio ----------
    picks = []  # list of (group, percentile_label, row_in_metrics)
    for grp in ["ava_test", "east_asia", "south_asia", "mena"]:
        sub = metrics[metrics["group"] == grp].copy()
        for q_label, q in [("p25", 0.25), ("p75", 0.75)]:
            target_val = np.quantile(sub["border_centre_ratio"].to_numpy(), q)
            # nearest image to that ratio
            idx = (sub["border_centre_ratio"] - target_val).abs().idxmin()
            picks.append((grp, q_label, sub.loc[idx]))

    print("\n=== 8 selected images (25th and 75th percentile of border ratio) ===")
    for grp, q_label, row in picks:
        print(f"  {grp:<12} {q_label}  ratio={row['border_centre_ratio']:.3f}  {row['identifier']}")

    # ---------- Render the 8-panel figure ----------
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=130)
    group_to_col = {"ava_test": 0, "east_asia": 1, "south_asia": 2, "mena": 3}
    label_to_row = {"p25": 0, "p75": 1}
    group_titles = {
        "ava_test": "AVA-test",
        "east_asia": "East Asia",
        "south_asia": "South Asia",
        "mena": "MENA",
    }
    for grp, q_label, row in picks:
        col = group_to_col[grp]
        rrow = label_to_row[q_label]
        ax = axes[rrow, col]
        img_pil = Image.open(row["image_path"]).convert("RGB")
        img_cropped = rgb_tf(img_pil)
        img_arr = np.array(img_cropped).astype(np.float32) / 255.0
        # Recompute Grad-CAM for this image (so the visualization is fresh)
        x = tf(img_pil).unsqueeze(0).to(DEVICE)
        cam = cam_extractor(input_tensor=x, targets=targets)[0]
        overlay = show_cam_on_image(img_arr, cam, use_rgb=True)
        ax.imshow(overlay)
        ax.set_xticks([]); ax.set_yticks([])
        if rrow == 0:
            ax.set_title(group_titles[grp], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{q_label} ratio", fontsize=10)
        ax.set_xlabel(f"r = {row['border_centre_ratio']:.2f}", fontsize=9)

    fig.suptitle("Grad-CAM attention on representative images "
                 "(rows: 25th and 75th percentile of border-to-centre ratio)",
                 fontsize=11)
    fig.tight_layout()
    out_fig = FIG_DIR / "gradcam_examples.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure: {out_fig}")


if __name__ == "__main__":
    main()
