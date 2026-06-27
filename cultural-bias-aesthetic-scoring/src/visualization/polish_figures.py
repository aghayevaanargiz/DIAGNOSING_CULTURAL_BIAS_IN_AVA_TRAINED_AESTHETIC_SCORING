"""
polish_figures.py

Regenerate all paper figures in a unified, polished style.

Outputs (in ~/aesthetics/figures/, both PDF and PNG):
  fig_distributions.pdf / .png   (1x2 panel: score + calibration)
  fig_gradcam.pdf / .png         (2x4 panel: Grad-CAM overlays)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------- Paths ----------
ROOT = Path.home() / "aesthetics"
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
CKPT_PATH = ROOT / "checkpoints" / "resnet18_ava_best.pt"
CBD_ROOT = Path("/home/tempuser1/Desktop/AESTHETICS_RESEARCH/cultural-bias-data")
SPLITS_DIR = ROOT / "data" / "splits"

# ---------- Style ----------
# Tufte-inspired muted palette. AVA grey is the reference;
# the three regions are warm-cool-cool to avoid hue collisions.
COLORS = {
    "ava_test":   "#3a3a3a",   # near-black grey, calm reference
    "east_asia":  "#c2543b",   # muted terracotta
    "south_asia": "#3e8e8a",   # muted teal
    "mena":       "#7a5ea8",   # muted plum
}
LABELS = {
    "ava_test":   "AVA-test",
    "east_asia":  "East Asia",
    "south_asia": "South Asia",
    "mena":       "MENA",
}
REGION_ORDER = ["east_asia", "south_asia", "mena"]

# Set global plot style
rcParams.update({
    "font.family": "serif",
    "font.serif":  ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,    # editable text in PDF (TrueType, not Type 3)
    "ps.fonttype": 42,
})


# =====================================================================
# Figure 1: predicted score distributions + per-image prediction width
# =====================================================================
def fig_distributions():
    ava = pd.read_csv(LOG_DIR / "ava_test_predictions.csv")
    nw  = pd.read_csv(LOG_DIR / "nonwestern_predictions.csv")
    shift = pd.read_csv(LOG_DIR / "dist_shift_summary.csv")
    shift = shift.set_index("group")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # ----- (a) predicted mean score -----
    ax = axes[0]
    bins = np.linspace(2.5, 8.0, 46)
    # AVA filled
    ax.hist(ava["pred_mean"], bins=bins, density=True,
            alpha=0.30, color=COLORS["ava_test"], linewidth=0,
            label=f'{LABELS["ava_test"]}')
    # Regions outlined
    for r in REGION_ORDER:
        v = nw[nw["region"] == r]["pred_mean"]
        ax.hist(v, bins=bins, density=True,
                histtype="step", linewidth=1.7,
                color=COLORS[r], label=f"{LABELS[r]}")
    ax.set_xlabel("Predicted mean aesthetic score")
    ax.set_ylabel("Density")
    ax.set_xlim(2.5, 8.0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4,
              fontsize=8.5, borderaxespad=0.3)
    # Subtle reference line at AVA mean
    ava_mean = ava["pred_mean"].mean()
    ax.axvline(ava_mean, color=COLORS["ava_test"],
               linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title("(a) Predicted score distributions", loc="left", pad=8)

    # ----- (b) per-image predicted-distribution width -----
    ax = axes[1]
    bins = np.linspace(1.0, 2.0, 36)
    ax.hist(ava["pred_std"], bins=bins, density=True,
            alpha=0.30, color=COLORS["ava_test"], linewidth=0,
            label=f'{LABELS["ava_test"]}')
    for r in REGION_ORDER:
        v = nw[nw["region"] == r]["pred_std"]
        ax.hist(v, bins=bins, density=True,
                histtype="step", linewidth=1.7,
                color=COLORS[r], label=f"{LABELS[r]}")
    ax.set_xlabel("Per-image predicted distribution width")
    ax.set_ylabel("Density")
    ax.set_xlim(1.0, 2.0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4,
              fontsize=8.5, borderaxespad=0.3)
    ax.set_title("(b) Per-image prediction width (calibration)",
                 loc="left", pad=8)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_distributions.pdf"
    out_png = FIG_DIR / "fig_distributions.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


# =====================================================================
# Figure 2: Grad-CAM overlays  (regenerate with polished style)
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32).to(DEVICE)


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_rgb():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
    ])


class MeanScoreTarget:
    def __init__(self, levels): self.levels = levels
    def __call__(self, out):
        p = torch.softmax(out, dim=-1)
        return (p * self.levels).sum()


def resolve_image_path(group, identifier):
    if group == "ava_test":
        return SPLITS_DIR.parent / "images" / f"{identifier}.jpg"
    return CBD_ROOT / group / identifier


def fig_gradcam():
    metrics = pd.read_csv(LOG_DIR / "gradcam_metrics.csv")

    # 25th/75th percentile picks per group
    picks = []
    for grp in ["ava_test", "east_asia", "south_asia", "mena"]:
        sub = metrics[metrics["group"] == grp].copy()
        ratios = sub["border_centre_ratio"].to_numpy()
        for q_label, q in [("p25", 0.25), ("p75", 0.75)]:
            target = np.quantile(ratios, q)
            idx = (sub["border_centre_ratio"] - target).abs().idxmin()
            picks.append((grp, q_label, sub.loc[idx]))

    # Load model
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [MeanScoreTarget(SCORE_LEVELS)]
    tf = make_transform()
    rgb_tf = make_rgb()

    # Grid: 2 rows (p25, p75) x 4 cols (groups)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.18})
    group_to_col = {"ava_test": 0, "east_asia": 1, "south_asia": 2, "mena": 3}
    label_to_row = {"p25": 0, "p75": 1}

    for grp, q_label, row in picks:
        col = group_to_col[grp]
        rrow = label_to_row[q_label]
        ax = axes[rrow, col]
        path = resolve_image_path(grp, row["identifier"])
        img_pil = Image.open(path).convert("RGB")
        img_crop = rgb_tf(img_pil)
        img_arr = np.array(img_crop).astype(np.float32) / 255.0
        x = tf(img_pil).unsqueeze(0).to(DEVICE)
        cam = cam_extractor(input_tensor=x, targets=targets)[0]
        overlay = show_cam_on_image(img_arr, cam, use_rgb=True,
                                     image_weight=0.55)
        ax.imshow(overlay)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if rrow == 0:
            ax.set_title(LABELS[grp], pad=4, fontsize=10.5)
        if col == 0:
            ax.set_ylabel(
                "25th percentile\nof border ratio" if q_label == "p25"
                else "75th percentile\nof border ratio",
                fontsize=9, rotation=90, labelpad=8)
        ax.set_xlabel(f"r = {row['border_centre_ratio']:.2f}",
                      fontsize=8.5, color="#444", labelpad=2)

    out_pdf = FIG_DIR / "fig_gradcam.pdf"
    out_png = FIG_DIR / "fig_gradcam.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


# =====================================================================
def main():
    print("Regenerating polished figures...")
    fig_distributions()
    fig_gradcam()
    print("\nDone. Use the .pdf versions in the paper.")


if __name__ == "__main__":
    main()
