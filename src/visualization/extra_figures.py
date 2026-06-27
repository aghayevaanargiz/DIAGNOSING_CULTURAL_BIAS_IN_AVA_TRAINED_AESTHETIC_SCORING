"""
extra_figures.py

Two additional paper figures in the unified polished style:

  fig_loss_curve.pdf / .png  -- training & validation EMD loss over
                                 epochs, with epoch-3 checkpoint marked
  fig_entropy.pdf / .png     -- distribution of per-image Grad-CAM
                                 attention entropy, by group
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- Paths ----------
ROOT = Path.home() / "aesthetics"
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------- Unified style (same as polish_figures.py) ----------
COLORS = {
    "ava_test":   "#3a3a3a",
    "east_asia":  "#c2543b",
    "south_asia": "#3e8e8a",
    "mena":       "#7a5ea8",
}
LABELS = {
    "ava_test":   "AVA-test",
    "east_asia":  "East Asia",
    "south_asia": "South Asia",
    "mena":       "MENA",
}
REGION_ORDER = ["east_asia", "south_asia", "mena"]

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =====================================================================
# Figure: training / validation loss curve
# =====================================================================
def fig_loss_curve():
    df = pd.read_csv(LOG_DIR / "train_log.csv")
    best_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    best_val = float(df["val_loss"].min())

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    ax.plot(df["epoch"], df["train_loss"],
            marker="o", markersize=3.5, linewidth=1.4,
            color=COLORS["ava_test"], label="Training EMD")
    ax.plot(df["epoch"], df["val_loss"],
            marker="s", markersize=3.5, linewidth=1.4,
            color=COLORS["east_asia"], label="Validation EMD")

    # Mark the retained checkpoint
    ax.scatter([best_epoch], [best_val], s=70, facecolors="none",
               edgecolors=COLORS["mena"], linewidths=1.5, zorder=5)
    ax.annotate(f"retained\ncheckpoint",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1.0, best_val - 0.012),
                fontsize=8, color="#444",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888",
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("EMD loss")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              fontsize=8.5, borderaxespad=0.3)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_loss_curve.pdf"
    out_png = FIG_DIR / "fig_loss_curve.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


# =====================================================================
# Figure: attention entropy distribution
# =====================================================================
def fig_entropy():
    df = pd.read_csv(LOG_DIR / "gradcam_metrics_v2.csv")

    ava = df[df["group"] == "ava_test"]["attention_entropy"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.0, 2.9))

    bins = np.linspace(0.55, 0.98, 50)
    ax.hist(ava, bins=bins, density=True,
            alpha=0.30, color=COLORS["ava_test"], linewidth=0,
            label=LABELS["ava_test"])
    for r in REGION_ORDER:
        v = df[df["group"] == r]["attention_entropy"].to_numpy()
        ax.hist(v, bins=bins, density=True,
                histtype="step", linewidth=1.7,
                color=COLORS[r], label=LABELS[r])

    # Subtle reference line at AVA median
    ax.axvline(float(np.median(ava)), color=COLORS["ava_test"],
               linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Grad-CAM attention entropy (normalised)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.55, 0.98)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              fontsize=8.5, borderaxespad=0.3)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_entropy.pdf"
    out_png = FIG_DIR / "fig_entropy.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


def main():
    print("Generating additional polished figures...")
    fig_loss_curve()
    fig_entropy()
    print("\nDone.")


if __name__ == "__main__":
    main()
