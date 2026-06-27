"""
dist_shift_analysis.py

Formal distribution-shift analysis. Compares predicted-score
distributions on AVA-test against the three non-Western regions.

For each region, computes vs AVA-test:
  - KS statistic + p-value on predicted mean score
  - Wasserstein distance on predicted mean score
  - KS + Wasserstein on per-image predicted-distribution width
    (calibration / uncertainty axis)

Reads:  logs/ava_test_predictions.csv, logs/nonwestern_predictions.csv
Writes: figures/score_distributions.png
        figures/calibration_widths.png
        logs/dist_shift_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path.home() / "aesthetics"
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

AVA_CSV = LOG_DIR / "ava_test_predictions.csv"
NW_CSV = LOG_DIR / "nonwestern_predictions.csv"

REGION_ORDER = ["east_asia", "south_asia", "mena"]
REGION_LABELS = {
    "east_asia": "East Asia",
    "south_asia": "South Asia",
    "mena": "MENA",
}
REGION_COLORS = {
    "ava_test":   "#444444",
    "east_asia":  "#e8743b",
    "south_asia": "#19a979",
    "mena":       "#945ecf",
}


def summarize(name, values):
    return {
        "group": name,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compare_to_ref(values, ref):
    ks_stat, ks_p = ks_2samp(values, ref)
    wd = wasserstein_distance(values, ref)
    return float(ks_stat), float(ks_p), float(wd)


def main():
    print(f"loading: {AVA_CSV}")
    ava = pd.read_csv(AVA_CSV)
    print(f"loading: {NW_CSV}")
    nw = pd.read_csv(NW_CSV)

    print(f"\nAVA-test n={len(ava)}, non-Western n={len(nw)}")
    print(f"non-Western per region: {nw['region'].value_counts().to_dict()}")

    ref_mean = ava["pred_mean"].to_numpy()
    ref_std = ava["pred_std"].to_numpy()

    rows = []
    rows.append({
        **summarize("ava_test", ref_mean),
        "ks_vs_ava_mean": 0.0,
        "ks_p_vs_ava_mean": 1.0,
        "wasserstein_vs_ava_mean": 0.0,
        "ks_vs_ava_width": 0.0,
        "ks_p_vs_ava_width": 1.0,
        "wasserstein_vs_ava_width": 0.0,
        "pred_width_mean": float(np.mean(ref_std)),
        "pred_width_std": float(np.std(ref_std, ddof=1)),
    })

    for region in REGION_ORDER:
        sub = nw[nw["region"] == region]
        vals_mean = sub["pred_mean"].to_numpy()
        vals_width = sub["pred_std"].to_numpy()

        ks_m, ksp_m, w_m = compare_to_ref(vals_mean, ref_mean)
        ks_w, ksp_w, w_w = compare_to_ref(vals_width, ref_std)

        rows.append({
            **summarize(region, vals_mean),
            "ks_vs_ava_mean": ks_m,
            "ks_p_vs_ava_mean": ksp_m,
            "wasserstein_vs_ava_mean": w_m,
            "ks_vs_ava_width": ks_w,
            "ks_p_vs_ava_width": ksp_w,
            "wasserstein_vs_ava_width": w_w,
            "pred_width_mean": float(np.mean(vals_width)),
            "pred_width_std": float(np.std(vals_width, ddof=1)),
        })

    summary = pd.DataFrame(rows)
    out_csv = LOG_DIR / "dist_shift_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved summary: {out_csv}")

    # Pretty-print
    print("\n=== Predicted mean score, vs AVA-test ===")
    print(f"{'group':<12} {'n':>5} {'mean':>7} {'std':>6} {'min':>6} {'max':>6} "
          f"{'KS':>7} {'p':>10} {'W':>7}")
    for r in rows:
        print(f"{r['group']:<12} {r['n']:>5d} {r['mean']:>7.3f} {r['std']:>6.3f} "
              f"{r['min']:>6.3f} {r['max']:>6.3f} "
              f"{r['ks_vs_ava_mean']:>7.4f} {r['ks_p_vs_ava_mean']:>10.2e} "
              f"{r['wasserstein_vs_ava_mean']:>7.4f}")

    print("\n=== Per-image predicted distribution width (uncertainty), vs AVA-test ===")
    print(f"{'group':<12} {'mean':>7} {'std':>6} "
          f"{'KS':>7} {'p':>10} {'W':>7}")
    for r in rows:
        print(f"{r['group']:<12} {r['pred_width_mean']:>7.3f} {r['pred_width_std']:>6.3f} "
              f"{r['ks_vs_ava_width']:>7.4f} {r['ks_p_vs_ava_width']:>10.2e} "
              f"{r['wasserstein_vs_ava_width']:>7.4f}")

    # ---------- Figure 1: predicted mean score distributions ----------
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    bins = np.linspace(2.5, 8.0, 45)
    ax.hist(ref_mean, bins=bins, density=True, alpha=0.35,
            color=REGION_COLORS["ava_test"], label=f"AVA-test (n={len(ref_mean)})")
    for region in REGION_ORDER:
        vals = nw[nw["region"] == region]["pred_mean"].to_numpy()
        ax.hist(vals, bins=bins, density=True, alpha=0.55, histtype="step",
                linewidth=2.0, color=REGION_COLORS[region],
                label=f"{REGION_LABELS[region]} (n={len(vals)})")
    ax.set_xlabel("Predicted mean aesthetic score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of predicted mean scores")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out1 = FIG_DIR / "score_distributions.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure: {out1}")

    # ---------- Figure 2: predicted distribution widths ----------
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    bins = np.linspace(1.0, 2.0, 35)
    ax.hist(ref_std, bins=bins, density=True, alpha=0.35,
            color=REGION_COLORS["ava_test"], label=f"AVA-test (n={len(ref_std)})")
    for region in REGION_ORDER:
        vals = nw[nw["region"] == region]["pred_std"].to_numpy()
        ax.hist(vals, bins=bins, density=True, alpha=0.55, histtype="step",
                linewidth=2.0, color=REGION_COLORS[region],
                label=f"{REGION_LABELS[region]} (n={len(vals)})")
    ax.set_xlabel("Predicted score-distribution width per image")
    ax.set_ylabel("Density")
    ax.set_title("Per-image prediction uncertainty (calibration view)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out2 = FIG_DIR / "calibration_widths.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out2}")


if __name__ == "__main__":
    main()
