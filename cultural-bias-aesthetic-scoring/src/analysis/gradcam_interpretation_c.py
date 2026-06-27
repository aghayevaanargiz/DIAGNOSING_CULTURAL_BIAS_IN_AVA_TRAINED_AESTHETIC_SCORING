"""
gradcam_interpretation_c.py

Adds two refinements to the Grad-CAM analysis:

1. Trimmed border-to-centre ratio (cap at 95th percentile per group),
   to control for the outlier distortion seen in AVA (std 2.644).

2. Attention entropy: Shannon entropy of the (normalised) Grad-CAM
   heatmap. Low entropy = focused on a small area. High entropy =
   diffuse, spread out. Captures "how decisively the model attends",
   independent of where the attention is.

Re-runs Grad-CAM on all 3008 images to compute both metrics in one
pass, then KS / Wasserstein vs AVA-test for each.

Outputs:
  logs/gradcam_metrics_v2.csv     -- per-image: ratio + entropy
  logs/gradcam_summary_v2.csv     -- per-group stats + tests
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path.home() / "aesthetics"
CKPT_PATH = ROOT / "checkpoints" / "resnet18_ava_best.pt"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

SPLITS_DIR = ROOT / "data" / "splits"
CBD_ROOT = Path("/home/tempuser1/Desktop/AESTHETICS_RESEARCH/cultural-bias-data")
PROV_CSV = CBD_ROOT / "provenance.csv"

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32).to(DEVICE)
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
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_centre_mask(h, w, inner_frac=INNER_FRAC):
    side = int(round(np.sqrt(inner_frac) * min(h, w)))
    cy, cx = h // 2, w // 2
    y0, x0 = cy - side // 2, cx - side // 2
    centre = np.zeros((h, w), dtype=np.float32)
    centre[y0:y0 + side, x0:x0 + side] = 1.0
    border = 1.0 - centre
    return centre, border


def border_centre_ratio(cam, centre_mask, border_mask):
    centre_mass = (cam * centre_mask).sum() / max(centre_mask.sum(), 1)
    border_mass = (cam * border_mask).sum() / max(border_mask.sum(), 1)
    return float(border_mass / (centre_mass + 1e-8))


def attention_entropy(cam):
    """Shannon entropy of the heatmap as a probability distribution
    over pixels. Higher = more diffuse attention. Returns the
    normalised entropy in [0, 1] (divided by log(N) where N = pixels)."""
    p = cam.astype(np.float64).flatten()
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total < 1e-12:
        return 0.0
    p = p / total
    nz = p[p > 0]
    H = -(nz * np.log(nz)).sum()
    Hmax = np.log(len(p))
    return float(H / Hmax)


class MeanScoreTarget:
    def __init__(self, levels):
        self.levels = levels

    def __call__(self, model_output):
        probs = torch.softmax(model_output, dim=-1)
        return (probs * self.levels).sum()


def collect_records():
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


def trim_outliers(values, cap_percentile=95):
    """Cap values above the given percentile (computed within the array)."""
    cap = np.percentile(values, cap_percentile)
    return np.clip(values, None, cap)


def compare_vs_ref(values, ref):
    ks_stat, p = ks_2samp(values, ref)
    w = wasserstein_distance(values, ref)
    return float(ks_stat), float(p), float(w)


def main():
    print(f"device: {DEVICE}")
    print(f"loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = [model.layer4[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)
    targets = [MeanScoreTarget(SCORE_LEVELS)]

    tf = make_transform()
    centre_mask, border_mask = make_centre_mask(IMG_SIZE, IMG_SIZE)

    records = collect_records()
    print(f"images: {len(records)}")

    rows = []
    for rec in tqdm(records, desc="grad-cam v2"):
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  skip {rec['image_path']}: {e}")
            continue
        x = tf(img).unsqueeze(0).to(DEVICE)
        cam = cam_extractor(input_tensor=x, targets=targets)[0]
        rows.append({
            "group": rec["group"],
            "identifier": rec["identifier"],
            "border_centre_ratio": border_centre_ratio(cam, centre_mask, border_mask),
            "attention_entropy": attention_entropy(cam),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(LOG_DIR / "gradcam_metrics_v2.csv", index=False)
    print(f"\nsaved: {LOG_DIR / 'gradcam_metrics_v2.csv'} ({len(metrics)} rows)")

    # Build group arrays
    by_group = {g: metrics[metrics["group"] == g] for g in
                ["ava_test", "east_asia", "south_asia", "mena"]}

    ref_ratio = by_group["ava_test"]["border_centre_ratio"].to_numpy()
    ref_ratio_trim = trim_outliers(ref_ratio, 95)
    ref_entropy = by_group["ava_test"]["attention_entropy"].to_numpy()

    summary = []
    for g, sub in by_group.items():
        ratio = sub["border_centre_ratio"].to_numpy()
        ratio_trim = trim_outliers(ratio, 95)
        entropy = sub["attention_entropy"].to_numpy()
        if g == "ava_test":
            ks_r, p_r, w_r = 0.0, 1.0, 0.0
            ks_rt, p_rt, w_rt = 0.0, 1.0, 0.0
            ks_e, p_e, w_e = 0.0, 1.0, 0.0
        else:
            ks_r, p_r, w_r = compare_vs_ref(ratio, ref_ratio)
            ks_rt, p_rt, w_rt = compare_vs_ref(ratio_trim, ref_ratio_trim)
            ks_e, p_e, w_e = compare_vs_ref(entropy, ref_entropy)
        summary.append({
            "group": g, "n": int(len(sub)),
            # raw ratio (legacy)
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_ks": ks_r, "ratio_p": p_r, "ratio_W": w_r,
            # trimmed ratio
            "ratio_trim_mean": float(ratio_trim.mean()),
            "ratio_trim_std": float(ratio_trim.std(ddof=1)),
            "ratio_trim_ks": ks_rt, "ratio_trim_p": p_rt, "ratio_trim_W": w_rt,
            # entropy
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std(ddof=1)),
            "entropy_p25": float(np.percentile(entropy, 25)),
            "entropy_median": float(np.median(entropy)),
            "entropy_p75": float(np.percentile(entropy, 75)),
            "entropy_ks": ks_e, "entropy_p": p_e, "entropy_W": w_e,
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(LOG_DIR / "gradcam_summary_v2.csv", index=False)

    print("\n=== Border ratio: RAW vs TRIMMED (cap at 95th pct) ===")
    print(f"{'group':<12} {'raw_mean':>9} {'raw_med':>8} {'trim_mean':>10} {'trim_std':>9} "
          f"{'KS_trim':>8} {'p_trim':>10} {'W_trim':>8}")
    for r in summary:
        print(f"{r['group']:<12} {r['ratio_mean']:>9.3f} {r['ratio_median']:>8.3f} "
              f"{r['ratio_trim_mean']:>10.3f} {r['ratio_trim_std']:>9.3f} "
              f"{r['ratio_trim_ks']:>8.4f} {r['ratio_trim_p']:>10.2e} {r['ratio_trim_W']:>8.4f}")

    print("\n=== Attention entropy (Shannon, normalised to [0,1]) ===")
    print(f"{'group':<12} {'mean':>7} {'std':>6} {'p25':>6} {'med':>6} {'p75':>6} "
          f"{'KS':>7} {'p':>10} {'W':>8}")
    for r in summary:
        print(f"{r['group']:<12} {r['entropy_mean']:>7.4f} {r['entropy_std']:>6.4f} "
              f"{r['entropy_p25']:>6.4f} {r['entropy_median']:>6.4f} {r['entropy_p75']:>6.4f} "
              f"{r['entropy_ks']:>7.4f} {r['entropy_p']:>10.2e} {r['entropy_W']:>8.4f}")


if __name__ == "__main__":
    main()
