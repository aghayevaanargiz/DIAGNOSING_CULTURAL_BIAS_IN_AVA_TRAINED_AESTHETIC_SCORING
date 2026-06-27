"""
prep_ava.py

Prepare a reproducible 35k subset of AVA with train/val/test splits.

Outputs three CSVs to ~/aesthetics/data/splits/:
    train.csv  (~28k rows, 80%)
    val.csv    (~3.5k rows, 10%)
    test.csv   (~3.5k rows, 10%)

Each CSV has columns:
    image_num, vote_1..vote_10, mean_score, std_score, image_path

Stratified by mean-score bin so all rating ranges are represented.
Fixed seed for reproducibility.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path.home() / "aesthetics" / "data"
CSV_PATH = DATA_DIR / "ground_truth_dataset.csv"
IMAGES_DIR = DATA_DIR / "images"
SPLITS_DIR = DATA_DIR / "splits"

SUBSET_SIZE = 35000
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
N_STRATA = 10

RANDOM_SEED = 42


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  loaded {len(df)} rows")

    vote_cols = [f"vote_{i}" for i in range(1, 11)]
    scores = np.arange(1, 11)
    probs = df[vote_cols].to_numpy()
    df["mean_score"] = probs @ scores
    df["std_score"] = np.sqrt(probs @ (scores ** 2) - df["mean_score"].to_numpy() ** 2)

    print(f"  mean_score range: {df['mean_score'].min():.3f} to {df['mean_score'].max():.3f}")
    print(f"  mean of mean_scores: {df['mean_score'].mean():.3f}")

    print("Verifying image files exist on disk ...")
    df["image_path"] = df["image_num"].astype(str).apply(
        lambda n: str(IMAGES_DIR / f"{n}.jpg")
    )
    exists_mask = np.array(
        [os.path.exists(p) for p in tqdm(df["image_path"], desc="  stat")]
    )
    missing = (~exists_mask).sum()
    print(f"  missing files: {missing}")
    df = df[exists_mask].reset_index(drop=True)
    print(f"  rows after file-existence filter: {len(df)}")

    print(f"Sampling {SUBSET_SIZE} images stratified into {N_STRATA} bins ...")
    df["_stratum"] = pd.cut(df["mean_score"], bins=N_STRATA, labels=False)
    per_stratum_quota = SUBSET_SIZE // N_STRATA

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    for s in range(N_STRATA):
        bucket = df[df["_stratum"] == s]
        take = min(per_stratum_quota, len(bucket))
        if take < per_stratum_quota:
            print(f"  WARN: stratum {s} has only {len(bucket)} images, requested {per_stratum_quota}")
        idx = rng.choice(bucket.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(df.loc[idx])
    subset = pd.concat(sampled_parts, ignore_index=True)
    print(f"  sampled {len(subset)} images")
    print("  stratum counts:")
    print(subset["_stratum"].value_counts().sort_index().to_string())

    subset = subset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(subset)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = subset.iloc[:n_train]
    val = subset.iloc[n_train : n_train + n_val]
    test = subset.iloc[n_train + n_val :]

    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["image_num"] + vote_cols + ["mean_score", "std_score", "image_path"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out_path = SPLITS_DIR / f"{name}.csv"
        part[out_cols].to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({len(part)} rows)")

    print("\nMean-score distribution per split:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        ms = part["mean_score"]
        print(f"  {name}: mean={ms.mean():.3f}  std={ms.std():.3f}  min={ms.min():.3f}  max={ms.max():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
