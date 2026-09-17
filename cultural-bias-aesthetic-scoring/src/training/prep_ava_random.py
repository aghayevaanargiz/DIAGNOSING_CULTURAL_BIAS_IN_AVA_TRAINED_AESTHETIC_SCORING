import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path.home() / "aesthetics" / "data"
CSV_PATH = DATA_DIR / "ground_truth_dataset.csv"
IMAGES_DIR = DATA_DIR / "images"
SPLITS_DIR = DATA_DIR / "splits"

SAMPLE_SIZE = 2321
RANDOM_SEED = 123

def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  loaded {len(df)} rows")

    vote_cols = [f"vote_{i}" for i in range(1, 11)]
    scores = np.arange(1, 11)
    probs = df[vote_cols].to_numpy()
    df["mean_score"] = probs @ scores
    df["std_score"] = np.sqrt(probs @ (scores ** 2) - df["mean_score"].to_numpy() ** 2)

    df["image_path"] = df["image_num"].astype(str).apply(lambda n: str(IMAGES_DIR / f"{n}.jpg"))
    exists_mask = np.array([os.path.exists(p) for p in tqdm(df["image_path"], desc="  stat")])
    df = df[exists_mask].reset_index(drop=True)
    print(f"  rows after file existence filter: {len(df)}")

    used_nums = set()
    for name in ["train.csv", "val.csv"]:
        p = SPLITS_DIR / name
        if p.exists():
            used_nums.update(pd.read_csv(p)["image_num"].tolist())
    print(f"  excluding {len(used_nums)} images already used in train or val")
    pool = df[~df["image_num"].isin(used_nums)].reset_index(drop=True)
    print(f"  pool size after exclusion: {len(pool)}")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(pool.index.to_numpy(), size=SAMPLE_SIZE, replace=False)
    sample = pool.loc[idx].reset_index(drop=True)

    print(f"Sampled {len(sample)} images at random, no stratification")
    print(f"  mean_score: avg {sample['mean_score'].mean():.3f}  std {sample['mean_score'].std():.3f}")

    out_cols = ["image_num"] + vote_cols + ["mean_score", "std_score", "image_path"]
    out_path = SPLITS_DIR / "test_random.csv"
    sample[out_cols].to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

if __name__ == "__main__":
    main()
