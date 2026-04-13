"""
Reproducible feature engineering pipeline.

Loads data/cleaned.csv → engineers 12 new features → removes redundant
and low-variance features → saves to data/features.csv.

Run from the repo root:
    python src/features/run_features.py

Output: data/features.csv — the enriched, deduplicated feature set
used by src/models/train.py for classifier training and embedding
extraction.
"""

import sys
import time
from pathlib import Path

# Add src/ to path so cross-package imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from features.engineering import create_features, select_features

CLEANED_PATH  = Path(__file__).resolve().parents[2] / "data" / "cleaned.csv"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "features.csv"


def run() -> None:
    print(f"\n{'='*56}")
    print("  Feature Engineering Pipeline")
    print(f"{'='*56}")

    # ---- Load ----
    print(f"\nLoading {CLEANED_PATH.name}...")
    df = pd.read_csv(CLEANED_PATH, index_col=0)
    print(f"  Input : {df.shape[0]:,} rows × {df.shape[1]} columns")

    t0 = time.time()

    # ---- Create features ----
    df_eng = create_features(df)
    new_cols = [c for c in df_eng.columns if c not in df.columns]
    print(f"\nEngineered {len(new_cols)} new feature(s):")
    for col in new_cols:
        print(f"  + {col}")

    # ---- Select features ----
    df_sel, selected_cols, log = select_features(df_eng)

    if log:
        print(f"\nFeature selection — dropped {len(log)} feature(s):")
        for entry in log:
            print(f"  - {entry}")
    else:
        print("\nFeature selection — no features dropped.")

    elapsed = time.time() - t0

    # ---- Save ----
    df_sel.to_csv(FEATURES_PATH)
    print(f"\nSaved → {FEATURES_PATH}")
    print(f"  Output: {df_sel.shape[0]:,} rows × {df_sel.shape[1]} columns")
    print(f"  Numeric features kept ({len(selected_cols)}): {', '.join(selected_cols)}")
    print(f"\n  Elapsed: {elapsed:.2f}s")
    print(f"\n{'='*56}\n")


if __name__ == "__main__":
    run()
