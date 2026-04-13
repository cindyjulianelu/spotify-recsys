"""
End-to-end training script for the two-stage recommendation pipeline.

Stage 1 — Genre classifier
    LightGBM trained on audio features to predict track_genre.
    Evaluated on a 20% held-out test set: accuracy, macro-F1, top-5 accuracy.

Stage 2 — Recommendation index
    Genre probability embeddings (predict_proba output) are extracted for
    all 113k tracks and indexed with cosine-similarity nearest neighbours.

Saved artefacts (to models/):
    classifier.pkl   — fitted LGBMClassifier
    embeddings.npy   — (n_tracks, 114) probability matrix
    recommender.pkl  — fitted TrackRecommender (NearestNeighbors + metadata)

Run from the repo root:
    python src/models/train.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the Python path so cross-package imports work
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from features.engineer import build_features, split
import models.classifier as clf_module
from models.recommender import TrackRecommender

CLEANED_PATH   = Path(__file__).resolve().parents[2] / "data"    / "cleaned.csv"
EMBEDDINGS_PATH = Path(__file__).resolve().parents[2] / "models" / "embeddings.npy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_metrics(metrics: dict) -> None:
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  Top-5 accuracy: {metrics['top5_accuracy']:.4f}")


def _demo(rec: TrackRecommender, df: pd.DataFrame, random_state: int = 42) -> None:
    sample = df.sample(1, random_state=random_state).iloc[0]
    print("\n  Query track:")
    print(f"    {sample['track_name']} — {sample['artists']} [{sample['track_genre']}]")
    try:
        recs = rec.recommend_by_name(sample["track_name"], k=5)
        print("\n  Top-5 recommendations:")
        for _, row in recs.iterrows():
            print(
                f"    [{row['similarity']:.3f}]  {row['track_name']} "
                f"— {row['artists']}  ({row['track_genre']})"
            )
    except KeyError as e:
        print(f"  Demo skipped: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"\n{'='*60}")
    print("  Two-Stage Recommendation Pipeline")
    print(f"{'='*60}")

    # ---- Load ----
    print("\n[1/5] Loading cleaned data...")
    df = pd.read_csv(CLEANED_PATH, index_col=0)
    print(f"  {len(df):,} tracks × {len(df.columns)} columns")

    # ---- Split ----
    print("\n[2/5] Splitting train / test (80/20, stratified)...")
    train_df, test_df = split(df)
    X_train, y_train, feature_cols = build_features(train_df)
    X_test,  y_test,  _            = build_features(test_df)
    print(f"  Train : {X_train.shape[0]:,} rows  |  Test : {X_test.shape[0]:,} rows")
    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # ---- Train classifier ----
    print("\n[3/5] Training genre classifier (LightGBM)...")
    model = clf_module.train(X_train, y_train)
    print("  Done.")

    print("\n  Evaluation on held-out test set:")
    metrics = clf_module.evaluate(model, X_test, y_test)
    _print_metrics(metrics)

    # ---- Extract embeddings for all tracks ----
    print("\n[4/5] Extracting genre embeddings for all tracks...")
    X_all, _, _ = build_features(df)
    embeddings = clf_module.get_embeddings(model, X_all)
    print(f"  Embedding matrix: {embeddings.shape}  (tracks × genres)")

    # ---- Build recommendation index ----
    print("\n[5/5] Building recommendation index...")
    rec = TrackRecommender(metric="cosine")
    rec.fit(embeddings, df)

    # ---- Save artefacts ----
    clf_path = clf_module.save(model)
    np.save(EMBEDDINGS_PATH, embeddings)
    rec_path = rec.save()

    print("\n  Saved:")
    print(f"    classifier   → {clf_path}")
    print(f"    embeddings   → {EMBEDDINGS_PATH}")
    print(f"    recommender  → {rec_path}")

    # ---- Demo ----
    print("\n  Quick demo:")
    _demo(rec, df)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
