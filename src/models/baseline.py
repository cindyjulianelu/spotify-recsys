"""
Baseline models — the floor everything else must beat.

Two reference points deliberately chosen:

  DummyClassifier (most_frequent)
      Predicts the most common genre every time. For a balanced
      114-class problem this is ~0.88% accuracy — the true floor.
      Any model that can't beat this is useless.

  LogisticRegression
      The best linear model. If non-linear models don't clearly
      beat this, the problem doesn't need complexity. It also
      provides a sensible AUC and top-5 reference.

Both are evaluated with 5-fold stratified CV (not just a single
train/test split) so the reported numbers are robust to partition luck.
Top-5 accuracy is reported alongside top-1 because for a 114-class
recommendation problem, "correct genre in top 5" is a more meaningful
bar than exact match.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features.engineer import build_features, split

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "features.csv"
MODEL_DIR     = Path(__file__).resolve().parents[2] / "models"

RANDOM_BASELINE = 1.0 / 114   # ~0.0088 — what random guessing gets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, object]:
    """Train both baseline models and return a name → fitted model dict."""
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)

    logreg = LogisticRegression(
        max_iter=1000,
        solver="saga",          # fast for large multiclass
        multi_class="multinomial",
        C=1.0,
        n_jobs=-1,
        random_state=42,
    )
    logreg.fit(X_train, y_train)

    return {"dummy": dummy, "logreg": logreg}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baselines(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Evaluate both baselines and return a formatted comparison DataFrame.

    CV is stratified to handle the 114-class balance.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for name, model in models.items():
        # Cross-validated accuracy (robust estimate)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        top5 = top_k_accuracy_score(
            y_test, y_proba, k=5, labels=model.classes_
        )
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )

        rows.append({
            "model":            name,
            "cv_accuracy":      f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "test_accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "test_f1_macro":    round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "test_top5_acc":    round(top5, 4),
            "test_auc_macro":   round(auc, 4),
            "vs_random":        f"{accuracy_score(y_test, y_pred) / RANDOM_BASELINE:.1f}×",
        })

    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_baselines(models: dict[str, object]) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        path = MODEL_DIR / f"baseline_{name}.pkl"
        joblib.dump(model, path)
        print(f"  Saved → {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Baseline Models")
    print(f"{'='*60}")
    print(f"\n  Random baseline (114-class uniform): {RANDOM_BASELINE*100:.2f}%")

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    train_df, test_df = split(df)
    X_train, y_train, _ = build_features(train_df)
    X_test,  y_test,  _ = build_features(test_df)

    print("\nTraining baselines...")
    t0 = time.time()
    models = train_baselines(X_train, y_train)
    elapsed = time.time() - t0

    print("\nEvaluating (5-fold CV + test set)...")
    results = evaluate_baselines(models, X_train, y_train, X_test, y_test)

    print(f"\n{results.to_string()}")
    print(f"\n  vs_random = test_accuracy ÷ {RANDOM_BASELINE*100:.2f}% (uniform random)")

    save_baselines(models)
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n{'='*60}\n")
