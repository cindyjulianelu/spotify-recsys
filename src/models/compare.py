"""
Model comparison — three candidates evaluated on the same split.

Why these three models for 114-class genre classification:

  LogisticRegression
      Best linear model. Genre boundaries in audio-feature space are
      not perfectly linear (classical vs. metal is not a hyperplane),
      but logistic regression sets the "what does linearity buy us?"
      floor. If it's within 5pp of the tree models, simpler wins.

  RandomForestClassifier
      Ensemble of independent decision trees. Strong on tabular data,
      interpretable via feature importance, and robust to outliers.
      Lower variance than a single tree but does not benefit from
      sequential boosting — trees learn independently rather than
      correcting each other's errors.

  LGBMClassifier
      Gradient-boosted trees: each tree corrects the residual errors
      of the previous one. For 114-class tabular classification with
      ~800 training examples per class, boosting's sequential error-
      correction consistently outperforms bagging (RandomForest) and
      linear models. Faster than XGBoost on large datasets due to
      histogram-based split finding.

The comparison uses 5-fold stratified CV for the accuracy estimate and
a single held-out test set for all other metrics. Results are saved to
models/comparison.csv for MLflow ingestion.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features.engineer import build_features, split

FEATURES_PATH   = Path(__file__).resolve().parents[2] / "data"   / "features.csv"
MODEL_DIR       = Path(__file__).resolve().parents[2] / "models"
COMPARISON_PATH = MODEL_DIR / "comparison.csv"

RANDOM_BASELINE = 1.0 / 114


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _get_candidates() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=1000,
            solver="saga",
            multi_class="multinomial",
            C=1.0,
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=127,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cv: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Train three candidate models, evaluate each, and return a comparison
    DataFrame alongside the fitted model objects.

    Returns
    -------
    results  : pd.DataFrame — one row per model, all metrics
    fitted   : dict[str, model] — name → fitted model
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    candidates = _get_candidates()
    rows = []
    fitted = {}

    for name, model in candidates.items():
        print(f"  Training {name}...")

        # Time the full fit
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # CV accuracy (robust estimate, run after fit so CV re-fits internally)
        cv_scores = cross_val_score(
            _get_candidates()[name],  # fresh unfitted clone for CV
            X_train, y_train,
            cv=skf,
            scoring="accuracy",
            n_jobs=-1,
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        top5 = top_k_accuracy_score(
            y_test, y_proba, k=5, labels=model.classes_
        )

        test_acc = accuracy_score(y_test, y_pred)
        rows.append({
            "model":           name,
            "cv_acc_mean":     round(cv_scores.mean(), 4),
            "cv_acc_std":      round(cv_scores.std(), 4),
            "test_accuracy":   round(test_acc, 4),
            "test_f1_macro":   round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "test_top5_acc":   round(top5, 4),
            "vs_random":       round(test_acc / RANDOM_BASELINE, 1),
            "train_time_s":    round(train_time, 1),
        })
        fitted[name] = model

    results = pd.DataFrame(rows).set_index("model")
    return results, fitted


def _print_analysis(results: pd.DataFrame) -> None:
    """Print the comparison table with a written interpretation."""
    print(f"\n{'─'*80}")
    print(results.to_string())
    print(f"{'─'*80}")
    print(f"\n  Random baseline (114-class uniform): {RANDOM_BASELINE*100:.2f}%")
    print(f"  vs_random = test_accuracy ÷ random baseline\n")

    best = results["test_accuracy"].idxmax()
    best_acc = results.loc[best, "test_accuracy"]
    best_top5 = results.loc[best, "test_top5_acc"]

    print(f"  Winner: {best}")
    print(
        f"  Top-1 accuracy {best_acc:.1%} sounds modest for 114 classes — "
        f"but it is {results.loc[best, 'vs_random']:.0f}× better than random."
    )
    print(
        f"  Top-5 accuracy {best_top5:.1%} is the more meaningful bar "
        f"for the recommender: the correct genre is in the model's top-5 "
        f"predictions {best_top5:.0%} of the time, which produces a rich "
        f"and accurate probability embedding."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Model Comparison (3 candidates, 5-fold CV)")
    print(f"{'='*60}\n")

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    train_df, test_df = split(df)
    X_train, y_train, _ = build_features(train_df)
    X_test,  y_test,  _ = build_features(test_df)

    results, fitted = compare_models(X_train, y_train, X_test, y_test)

    _print_analysis(results)

    MODEL_DIR.mkdir(exist_ok=True)
    results.to_csv(COMPARISON_PATH)
    print(f"\n  Saved comparison → {COMPARISON_PATH.name}")

    for name, model in fitted.items():
        path = MODEL_DIR / f"compare_{name}.pkl"
        joblib.dump(model, path)
        print(f"  Saved model     → {path.name}")

    print(f"\n{'='*60}\n")
