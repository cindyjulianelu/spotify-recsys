"""
MLflow experiment tracking — full training pipeline.

Logs four runs under the "spotify-genre-classification" experiment:
  1. dummy-baseline        — DummyClassifier (most_frequent)
  2. logreg-baseline       — LogisticRegression (best linear model)
  3. lightgbm-default      — LightGBM with sensible defaults
  4. lightgbm-tuned        — LightGBM with Optuna-tuned hyperparameters

MLflow is configured to use FILE-BASED tracking (no server required).
Runs are stored in mlruns/ at the project root and viewable at any time
with:

    mlflow ui --port 5001

Why file-based and not the tutorial's `mlflow server`:
  - No port conflicts (common problem with the default 5000)
  - No background process to manage
  - mlruns/ is gitignored — experiment history stays local
  - `mlflow ui` is lighter than `mlflow server` and sufficient for
    local development

What is logged per run:
  - params : model name, all hyperparameters
  - metrics: accuracy, f1_macro, top5_accuracy, train_time_s,
             vs_random_multiplier
  - tags   : model_type, stage
  - artifact: feature_importance.csv (LightGBM runs only — the one
              artefact actually worth logging; pkl files are already
              in models/ and don't need duplicating inside mlruns/)

The winning model (lightgbm-tuned) is also saved as
models/production_model.pkl for the recommendation pipeline.
"""

import sys
import json
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import mlflow
import pandas as pd
import lightgbm as lgb

from features.engineer import build_features, split
from baseline import train_baselines, evaluate_baselines
from compare  import compare_models
from tuning   import tune_lightgbm, train_tuned, evaluate_tuned, N_TRIALS

FEATURES_PATH    = Path(__file__).resolve().parents[2] / "data"   / "features.csv"
MODEL_DIR        = Path(__file__).resolve().parents[2] / "models"
MLRUNS_DIR       = Path(__file__).resolve().parents[2] / "mlruns"
PRODUCTION_PATH  = MODEL_DIR / "production_model.pkl"
PARAMS_PATH      = MODEL_DIR / "best_params.json"

RANDOM_BASELINE  = 1.0 / 114
EXPERIMENT_NAME  = "spotify-genre-classification"


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _log_feature_importance(model: lgb.LGBMClassifier, feature_names: list[str]) -> None:
    """Log feature importance as a CSV artifact (lightweight, informative)."""
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False, prefix="feature_importance_"
    ) as f:
        importance.to_csv(f, index=False)
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="feature_importance")
    Path(tmp_path).unlink(missing_ok=True)


def _log_run(
    run_name: str,
    model_type: str,
    params: dict,
    metrics: dict,
    model=None,
    feature_names: list[str] | None = None,
    stage: str = "experiment",
) -> None:
    """Start an MLflow run and log params, metrics, and optional artefacts."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("stage", stage)

        mlflow.log_params(params)
        mlflow.log_metrics({
            **metrics,
            "random_baseline":    round(RANDOM_BASELINE, 6),
            "vs_random":          round(metrics.get("accuracy", 0) / RANDOM_BASELINE, 2),
        })

        if model is not None and feature_names is not None and isinstance(model, lgb.LGBMClassifier):
            _log_feature_importance(model, feature_names)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"\n{'='*60}")
    print("  MLflow Experiment: spotify-genre-classification")
    print(f"{'='*60}\n")

    # File-based tracking — no server, no port conflicts
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)

    MODEL_DIR.mkdir(exist_ok=True)

    # ---- Load and split ----
    print("[1/5] Loading features.csv...")
    df = pd.read_csv(FEATURES_PATH, index_col=0)
    train_df, test_df = split(df)
    X_train, y_train, feature_names = build_features(train_df)
    X_test,  y_test,  _             = build_features(test_df)
    print(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}  Features: {len(feature_names)}")

    # ---- Baseline runs ----
    print("\n[2/5] Training and logging baselines...")
    t0 = time.time()
    baselines = train_baselines(X_train, y_train)
    baseline_results = evaluate_baselines(baselines, X_train, y_train, X_test, y_test, cv=5)

    for name, row in baseline_results.iterrows():
        _log_run(
            run_name   = f"{name}-baseline",
            model_type = name,
            params     = {"model": name, "strategy": "most_frequent" if name == "dummy" else "multinomial"},
            metrics    = {
                "accuracy":      float(row["test_accuracy"]),
                "f1_macro":      float(row["test_f1_macro"]),
                "top5_accuracy": float(row["test_top5_acc"]),
                "train_time_s":  round(time.time() - t0, 1),
            },
            stage = "baseline",
        )
        print(f"  Logged {name}: accuracy={row['test_accuracy']}")

    # ---- Model comparison ----
    print("\n[3/5] Running model comparison (3 candidates)...")
    comparison, fitted_models = compare_models(X_train, y_train, X_test, y_test, cv=5)

    for name, row in comparison.iterrows():
        model = fitted_models[name]
        model_params = {"model": name}
        if hasattr(model, "get_params"):
            core_params = {k: v for k, v in model.get_params().items()
                          if k in ("num_leaves", "learning_rate", "n_estimators",
                                   "n_estimators", "max_depth", "C")}
            model_params.update(core_params)

        _log_run(
            run_name    = f"compare-{name}",
            model_type  = name,
            params      = model_params,
            metrics     = {
                "accuracy":       float(row["test_accuracy"]),
                "f1_macro":       float(row["test_f1_macro"]),
                "top5_accuracy":  float(row["test_top5_acc"]),
                "cv_acc_mean":    float(row["cv_acc_mean"]),
                "cv_acc_std":     float(row["cv_acc_std"]),
                "train_time_s":   float(row["train_time_s"]),
            },
            model        = model if isinstance(model, lgb.LGBMClassifier) else None,
            feature_names= feature_names if isinstance(model, lgb.LGBMClassifier) else None,
            stage        = "comparison",
        )
        print(f"  Logged compare-{name}: accuracy={row['test_accuracy']}")

    # ---- Optuna tuning ----
    print(f"\n[4/5] Optuna tuning ({N_TRIALS} trials)...")
    best_params, study = tune_lightgbm(X_train, y_train)

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n  Best CV accuracy: {study.best_value:.4f}")

    tuned_model = train_tuned(X_train, y_train, best_params)
    default_acc = comparison.loc["lightgbm", "test_accuracy"] if "lightgbm" in comparison.index else None
    tuned_metrics = evaluate_tuned(tuned_model, X_test, y_test, default_accuracy=default_acc)

    _log_run(
        run_name     = "lightgbm-tuned",
        model_type   = "lightgbm",
        params       = {**best_params, "model": "lightgbm-tuned", "n_optuna_trials": N_TRIALS},
        metrics      = {**tuned_metrics, "optuna_best_cv": round(study.best_value, 4)},
        model        = tuned_model,
        feature_names= feature_names,
        stage        = "production",
    )
    print(f"  Logged lightgbm-tuned: accuracy={tuned_metrics['accuracy']}")

    # ---- Save production model ----
    print("\n[5/5] Saving production model...")
    joblib.dump(tuned_model, PRODUCTION_PATH)
    print(f"  Saved → {PRODUCTION_PATH.name}")

    # ---- Summary ----
    _print_summary(baseline_results, comparison, tuned_metrics)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_summary(
    baseline_results: pd.DataFrame,
    comparison: pd.DataFrame,
    tuned_metrics: dict,
) -> None:
    print(f"\n{'='*60}")
    print("  Experiment Summary")
    print(f"{'='*60}")

    print(f"\n  Random baseline       : {RANDOM_BASELINE*100:.2f}%")

    if "dummy" in baseline_results.index:
        print(f"  DummyClassifier       : {baseline_results.loc['dummy', 'test_accuracy']*100:.2f}%")
    if "logreg" in baseline_results.index:
        print(f"  LogisticRegression    : {baseline_results.loc['logreg', 'test_accuracy']*100:.2f}%")
    if "random_forest" in comparison.index:
        print(f"  RandomForest          : {comparison.loc['random_forest', 'test_accuracy']*100:.2f}%")
    if "lightgbm" in comparison.index:
        print(f"  LightGBM (default)    : {comparison.loc['lightgbm', 'test_accuracy']*100:.2f}%")

    print(f"  LightGBM (tuned)  ★   : {tuned_metrics['accuracy']*100:.2f}%")
    print(f"    F1 macro             : {tuned_metrics['f1_macro']:.4f}")
    print(f"    Top-5 accuracy       : {tuned_metrics['top5_accuracy']*100:.2f}%")

    if "improvement_vs_default" in tuned_metrics:
        delta = tuned_metrics["improvement_vs_default"]
        sign = "+" if delta >= 0 else ""
        print(f"    Tuning gain          : {sign}{delta*100:.2f}pp over default LightGBM")

    print("\n  View all runs:")
    print("    mlflow ui --port 5001")
    print("    → open http://localhost:5001")
    print("\n  Production model saved to: models/production_model.pkl")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run()
