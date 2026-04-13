"""
LightGBM hyperparameter tuning with Optuna.

Goes beyond the tutorial's "30 trials" in three ways:

  1. 50 trials with a richer search space (8 hyperparameters vs. 3-4)
     Covers regularisation (reg_alpha, reg_lambda), tree structure
     (num_leaves, min_child_samples), and sampling (subsample,
     colsample_bytree) — the parameters that matter most for
     generalisation on a 114-class tabular problem.

  2. Trial history saved to CSV alongside best_params.json
     So you can plot the optimisation curve and see how accuracy
     evolved across trials — useful for deciding whether more trials
     would help.

  3. Improvement report
     Prints how much the tuned model gains over the default-params
     classifier, both in top-1 accuracy and top-5 accuracy.

Optuna uses TPE (Tree Parzen Estimator) by default — a Bayesian
sampler that builds a probabilistic model of which hyperparameter
regions produce good results. It is more sample-efficient than
random search and much faster than grid search.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features.engineer import build_features, split

FEATURES_PATH   = Path(__file__).resolve().parents[2] / "data"   / "features.csv"
MODEL_DIR       = Path(__file__).resolve().parents[2] / "models"
PARAMS_PATH     = MODEL_DIR / "best_params.json"
HISTORY_PATH    = MODEL_DIR / "trial_history.csv"
TUNED_PATH      = MODEL_DIR / "tuned_model.pkl"

N_TRIALS = 50
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_train, y_train, cv_folds):
    """
    Return an Optuna objective function closed over the training data.

    The search space covers:
      - Tree architecture  : num_leaves, min_child_samples
      - Learning dynamics  : learning_rate, n_estimators
      - Feature/row sampling: subsample, colsample_bytree
      - Regularisation     : reg_alpha, reg_lambda
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        model = lgb.LGBMClassifier(
            **params,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf,
            scoring="accuracy",
            n_jobs=-1,
        )
        return scores.mean()

    return objective


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = N_TRIALS,
    cv_folds: int = CV_FOLDS,
) -> dict:
    """
    Run Optuna hyperparameter search for LightGBM.

    Suppresses per-trial output — prints a progress update every 10 trials
    instead. Returns the best hyperparameter dict.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = _make_objective(X_train, y_train, cv_folds)

    # Callback: print brief progress every 10 trials
    def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if (trial.number + 1) % 10 == 0:
            print(
                f"  Trial {trial.number + 1:>3}/{n_trials}  "
                f"best so far: {study.best_value:.4f}"
            )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )

    return study.best_params, study


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def train_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> lgb.LGBMClassifier:
    """Train final model on the full training set with best params."""
    model = lgb.LGBMClassifier(
        **best_params,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_tuned(
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    default_accuracy: float | None = None,
) -> dict:
    """
    Evaluate tuned model and optionally report improvement over default.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=model.classes_)

    metrics = {
        "accuracy":      round(acc, 4),
        "f1_macro":      round(f1, 4),
        "top5_accuracy": round(top5, 4),
    }

    if default_accuracy is not None:
        delta = acc - default_accuracy
        metrics["improvement_vs_default"] = round(delta, 4)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  LightGBM Hyperparameter Tuning (Optuna, {N_TRIALS} trials)")
    print(f"{'='*60}\n")

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    train_df, test_df = split(df)
    X_train, y_train, _ = build_features(train_df)
    X_test,  y_test,  _ = build_features(test_df)

    # Default-params reference score
    print("  Training default LightGBM for reference...")
    default_model = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=127, learning_rate=0.05,
        n_jobs=-1, random_state=42, verbose=-1
    )
    default_model.fit(X_train, y_train)
    default_acc = accuracy_score(y_test, default_model.predict(X_test))
    print(f"  Default test accuracy: {default_acc:.4f}\n")

    # Optuna search
    print(f"  Running {N_TRIALS} Optuna trials ({CV_FOLDS}-fold CV)...")
    t0 = time.time()
    best_params, study = tune_lightgbm(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  Search complete in {elapsed:.0f}s")
    print(f"  Best CV accuracy : {study.best_value:.4f}")
    print(f"  Best params      : {best_params}")

    # Save artefacts
    MODEL_DIR.mkdir(exist_ok=True)

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Saved best params → {PARAMS_PATH.name}")

    history = study.trials_dataframe()[["number", "value", "state"]]
    history.columns = ["trial", "cv_accuracy", "state"]
    history.to_csv(HISTORY_PATH, index=False)
    print(f"  Saved trial history → {HISTORY_PATH.name}")

    # Train and evaluate final model
    print("\n  Training final model with best params...")
    tuned_model = train_tuned(X_train, y_train, best_params)
    metrics = evaluate_tuned(tuned_model, X_test, y_test, default_accuracy=default_acc)

    print("\n  Final model metrics:")
    print(f"    Accuracy (top-1) : {metrics['accuracy']:.4f}")
    print(f"    F1 macro         : {metrics['f1_macro']:.4f}")
    print(f"    Accuracy (top-5) : {metrics['top5_accuracy']:.4f}")
    if "improvement_vs_default" in metrics:
        delta = metrics["improvement_vs_default"]
        sign = "+" if delta >= 0 else ""
        print(f"    vs default       : {sign}{delta:.4f} ({sign}{delta*100:.2f}pp)")

    joblib.dump(tuned_model, TUNED_PATH)
    print(f"\n  Saved tuned model → {TUNED_PATH.name}")
    print(f"\n{'='*60}\n")
