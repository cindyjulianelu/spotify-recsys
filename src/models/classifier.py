"""
Genre classifier — Stage 1 of the two-stage recommendation pipeline.

Trains a LightGBM multi-class classifier to predict track_genre from
audio features. The key output is not just the predicted label but the
full 114-dimensional probability vector from predict_proba(), which
serves as a learned embedding for Stage 2 (recommendation).

Why LightGBM:
- Handles mixed numeric/discrete features without scaling
- Fast training on ~90k rows × 15 features
- predict_proba() produces well-calibrated probabilities at scale

Why use probabilities as embeddings:
- A hard genre label discards confidence information
- A track that scores 40% jazz / 30% blues / 15% soul is described
  more richly than just "jazz"
- Two tracks with similar probability distributions are genuinely
  musically similar, even if they carry different hard labels
"""

import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    num_leaves: int = 127,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM genre classifier.

    Parameters
    ----------
    X_train, y_train : feature matrix and genre label array
    n_estimators     : number of boosting rounds
    num_leaves       : max leaves per tree (controls model complexity)
    learning_rate    : shrinkage rate
    random_state     : reproducibility seed

    Returns
    -------
    Fitted LGBMClassifier
    """
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the classifier on a held-out test set.

    Returns a dict with three metrics:
      accuracy      — fraction of exact genre matches
      f1_macro      — unweighted mean F1 across all 114 genres
                      (penalises the model equally for each genre
                      regardless of how many tracks it has)
      top5_accuracy — fraction where the correct genre appears in
                      the model's top-5 predictions; useful signal
                      for a 114-class problem where "jazz adjacent
                      to blues" is not the same as a true error
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy":      round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro":      round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "top5_accuracy": round(float(top_k_accuracy_score(y_test, y_proba, k=5, labels=model.classes_)), 4),
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_embeddings(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return the genre probability matrix: shape (n_tracks, n_genres).

    Each row is a probability distribution over all 114 genres.
    This is the embedding passed to the recommendation stage — it
    encodes not just "what genre is this?" but "how genre-like is
    this track across every genre in the dataset?".
    """
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(model: lgb.LGBMClassifier, name: str = "classifier.pkl") -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path


def load(name: str = "classifier.pkl") -> lgb.LGBMClassifier:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"No saved classifier at {path}. Run train.py first."
        )
    return joblib.load(path)
