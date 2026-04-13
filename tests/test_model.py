"""
Tests for src/models/classifier.py

Trains a tiny model (5 genres × 30 samples, n_estimators=5) to keep the
test suite fast — the goal is interface correctness, not accuracy.

No dependency on trained pickle files (models/ is gitignored).
"""

import numpy as np
import pytest

from features.engineer import FEATURE_COLS, build_features, split
from classifier import evaluate, get_embeddings, load, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arrays(n_per_class: int = 30, n_classes: int = 5, seed: int = 0):
    """
    Return (X_train, X_test, y_train, y_test) with the full 15-feature schema.
    Uses n_estimators=5 in training so tests run in < 1 s.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    rng    = np.random.default_rng(seed)
    n      = n_per_class * n_classes
    genres = [f"genre_{i}" for i in range(n_classes)]

    df = pd.DataFrame(
        {
            "track_id":         [f"id_{i}" for i in range(n)],
            "artists":          [f"A{i}" for i in range(n)],
            "album_name":       [f"Alb{i}" for i in range(n)],
            "track_name":       [f"T{i}" for i in range(n)],
            "popularity":       rng.integers(0, 101, n),
            "duration_ms":      rng.integers(60_000, 600_000, n),
            "explicit":         rng.choice([True, False], n),
            "danceability":     rng.uniform(0, 1, n),
            "energy":           rng.uniform(0, 1, n),
            "key":              rng.integers(0, 12, n),
            "loudness":         rng.uniform(-30, 0, n),
            "mode":             rng.integers(0, 2, n),
            "speechiness":      rng.uniform(0, 1, n),
            "acousticness":     rng.uniform(0, 1, n),
            "instrumentalness": rng.uniform(0, 1, n),
            "liveness":         rng.uniform(0, 1, n),
            "valence":          rng.uniform(0, 1, n),
            "tempo":            rng.uniform(60, 180, n),
            "time_signature":   rng.choice([3, 4, 5], n),
            "track_genre":      np.repeat(genres, n_per_class),
        }
    )
    X, y, _ = build_features(df)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def tiny_model():
    """A model trained on 5 genres × 30 samples. Reused across tests in this module."""
    X_train, _, y_train, _ = _make_arrays()
    return train(X_train, y_train, n_estimators=5, num_leaves=8, learning_rate=0.1)


@pytest.fixture(scope="module")
def tiny_split():
    return _make_arrays()


# ---------------------------------------------------------------------------
# Tests — train()
# ---------------------------------------------------------------------------

def test_train_returns_fitted_model(tiny_model):
    """train() should return a fitted LGBMClassifier."""
    import lightgbm as lgb
    assert isinstance(tiny_model, lgb.LGBMClassifier)
    assert hasattr(tiny_model, "classes_"), "Model should expose .classes_ after fit"


def test_model_classes_match_training_labels(tiny_split, tiny_model):
    """model.classes_ should contain exactly the labels present in y_train."""
    _, _, y_train, _ = tiny_split
    assert set(tiny_model.classes_) == set(y_train), (
        "model.classes_ should be the unique training labels"
    )


# ---------------------------------------------------------------------------
# Tests — predict / predict_proba
# ---------------------------------------------------------------------------

def test_predict_returns_correct_length(tiny_split, tiny_model):
    _, X_test, _, _ = tiny_split
    y_pred = tiny_model.predict(X_test)
    assert len(y_pred) == len(X_test)


def test_predict_labels_are_known_classes(tiny_split, tiny_model):
    """All predicted labels must be classes the model was trained on."""
    _, X_test, _, _ = tiny_split
    y_pred = tiny_model.predict(X_test)
    unknown = set(y_pred) - set(tiny_model.classes_)
    assert len(unknown) == 0, f"Predicted unknown classes: {unknown}"


def test_predict_proba_shape(tiny_split, tiny_model):
    """predict_proba should return (n_samples, n_classes)."""
    _, X_test, _, _ = tiny_split
    proba = tiny_model.predict_proba(X_test)

    assert proba.shape[0] == len(X_test)
    assert proba.shape[1] == len(tiny_model.classes_)


def test_predict_proba_sums_to_one(tiny_split, tiny_model):
    """Each row of predict_proba must sum to ~1.0 (probability simplex)."""
    _, X_test, _, _ = tiny_split
    proba = tiny_model.predict_proba(X_test)
    row_sums = proba.sum(axis=1)

    np.testing.assert_allclose(
        row_sums, np.ones(len(X_test)), atol=1e-5,
        err_msg="predict_proba row sums should equal 1.0",
    )


def test_predict_proba_non_negative(tiny_split, tiny_model):
    """All probabilities must be >= 0."""
    _, X_test, _, _ = tiny_split
    proba = tiny_model.predict_proba(X_test)
    assert (proba >= 0).all(), "Found negative probabilities"


# ---------------------------------------------------------------------------
# Tests — evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_returns_expected_keys(tiny_split, tiny_model):
    """evaluate() must return all three metric keys."""
    _, X_test, _, y_test = tiny_split
    metrics = evaluate(tiny_model, X_test, y_test)

    assert "accuracy"      in metrics
    assert "f1_macro"      in metrics
    assert "top5_accuracy" in metrics


def test_evaluate_accuracy_in_range(tiny_split, tiny_model):
    """Accuracy must be in [0, 1]."""
    _, X_test, _, y_test = tiny_split
    metrics = evaluate(tiny_model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_f1_in_range(tiny_split, tiny_model):
    _, X_test, _, y_test = tiny_split
    metrics = evaluate(tiny_model, X_test, y_test)
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_evaluate_top5_in_range(tiny_split, tiny_model):
    _, X_test, _, y_test = tiny_split
    metrics = evaluate(tiny_model, X_test, y_test)
    assert 0.0 <= metrics["top5_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tests — get_embeddings()
# ---------------------------------------------------------------------------

def test_get_embeddings_shape(tiny_split, tiny_model):
    """Embedding matrix should be (n_tracks, n_genres)."""
    _, X_test, _, _ = tiny_split
    embeddings = get_embeddings(tiny_model, X_test)

    assert embeddings.shape[0] == len(X_test)
    assert embeddings.shape[1] == len(tiny_model.classes_)


def test_get_embeddings_are_probabilities(tiny_split, tiny_model):
    """Embeddings are probability vectors — must sum to 1 and be non-negative."""
    _, X_test, _, _ = tiny_split
    embeddings = get_embeddings(tiny_model, X_test)

    assert (embeddings >= 0).all()
    np.testing.assert_allclose(
        embeddings.sum(axis=1), np.ones(len(X_test)), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Tests — load()
# ---------------------------------------------------------------------------

def test_load_raises_when_model_missing(tmp_path, monkeypatch):
    """
    load() should raise FileNotFoundError for a non-existent pkl.
    We monkeypatch MODEL_DIR to a temp directory with no pkl files.
    """
    import classifier as clf_module
    monkeypatch.setattr(clf_module, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load("nonexistent_model.pkl")
