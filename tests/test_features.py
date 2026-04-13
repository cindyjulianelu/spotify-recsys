"""
Tests for src/features/engineer.py

Covers build_features() and split() using synthetic DataFrames —
no dependency on cleaned.csv, runs cleanly in CI.
"""

import numpy as np
import pandas as pd
import pytest

from features.engineer import FEATURE_COLS, build_features, split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_per_class: int = 50, n_classes: int = 4, seed: int = 42) -> pd.DataFrame:
    """Return a minimal valid cleaned DataFrame for feature engineering tests."""
    rng    = np.random.default_rng(seed)
    n      = n_per_class * n_classes
    genres = [f"genre_{i}" for i in range(n_classes)]

    return pd.DataFrame(
        {
            "track_id":         [f"id_{i}" for i in range(n)],
            "artists":          [f"Artist {i % 20}" for i in range(n)],
            "album_name":       [f"Album {i % 30}" for i in range(n)],
            "track_name":       [f"Track {i}" for i in range(n)],
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


# ---------------------------------------------------------------------------
# Tests — build_features
# ---------------------------------------------------------------------------

def test_build_features_returns_three_items():
    df = _make_df()
    result = build_features(df)
    assert len(result) == 3, "build_features should return (X, y, feature_cols)"


def test_build_features_X_shape():
    """X should have shape (n_rows, n_features) with the canonical 15 features."""
    df       = _make_df(n_per_class=40, n_classes=3)
    X, y, cols = build_features(df)

    assert X.shape[0] == len(df), "X row count should match DataFrame row count"
    assert X.shape[1] == len(FEATURE_COLS), (
        f"Expected {len(FEATURE_COLS)} features, got {X.shape[1]}"
    )


def test_build_features_y_length():
    df       = _make_df()
    X, y, _  = build_features(df)
    assert len(y) == len(df)


def test_build_features_no_nan():
    """Feature matrix must contain no NaN values."""
    df       = _make_df()
    X, y, _  = build_features(df)

    assert not np.isnan(X).any(), "X contains NaN values"
    assert not pd.isnull(y).any(), "y contains NaN values"


def test_build_features_feature_names():
    """Returned feature_cols must exactly match FEATURE_COLS in order."""
    df         = _make_df()
    _, _, cols = build_features(df)

    assert cols == FEATURE_COLS


def test_build_features_explicit_cast_to_int():
    """
    explicit is stored as bool in the cleaned DataFrame.
    build_features must cast it to int (0/1) before returning.
    """
    df       = _make_df()
    X, _, _  = build_features(df)

    explicit_idx = FEATURE_COLS.index("explicit")
    col_vals     = X[:, explicit_idx]

    assert set(col_vals.astype(int).tolist()).issubset({0, 1}), (
        "explicit column should contain only 0 or 1 after casting"
    )


def test_zero_one_features_in_valid_range():
    """All 0–1 audio features (danceability, energy, etc.) must stay in [0, 1]."""
    df       = _make_df()
    X, _, _  = build_features(df)

    unit_features = [
        "danceability", "energy", "valence", "speechiness",
        "acousticness", "instrumentalness", "liveness",
    ]
    for feat in unit_features:
        idx = FEATURE_COLS.index(feat)
        col = X[:, idx]
        assert col.min() >= 0.0, f"{feat} has values below 0"
        assert col.max() <= 1.0, f"{feat} has values above 1"


def test_build_features_x_dtype_float64():
    """X must be float64 (required by LightGBM)."""
    df      = _make_df()
    X, _, _ = build_features(df)
    assert X.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests — split
# ---------------------------------------------------------------------------

def test_split_returns_two_dataframes():
    df       = _make_df()
    train_df, test_df = split(df)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df,  pd.DataFrame)


def test_split_sizes():
    """Default test_size=0.2 should give roughly 80/20 split."""
    df               = _make_df(n_per_class=100, n_classes=4)
    train_df, test_df = split(df)

    total    = len(df)
    expected_test = int(total * 0.2)
    # Allow ±5% tolerance for stratification rounding
    assert abs(len(test_df) - expected_test) <= total * 0.05


def test_split_no_overlap():
    """Train and test sets must be disjoint (no shared rows)."""
    df               = _make_df()
    train_df, test_df = split(df)

    # Check by row index — there should be no common indices
    common = set(train_df.index) & set(test_df.index)
    assert len(common) == 0, f"Found {len(common)} overlapping rows between train and test"


def test_split_preserves_columns():
    """Split should not add or remove columns."""
    df               = _make_df()
    train_df, test_df = split(df)

    assert set(train_df.columns) == set(df.columns)
    assert set(test_df.columns)  == set(df.columns)


def test_split_stratification():
    """
    Stratified split should preserve genre proportions.
    Each genre should appear in both train and test.
    """
    df               = _make_df(n_per_class=50, n_classes=4)
    train_df, test_df = split(df)

    train_genres = set(train_df["track_genre"].unique())
    test_genres  = set(test_df["track_genre"].unique())

    assert train_genres == test_genres, (
        "All genres should appear in both train and test (stratified split)"
    )
