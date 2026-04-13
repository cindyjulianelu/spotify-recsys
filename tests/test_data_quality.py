"""
Tests for src/data/quality.py

All tests use synthetic DataFrames — no dependency on data files,
so the suite runs cleanly in CI without cleaned.csv present.
"""

import numpy as np
import pandas as pd
import pytest

from quality import check_data_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(n_per_class: int = 100, n_classes: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Return a minimal valid DataFrame that should pass all quality checks.

    Uses realistic value ranges derived from the Spotify API spec so that
    every numeric bounds check clears. Balanced classes so the imbalance
    threshold is not triggered.
    """
    rng  = np.random.default_rng(seed)
    n    = n_per_class * n_classes
    genres = [f"genre_{i}" for i in range(n_classes)]

    return pd.DataFrame(
        {
            "track_id":        [f"id_{i}" for i in range(n)],
            "artists":         [f"Artist {i % 30}" for i in range(n)],
            "album_name":      [f"Album {i % 50}" for i in range(n)],
            "track_name":      [f"Track {i}" for i in range(n)],
            "popularity":      rng.integers(0, 101, n).astype("int64"),
            "duration_ms":     rng.integers(60_000, 600_000, n).astype("int64"),
            "explicit":        rng.choice([True, False], n),
            "danceability":    rng.uniform(0.0, 1.0, n).astype("float64"),
            "energy":          rng.uniform(0.0, 1.0, n).astype("float64"),
            "key":             rng.integers(0, 12, n).astype("int64"),
            "loudness":        rng.uniform(-30.0, 0.0, n).astype("float64"),
            "mode":            rng.integers(0, 2, n).astype("int64"),
            "speechiness":     rng.uniform(0.0, 1.0, n).astype("float64"),
            "acousticness":    rng.uniform(0.0, 1.0, n).astype("float64"),
            "instrumentalness":rng.uniform(0.0, 1.0, n).astype("float64"),
            "liveness":        rng.uniform(0.0, 1.0, n).astype("float64"),
            "valence":         rng.uniform(0.0, 1.0, n).astype("float64"),
            "tempo":           rng.uniform(60.0, 180.0, n).astype("float64"),
            "time_signature":  rng.choice([3, 4, 5], n).astype("int64"),
            "track_genre":     np.repeat(genres, n_per_class),
        }
    )


# ---------------------------------------------------------------------------
# Tests — passes on valid data
# ---------------------------------------------------------------------------

def test_valid_df_passes_quality_gate():
    """A well-formed DataFrame should clear the quality gate (success=True)."""
    df     = _make_valid_df()
    result = check_data_quality(df, target_col="track_genre")

    assert result["success"] is True, (
        f"Quality gate failed unexpectedly.\nFailures: {result['failures']}"
    )


def test_result_structure():
    """The returned dict must have the four expected top-level keys."""
    df     = _make_valid_df()
    result = check_data_quality(df)

    assert "success"    in result
    assert "failures"   in result
    assert "warnings"   in result
    assert "statistics" in result


def test_statistics_contain_row_count():
    """statistics.total_rows should equal len(df)."""
    df     = _make_valid_df(n_per_class=50)
    result = check_data_quality(df)

    assert result["statistics"]["total_rows"] == len(df)


# ---------------------------------------------------------------------------
# Tests — catches broken data
# ---------------------------------------------------------------------------

def test_missing_required_column_is_failure():
    """Dropping a required column must produce a schema failure."""
    df     = _make_valid_df().drop(columns=["danceability"])
    result = check_data_quality(df)

    assert result["success"] is False
    assert any("danceability" in msg for msg in result["failures"])


def test_too_few_rows_is_failure():
    """A DataFrame with fewer than 100 rows should fail the row-count check."""
    df     = _make_valid_df(n_per_class=10, n_classes=2)   # 20 rows
    result = check_data_quality(df)

    assert result["success"] is False
    assert any("row_count" in msg for msg in result["failures"])


def test_duration_zero_is_failure():
    """duration_ms = 0 is a critical error (track has no audio)."""
    df               = _make_valid_df()
    df.loc[0, "duration_ms"] = 0
    result           = check_data_quality(df)

    assert result["success"] is False
    assert any("duration" in msg for msg in result["failures"])


def test_time_signature_zero_is_failure():
    """time_signature = 0 (meter undetected) is a critical error."""
    df                        = _make_valid_df()
    df.loc[0, "time_signature"] = 0
    result                    = check_data_quality(df)

    assert result["success"] is False
    assert any("time_signature" in msg for msg in result["failures"])


def test_single_class_target_is_failure():
    """A target column with only one class should fail."""
    df               = _make_valid_df()
    df["track_genre"] = "only_genre"
    result           = check_data_quality(df, target_col="track_genre")

    assert result["success"] is False
    assert any("target" in msg for msg in result["failures"])


def test_missing_target_column_is_failure():
    """A DataFrame missing the target column entirely should fail."""
    df     = _make_valid_df().drop(columns=["track_genre"])
    result = check_data_quality(df, target_col="track_genre")

    assert result["success"] is False


# ---------------------------------------------------------------------------
# Tests — warnings (not failures)
# ---------------------------------------------------------------------------

def test_tempo_above_200_raises_warning():
    """
    Tempo > 200 BPM triggers a musical-metadata warning (not a failure).
    The correct fix is halving, not dropping — so it's a soft signal.
    """
    df                   = _make_valid_df()
    df.loc[0, "tempo"]   = 240.0
    result               = check_data_quality(df)

    assert any("tempo" in msg for msg in result["warnings"])


def test_long_duration_raises_warning():
    """duration_ms > 3,600,000 (1 hr) is a warning, not a failure."""
    df                        = _make_valid_df()
    df.loc[0, "duration_ms"]  = 4_000_000
    result                    = check_data_quality(df)

    assert any("duration" in msg for msg in result["warnings"])
