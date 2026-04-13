import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "track_genre"

# All audio features fed to the classifier.
# LightGBM is tree-based and does not require scaling, so we pass
# everything as-is. Explicit is cast to int (True/False → 1/0).
# Key and time_signature are treated as ordinal integers — adequate
# for tree splits even though the underlying musical relationships
# are not fully ordinal.
FEATURE_COLS = [
    # Continuous 0–1
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    # Continuous, other ranges
    "loudness",
    "tempo",
    "popularity",
    "duration_ms",
    # Discrete musical structure
    "key",
    "mode",
    "time_signature",
    # Binary
    "explicit",
]


def build_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract the feature matrix and label array from a cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of cleaner.clean_data).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — genre strings
    feature_cols : list[str]            — column names matching X
    """
    df = df.copy()
    df["explicit"] = df["explicit"].astype(int)

    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET].values

    return X, y, FEATURE_COLS


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split, keeping rows as DataFrames so that
    track metadata (track_id, track_name, artists) is available
    for both training and later index construction.

    Parameters
    ----------
    df          : cleaned dataframe
    test_size   : fraction held out for evaluation
    random_state: reproducibility seed

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
