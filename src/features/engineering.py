"""
Feature engineering for the Spotify recommendation system.

Two public functions:
  create_features(df)  — adds 12 new columns across 3 categories
  select_features(df)  — removes redundant and near-zero-variance features

Why engineer features at all for a tree-based model?
LightGBM can in principle discover interactions itself, but explicitly
encoding known musical relationships gives it a head start — especially
useful when the training set is ~90k rows across 114 classes, where
each class has only ~800 training rows on average.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLEANED_PATH  = Path(__file__).resolve().parents[2] / "data" / "cleaned.csv"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "features.csv"

# Columns that are metadata or the target — excluded from numeric operations
META_COLS = {"track_id", "artists", "album_name", "track_name", "track_genre"}


# ---------------------------------------------------------------------------
# Feature creation
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 12 new features from the cleaned Spotify dataset.

    Organised into three categories:

    1. Domain features    — derived from Spotify's audio analysis spec
                           and music theory knowledge
    2. Statistical features — composite scores summarising related dimensions
                           (note: no rolling windows — rows are independent
                           tracks, not a time series)
    3. Interaction features — products of pairs of features whose joint
                           behaviour carries more signal than either alone

    Parameters
    ----------
    df : pd.DataFrame — output of cleaner.clean_data()

    Returns
    -------
    pd.DataFrame with original columns + 12 new feature columns
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Category 1: Domain features (6 features)
    # ------------------------------------------------------------------

    # Spotify's own documentation defines speechiness > 0.66 as "almost
    # entirely spoken word" (podcasts, audiobooks, slam poetry). Values
    # 0.33–0.66 indicate tracks with both music and speech (rap); below
    # 0.33 is instrumental or sung music. Binarising at 0.66 cleanly
    # separates spoken-word recordings from music.
    df["is_spoken_word"] = (df["speechiness"] > 0.66).astype(np.int8)

    # Spotify's documentation says liveness > 0.8 has a strong probability
    # of being a live recording. Live tracks carry crowd noise, reverb, and
    # dynamic variation that differs acoustically from studio versions —
    # even covers of the same song. A binary flag is more useful here than
    # the continuous value because the distribution is right-skewed and
    # most genre models treat "live" as a categorical property.
    df["is_live_recording"] = (df["liveness"] > 0.8).astype(np.int8)

    # Instrumentalness has a strongly bimodal distribution: tracks either
    # have vocals (value ≈ 0) or they don't (value ≈ 1). The 0.5 threshold
    # sits in the low-density valley between the two modes. Binarising
    # here captures a genuine musical dichotomy and avoids the model
    # trying to interpolate across an essentially empty middle range.
    df["is_instrumental"] = (df["instrumentalness"] > 0.5).astype(np.int8)

    # Same reasoning for acousticness — the distribution is bimodal,
    # reflecting the acoustic/electric divide. Classical, folk, and
    # singer-songwriter genres cluster near 1; electronic and hip-hop
    # near 0. The gap in the middle is real: very few tracks sit at 0.5.
    df["is_acoustic"] = (df["acousticness"] > 0.5).astype(np.int8)

    # Loudness is the only audio feature in dB rather than [0, 1].
    # Mapping [−60, 0] → [0, 1] puts it on the same scale as the
    # other continuous features. Useful if the feature vector is ever
    # fed to a distance-based model (KNN, cosine similarity) where
    # scale disparity would distort distances.
    # NOTE: this will be removed by the correlation filter in
    # select_features() because it is a perfect linear transform of
    # loudness (r = 1.0). It is included here to demonstrate that the
    # filter works correctly.
    df["loudness_norm"] = (df["loudness"] + 60.0) / 60.0

    # After the octave-doubling correction in cleaning, tempo values sit
    # in roughly [30, 200] BPM. Dividing by 200 maps to [0, 1], which
    # makes tempo directly comparable to danceability and energy when
    # computing interaction features below.
    # Same note as loudness_norm: r = 1.0 with tempo, will be removed
    # by the correlation filter — preserved here as a building block
    # for the interaction features.
    df["tempo_norm"] = df["tempo"] / 200.0

    # ------------------------------------------------------------------
    # Category 2: Statistical features (2 features)
    # ------------------------------------------------------------------

    # Average of the three features most predictive of "energetic
    # experience". A danceable, energetic, happy track scores high;
    # a slow, quiet, melancholic track scores low. Useful as a
    # single "vibe intensity" axis and as a faster decision tree split
    # than checking each of the three independently.
    df["audio_brightness"] = (
        df["energy"] + df["valence"] + df["danceability"]
    ) / 3.0

    # Tracks high in speechiness, liveness, or instrumentalness deviate
    # from the "standard studio song" template. This composite is a proxy
    # for "how atypical is this track's audio character?" Genres like
    # stand-up comedy, live jazz, and ambient music all score high for
    # different reasons, but the composite usefully groups them.
    df["audio_atypicality"] = (
        df["speechiness"] + df["liveness"] + df["instrumentalness"]
    )

    # ------------------------------------------------------------------
    # Category 3: Interaction features (4 features)
    # ------------------------------------------------------------------

    # Danceability × energy: captures "club banger" tracks that are high
    # on both axes. A slow R&B ballad can be danceable but has low energy;
    # a fast thrash-metal track has high energy but low danceability.
    # Only tracks that score high on BOTH are genuine dance-floor material.
    df["danceability_x_energy"] = df["danceability"] * df["energy"]

    # Valence × energy: the clearest two-axis mood signal in the dataset.
    #   High valence + high energy → euphoric (pop anthem, dance)
    #   High valence + low energy  → gentle/content (ambient, folk)
    #   Low valence  + high energy → intense/aggressive (metal, punk)
    #   Low valence  + low energy  → melancholic (sad indie, classical minor)
    # The product collapses this to "joyful intensity" — a single axis
    # that separates the four quadrants more efficiently than using
    # valence and energy as independent features.
    df["valence_x_energy"] = df["valence"] * df["energy"]

    # Acousticness × (1 − instrumentalness): the canonical signal for
    # acoustic AND vocal tracks. Folk, singer-songwriter, and unplugged
    # recordings score high. An electric track with vocals scores low
    # (low acousticness); a classical guitar solo scores low (high
    # instrumentalness); only acoustic + vocal reaches a high product.
    df["acoustic_vocal"] = df["acousticness"] * (1.0 - df["instrumentalness"])

    # Normalised tempo × danceability: fast AND danceable = dance-music
    # signature (house, techno, reggaeton). Tempo alone doesn't capture
    # this — a fast Bach invention is not danceable. Danceability alone
    # doesn't capture it — slow reggae is danceable but not fast.
    df["tempo_x_danceability"] = df["tempo_norm"] * df["danceability"]

    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.95,
    variance_fraction: float = 0.01,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Remove redundant and near-zero-variance numeric features.

    Pass 1 — Correlation filter
        If two numeric features have |r| > corr_threshold, drop the
        second (later-appearing) one. Prevents multicollinearity
        without losing the first occurrence of the signal.

    Pass 2 — Variance filter
        Drop any numeric feature whose variance is below
        variance_fraction × mean variance across all remaining
        features. A near-constant feature carries almost no
        discriminative signal and only adds noise.

    Parameters
    ----------
    df                : dataframe with original + engineered features
    corr_threshold    : absolute correlation above which a feature is
                        considered redundant (default 0.95)
    variance_fraction : features below this fraction of mean variance
                        are dropped (default 0.01 = 1%)

    Returns
    -------
    reduced_df      : pd.DataFrame — metadata + selected numeric cols
    selected_cols   : list[str]    — retained numeric feature names
    log             : list[str]    — human-readable drop reasons
    """
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype.kind in ("f", "i", "u", "b")
        and c not in META_COLS
    ]

    log: list[str] = []

    # ---- Pass 1: correlation filter ----
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    dropped_corr: set[str] = set()
    for col in numeric_cols:
        if col in dropped_corr:
            continue
        high = upper.loc[col]
        # NaN comparisons return False, so no dropna() needed
        for other, val in high.items():
            if val > corr_threshold and other not in dropped_corr:
                dropped_corr.add(other)
                log.append(
                    f"Dropped '{other}' "
                    f"(|r| = {val:.3f} with '{col}', above {corr_threshold} threshold)"
                )

    after_corr = [c for c in numeric_cols if c not in dropped_corr]

    # ---- Pass 2: variance filter ----
    variances = df[after_corr].var()
    mean_var = variances.mean()
    var_threshold = variance_fraction * mean_var

    dropped_var: list[str] = []
    for col, v in variances.items():
        if v < var_threshold:
            dropped_var.append(col)
            log.append(
                f"Dropped '{col}' "
                f"(variance = {v:.6f} < threshold {var_threshold:.6f})"
            )

    selected_cols = [c for c in after_corr if c not in dropped_var]

    # Preserve metadata + target alongside selected numeric features
    meta_present = [c for c in META_COLS if c in df.columns]
    # Also keep explicit (bool) and track_genre (target) if present
    extra = [c for c in ("explicit", "track_genre") if c in df.columns]
    keep = list(dict.fromkeys(meta_present + extra + selected_cols))

    return df[keep], selected_cols, log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    df = pd.read_csv(CLEANED_PATH, index_col=0)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    t0 = time.time()
    df_eng = create_features(df)
    new_cols = [c for c in df_eng.columns if c not in df.columns]
    print(f"\nEngineered {len(new_cols)} new features:")
    for col in new_cols:
        print(f"  {col}")

    df_sel, selected, log = select_features(df_eng)
    elapsed = time.time() - t0

    print("\nFeature selection log:")
    for entry in log:
        print(f"  - {entry}")

    print(f"\nAfter selection: {len(selected)} numeric features kept")
    print(f"Selected: {selected}")

    # Sanity checks
    null_count = df_sel[selected].isnull().sum().sum()
    inf_count  = np.isinf(df_sel[selected].select_dtypes("number").values).sum()
    print(f"\nSanity check — nulls: {null_count}  |  infinite values: {inf_count}")

    print(f"\nElapsed: {elapsed:.2f}s")
