import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA: dict[str, str] = {
    "track_id":          "object",
    "artists":           "object",
    "album_name":        "object",
    "track_name":        "object",
    "popularity":        "int64",
    "duration_ms":       "int64",
    "explicit":          "bool",
    "danceability":      "float64",
    "energy":            "float64",
    "key":               "int64",
    "loudness":          "float64",
    "mode":              "int64",
    "speechiness":       "float64",
    "acousticness":      "float64",
    "instrumentalness":  "float64",
    "liveness":          "float64",
    "valence":           "float64",
    "tempo":             "float64",
    "time_signature":    "int64",
    "track_genre":       "object",
}

# (min_allowed, max_allowed)  —  inclusive, based on Spotify API spec
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "popularity":        (0,    100),
    # duration_ms intentionally excluded — handled by _check_duration_anomalies (check 6)
    "danceability":      (0.0,  1.0),
    "energy":            (0.0,  1.0),
    "key":               (0,    11),
    "loudness":          (-60,  5),
    "mode":              (0,    1),
    "speechiness":       (0.0,  1.0),
    "acousticness":      (0.0,  1.0),
    "instrumentalness":  (0.0,  1.0),
    "liveness":          (0.0,  1.0),
    "valence":           (0.0,  1.0),
    # tempo and time_signature intentionally excluded — handled by
    # _check_musical_metadata (check 7) with domain-specific logic
}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 1 — required columns exist with correct dtypes."""
    failures, warnings = [], []
    for col, expected_dtype in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            failures.append(f"[schema] Missing required column: '{col}'")
            continue
        actual = str(df[col].dtype)
        if actual != expected_dtype:
            warnings.append(
                f"[schema] '{col}' expected dtype '{expected_dtype}', got '{actual}'"
            )
    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 2 — enough rows to be useful."""
    failures, warnings = [], []
    n = len(df)
    if n < 100:
        failures.append(f"[row_count] Only {n:,} rows — minimum is 100")
    elif n < 1_000:
        warnings.append(f"[row_count] Only {n:,} rows — results may be unreliable (< 1,000)")
    return failures, warnings


def _check_null_rates(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 3 — null rates per column."""
    failures, warnings = [], []
    null_rates = df.isnull().mean()
    for col, rate in null_rates.items():
        pct = rate * 100
        if rate > 0.50:
            failures.append(f"[nulls] '{col}' is {pct:.1f}% null (> 50% threshold)")
        elif rate > 0.20:
            warnings.append(f"[nulls] '{col}' is {pct:.1f}% null (> 20% — consider imputation)")
    return failures, warnings


def _check_value_ranges(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 4 — numeric columns stay within sensible bounds."""
    failures, warnings = [], []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        below = (series < lo).sum()
        above = (series > hi).sum()
        if below:
            warnings.append(
                f"[range] '{col}' has {below:,} value(s) below minimum {lo}"
            )
        if above:
            warnings.append(
                f"[range] '{col}' has {above:,} value(s) above maximum {hi}"
            )
    return failures, warnings


def _check_target_distribution(
    df: pd.DataFrame, target_col: str
) -> tuple[list[str], list[str]]:
    """Check 5 — target column has enough classes, none critically rare."""
    failures, warnings = [], []

    if target_col not in df.columns:
        failures.append(f"[target] Target column '{target_col}' not found")
        return failures, warnings

    counts = df[target_col].value_counts(dropna=False)
    n_classes = len(counts)

    if n_classes < 2:
        failures.append(
            f"[target] '{target_col}' has only {n_classes} class — need 2+"
        )
        return failures, warnings

    # For multiclass (>2), scale the imbalance threshold proportionally.
    # Warn if any class holds less than half its expected uniform share.
    if n_classes == 2:
        imbalance_threshold = 0.05
    else:
        imbalance_threshold = max(0.01, (1.0 / n_classes) * 0.5)

    rare = counts[counts / len(df) < imbalance_threshold]
    if not rare.empty:
        worst_cls = rare.idxmin()
        worst_pct = counts[worst_cls] / len(df) * 100
        warnings.append(
            f"[target] '{target_col}' has {len(rare)} class(es) below "
            f"{imbalance_threshold*100:.1f}% threshold "
            f"(smallest: '{worst_cls}' at {worst_pct:.2f}%)"
        )

    return failures, warnings


def _check_duration_anomalies(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 6 — duration_ms: zero entries are critical, outlier-long entries are warnings.

    Kept separate from the generic value-ranges check (check 4) because the two
    failure modes have different severities and different root causes:

      - duration_ms = 0  →  critical failure. The track has no audio. A model
        trained on it would see a legitimate feature vector attached to a broken
        record. Should be dropped before any further processing.

      - duration_ms > 3,600,000 (1 hr)  →  warning. These are real audio files
        (DJ mixes, live sets) scraped under genre labels. They are not data errors
        but they are not songs either, and keeping them skews duration as a feature.
        Reported with genre context so the caller can make an informed decision.
    """
    failures, warnings = [], []

    if "duration_ms" not in df.columns:
        return failures, warnings

    zero = (df["duration_ms"] == 0).sum()
    if zero:
        failures.append(
            f"[duration] {zero:,} row(s) with duration_ms = 0 — no audio, must be dropped"
        )

    long_mask = df["duration_ms"] > 3_600_000
    n_long = long_mask.sum()
    if n_long:
        top_genres = (
            df.loc[long_mask, "track_genre"].value_counts().head(3).to_dict()
            if "track_genre" in df.columns
            else {}
        )
        genre_str = ", ".join(f"{g} ({n})" for g, n in top_genres.items())
        max_min = df["duration_ms"].max() / 60_000
        warnings.append(
            f"[duration] {n_long:,} row(s) exceed 1 hr "
            f"(longest: {max_min:.1f} min) — likely DJ mixes/live sets. "
            f"Top genres: {genre_str}"
        )

    return failures, warnings


def _check_musical_metadata(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 7 — domain-specific validation for tempo and time_signature.

    Generic range checks cannot catch these problems because the values are
    technically within the allowed numeric range. Catching them requires
    knowing what the numbers mean musically.

    time_signature = 0
        Spotify's way of saying it could not detect the meter. Not a real
        time signature — treat it the same as a null and impute.

    time_signature = 1
        1/4 time (one beat per bar) is virtually non-existent in recorded
        music. Almost certainly a detection artifact rather than a genuine
        meter. Flag as a warning.

    tempo > 200 BPM
        The top 0.1% of the dataset. The genres involved (trip-hop, blues,
        children's music, piano) are nowhere near 200 BPM in reality. This
        is a known MIR problem called tempo octave error: the BPM detector
        locks onto the eighth-note subdivisions instead of the beat itself
        and reports at 2× the true tempo. The fix is to halve the value,
        not drop the row.
    """
    failures, warnings = [], []

    if "time_signature" in df.columns:
        zero_ts = (df["time_signature"] == 0).sum()
        if zero_ts:
            failures.append(
                f"[music] {zero_ts:,} row(s) with time_signature = 0 "
                f"— meter undetected by Spotify, must be imputed before modelling"
            )
        one_ts = (df["time_signature"] == 1).sum()
        if one_ts:
            warnings.append(
                f"[music] {one_ts:,} row(s) with time_signature = 1 "
                f"— 1/4 time is virtually absent in recorded music, likely a detection artifact"
            )

    if "tempo" in df.columns:
        fast = (df["tempo"] > 200).sum()
        if fast:
            top_genres = (
                df.loc[df["tempo"] > 200, "track_genre"].value_counts().head(3).to_dict()
                if "track_genre" in df.columns else {}
            )
            genre_str = ", ".join(f"{g} ({n})" for g, n in top_genres.items())
            warnings.append(
                f"[music] {fast:,} tempo value(s) above 200 BPM — likely tempo octave "
                f"doubling (detector counted subdivisions instead of beats). "
                f"Top genres: {genre_str}"
            )

    return failures, warnings


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------

def check_data_quality(
    df: pd.DataFrame,
    target_col: str = "track_genre",
) -> dict:
    """Run all 7 data quality checks and return a structured report."""
    all_failures: list[str] = []
    all_warnings: list[str] = []

    for check_fn in (
        lambda df: _check_schema(df),
        lambda df: _check_row_count(df),
        lambda df: _check_null_rates(df),
        lambda df: _check_value_ranges(df),
        lambda df: _check_target_distribution(df, target_col),
        lambda df: _check_duration_anomalies(df),
        lambda df: _check_musical_metadata(df),
    ):
        f, w = check_fn(df)
        all_failures.extend(f)
        all_warnings.extend(w)

    statistics = {
        "total_rows":            len(df),
        "total_columns":         len(df.columns),
        "total_nulls_by_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "out_of_bounds_by_column": {
            col: {
                "below": int((df[col].dropna() < lo).sum()),
                "above": int((df[col].dropna() > hi).sum()),
            }
            for col, (lo, hi) in NUMERIC_BOUNDS.items()
            if col in df.columns
            and ((df[col].dropna() < lo).any() or (df[col].dropna() > hi).any())
        },
        "target_class_distribution": (
            (df[target_col].value_counts(normalize=True) * 100)
            .round(2)
            .to_dict()
            if target_col in df.columns
            else {}
        ),
    }

    return {
        "success":    len(all_failures) == 0,
        "failures":   all_failures,
        "warnings":   all_warnings,
        "statistics": statistics,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    status = "PASSED" if report["success"] else "FAILED"
    print(f"\n{'='*56}")
    print(f"  Data Quality Gate: {status}")
    print(f"{'='*56}")

    stats = report["statistics"]
    print(f"\n  Rows:    {stats['total_rows']:,}")
    print(f"  Columns: {stats['total_columns']}")

    if report["failures"]:
        print(f"\n  CRITICAL ({len(report['failures'])})")
        for msg in report["failures"]:
            print(f"    x  {msg}")

    if report["warnings"]:
        print(f"\n  WARNINGS ({len(report['warnings'])})")
        for msg in report["warnings"]:
            print(f"    !  {msg}")

    if not report["failures"] and not report["warnings"]:
        print("\n  All checks clean — no issues found.")

    oob = stats.get("out_of_bounds_by_column")
    if oob:
        print(f"\n  Out-of-bounds columns:")
        for col, counts in oob.items():
            print(f"    {col:<25}  below={counts['below']:,}  above={counts['above']:,}")

    nulls = stats.get("total_nulls_by_column")
    if nulls:
        print(f"\n  Null counts:")
        for col, n in nulls.items():
            pct = n / stats["total_rows"] * 100
            print(f"    {col:<25}  {n:>6,}  ({pct:.2f}%)")

    print(f"\n{'='*56}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dataset.csv"
    df = pd.read_csv(DATA_PATH, index_col=0)

    report = check_data_quality(df, target_col="track_genre")
    _print_report(report)
