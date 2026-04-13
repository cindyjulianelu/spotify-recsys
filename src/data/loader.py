import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dataset.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows:,} rows x {cols} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    print("\nColumn names and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<25} {dtype}")


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    stats = numeric.agg(["mean", "std", "min", "max"]).T
    stats.columns = ["mean", "std", "min", "max"]
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))


def print_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    if missing.empty:
        print("  No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"count": missing, "pct": pct})
        for col, row in report.iterrows():
            print(f"  {col:<25} {int(row['count']):>6,}  ({row['pct']}%)")


def print_outlier_flags(df: pd.DataFrame) -> None:
    print("\nOutlier flags:")

    zero_dur = (df["duration_ms"] == 0).sum()
    long_10  = (df["duration_ms"] > 600_000).sum()
    long_30  = (df["duration_ms"] > 1_800_000).sum()
    max_dur_min = df["duration_ms"].max() / 60_000
    print(f"  duration_ms = 0              {zero_dur:>6,}  (likely data error — drop)")
    print(f"  duration_ms > 10 min         {long_10:>6,}  (DJ mixes / live sets)")
    print(f"  duration_ms > 30 min         {long_30:>6,}  (longest: {max_dur_min:.1f} min)")

    zero_tempo = (df["tempo"] == 0).sum()
    print(f"  tempo = 0                    {zero_tempo:>6,}  (BPM undetected — treat as NaN)")

    zero_pop = (df["popularity"] == 0).sum()
    zero_pop_pct = zero_pop / len(df) * 100
    print(f"  popularity = 0               {zero_pop:>6,}  ({zero_pop_pct:.1f}% — obscure/delisted tracks)")


if __name__ == "__main__":
    df = load_dataset()
    print_shape(df)
    print_dtypes(df)
    print_summary_stats(df)
    print_missing(df)
    print_outlier_flags(df)
