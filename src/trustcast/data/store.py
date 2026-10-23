"""Read and write series in the data/ folder.

Every program gets its data through this module, so the files only live in
one place. Layout (flat):

    data/<name>.csv              date,value                   public series
    data/<name>_vintages.csv     date,realtime_start,value    every release (optional)
    data/private/<name>.csv      date,value                   client and partner data (not in git)

load_series(name) looks in data/ first, then data/private/.
Set TRUSTCAST_DATA to use a folder outside the repo (e.g. an external drive).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return Path(os.getenv("TRUSTCAST_DATA", ROOT / "data"))


def private_dir() -> Path:
    return data_dir() / "private"


def series_path(name: str, private: bool = False) -> Path:
    return (private_dir() if private else data_dir()) / f"{name}.csv"


def vintages_path(name: str) -> Path:
    return data_dir() / f"{name}_vintages.csv"


def read_csv_series(path: str | Path) -> pd.Series:
    """Any CSV whose first column is a date and second is a value.
    Blank or non-numeric values (FRED writes ".") are dropped."""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: need two columns, a date then a value")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"), name=Path(path).stem)
    s = s[s.index.notna()].dropna().sort_index()
    s.index.name = "date"
    return s.astype(float)


def load_series(name: str) -> pd.Series:
    """Load data/<name>.csv, or data/private/<name>.csv if it's not public."""
    for path in (series_path(name), series_path(name, private=True)):
        if path.exists():
            return read_csv_series(path).rename(name)
    raise FileNotFoundError(
        f"No '{name}.csv' in {data_dir()} or {private_dir()}. "
        "Run: python scripts/fetch_data.py, or save the CSV there yourself."
    )


def save_series(s: pd.Series, name: str, private: bool = False) -> Path:
    path = series_path(name, private)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename("value").rename_axis("date").to_csv(path, date_format="%Y-%m-%d")
    return path


def load_vintages(name: str) -> pd.DataFrame:
    path = vintages_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/fetch_data.py --vintages")
    return pd.read_csv(path, parse_dates=["date", "realtime_start"])


def save_vintages(df: pd.DataFrame, name: str) -> Path:
    path = vintages_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "realtime_start", "value"]].to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


# ---------- Datasets: many series in one long table (series_id, date, value) ----------

def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def dataset_path(name: str, private: bool = False) -> Path:
    """data/<name>.parquet (or .csv.gz if pyarrow isn't installed)."""
    ext = ".parquet" if _has_parquet() else ".csv.gz"
    return (private_dir() if private else data_dir()) / f"{name}{ext}"


def save_dataset(df: pd.DataFrame, name: str, private: bool = False) -> Path:
    path = dataset_path(name, private)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_dataset(name: str) -> pd.DataFrame:
    """Load data/<name>.parquet|.csv.gz, or the same under data/private/."""
    for folder in (data_dir(), private_dir()):
        for ext in (".parquet", ".csv.gz"):
            path = folder / f"{name}{ext}"
            if path.exists():
                df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
                return df.assign(date=pd.to_datetime(df["date"]))
    raise FileNotFoundError(f"No dataset '{name}' in {data_dir()}. Run: python scripts/etl.py {name}")


def available() -> list[str]:
    """Series names in data/ and data/private/ (vintage files excluded)."""
    names = []
    for folder, prefix in ((data_dir(), ""), (private_dir(), "private/")):
        if folder.exists():
            names += [prefix + p.stem for p in sorted(folder.glob("*.csv"))
                      if not p.stem.endswith("_vintages")]
    return names
