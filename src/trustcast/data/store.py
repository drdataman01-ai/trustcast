"""Read and write series in the data/ folder.

Every program gets its data through this module, so the files only live in
one place. Layout:

    data/<domain>/<name>.csv                  date,value          latest values
    data/<domain>/vintages/<name>.csv         date,realtime_start,value   every release
    data/private/                             client and partner data (not in git)

Set TRUSTCAST_DATA to use a folder outside the repo (e.g. an external drive).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return Path(os.getenv("TRUSTCAST_DATA", ROOT / "data"))


def series_path(domain: str, name: str) -> Path:
    return data_dir() / domain / f"{name}.csv"


def vintages_path(domain: str, name: str) -> Path:
    return data_dir() / domain / "vintages" / f"{name}.csv"


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


def load_series(domain: str, name: str) -> pd.Series:
    path = series_path(domain, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_data.py --domain {domain}"
        )
    return read_csv_series(path).rename(name)


def save_series(s: pd.Series, domain: str, name: str) -> Path:
    path = series_path(domain, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename("value").rename_axis("date").to_csv(path, date_format="%Y-%m-%d")
    return path


def load_vintages(domain: str, name: str) -> pd.DataFrame:
    path = vintages_path(domain, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_data.py --domain {domain} --vintages"
        )
    return pd.read_csv(path, parse_dates=["date", "realtime_start"])


def save_vintages(df: pd.DataFrame, domain: str, name: str) -> Path:
    path = vintages_path(domain, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "realtime_start", "value"]].to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def available(domain: str) -> list[str]:
    folder = data_dir() / domain
    return sorted(p.stem for p in folder.glob("*.csv")) if folder.exists() else []
