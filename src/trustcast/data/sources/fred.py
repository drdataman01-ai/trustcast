"""FRED / ALFRED loader for macro series. Needs FRED_API_KEY in .env."""

from __future__ import annotations

import os

import pandas as pd


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download(ids: dict[str, str]) -> pd.DataFrame:
    """Latest values for {name: fred_id}, no API key needed.
    Returns a long table: series_id (name), date, value."""
    frames = []
    for name, fred_id in ids.items():
        df = pd.read_csv(FRED_CSV.format(fred_id))
        frames.append(pd.DataFrame({
            "series_id": name,
            "date": df.iloc[:, 0],
            "value": pd.to_numeric(df.iloc[:, 1], errors="coerce"),  # FRED marks gaps with "."
        }))
    return pd.concat(frames, ignore_index=True)


def latest(fred_id: str) -> pd.Series:
    """Latest revised values of a series."""
    from fredapi import Fred

    return Fred(api_key=os.environ["FRED_API_KEY"]).get_series(fred_id)


def all_vintages(fred_id: str) -> pd.DataFrame:
    """Every release of every observation (ALFRED): columns date, realtime_start, value.
    Used by vintages.py to rebuild what was known on each forecast date."""
    from fredapi import Fred

    return Fred(api_key=os.environ["FRED_API_KEY"]).get_series_all_releases(fred_id)
