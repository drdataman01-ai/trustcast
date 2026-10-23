"""FRED / ALFRED loader for macro series. Needs FRED_API_KEY in .env."""

from __future__ import annotations

import os

import pandas as pd


def latest(fred_id: str) -> pd.Series:
    """Latest revised values of a series."""
    from fredapi import Fred

    return Fred(api_key=os.environ["FRED_API_KEY"]).get_series(fred_id)


def all_vintages(fred_id: str) -> pd.DataFrame:
    """Every release of every observation (ALFRED): columns date, realtime_start, value.
    Used by vintages.py to rebuild what was known on each forecast date."""
    from fredapi import Fred

    return Fred(api_key=os.environ["FRED_API_KEY"]).get_series_all_releases(fred_id)
