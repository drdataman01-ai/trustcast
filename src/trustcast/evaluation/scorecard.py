"""Build the monthly scorecard from back-test results and save it to archive/."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ARCHIVE = Path(__file__).resolve().parents[3] / "archive"


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model (and series, if present): average metrics, ranked by MASE."""
    keys = [c for c in ("series", "model") if c in results.columns]
    numeric = results.drop(columns=["origin"], errors="ignore")
    table = numeric.groupby(keys).mean(numeric_only=True).reset_index()
    return table.sort_values(keys[:-1] + ["mase"]).reset_index(drop=True)


def publish(table: pd.DataFrame, domain: str, month: str | None = None) -> Path:
    """Write archive/<domain>/<YYYY-MM>/scorecard.csv. Commit this file: it is the public record."""
    month = month or date.today().strftime("%Y-%m")
    out = ARCHIVE / domain / month
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scorecard.csv"
    table.to_csv(path, index=False)
    return path
