"""Any CSV, wide (date + one column per series) or long (series_id, date, value)."""

from __future__ import annotations

import pandas as pd


def read(path: str, date_col: str | None = None, id_col: str | None = None,
         value_col: str | None = None) -> pd.DataFrame:
    """Return a long table: series_id, date, value.

    Long CSV: give id_col and value_col.
    Wide CSV: every column except the date becomes a series (named by its header).
    date_col defaults to the first column.
    """
    df = pd.read_csv(path)
    date_col = date_col or df.columns[0]
    if id_col:
        long = df.rename(columns={id_col: "series_id", date_col: "date",
                                  value_col or "value": "value"})
    else:
        long = df.melt(id_vars=date_col, var_name="series_id", value_name="value")
        long = long.rename(columns={date_col: "date"})
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["series_id", "date", "value"]]
