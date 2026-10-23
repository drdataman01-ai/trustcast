"""Pick one series to forecast, from data/<name>.csv or from an ETL dataset,
and work out the dates of the forecast steps. Shared by the forecasting scripts."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import etl, store
from .prepare import PrepConfig, prepare_one


@dataclass
class Selected:
    label: str            # e.g. "sp500/LMT" or "unemployment"
    y: pd.Series          # model input (log scale when log_scale)
    levels: pd.Series     # original units, for charts and the last value
    log_scale: bool       # True: forecasts must be exponentiated back to levels
    filled: int = 0       # missing values filled during preparation


def load_one(series: str, dataset: str | None = None, transform: str = "log") -> Selected:
    """A single series. With a dataset, it's prepared with that dataset's settings
    (calendar alignment, gap filling) and put on the `transform` scale ("log" or "none")."""
    if not dataset:
        s = store.load_series(series)
        return Selected(series, s, s, False)

    df = store.load_dataset(dataset)
    ids = df["series_id"].unique()
    if series not in ids:
        close = difflib.get_close_matches(series, ids, n=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"'{series}' is not in dataset '{dataset}'.{hint}")

    cfg = dict(etl.load_spec(dataset).get("prepare", {}))
    cfg.update(transform=transform, min_obs=32)
    cfg = PrepConfig(**cfg)
    calendar = pd.DatetimeIndex(sorted(df["date"].unique())) if cfg.align and not cfg.freq else None
    raw = df[df["series_id"] == series].set_index("date")["value"].rename(series)
    y, info = prepare_one(raw, cfg, calendar)
    if y is None:
        raise ValueError(f"{series} can't be used: {info['reason']}.")
    levels = np.exp(y) if transform == "log" else y
    return Selected(f"{dataset}/{series}", y, levels, transform == "log", int(info.get("gaps") or 0))


def future_dates(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Dates for the forecast steps, using the data's own frequency.
    Trading-day data (gaps for weekends and holidays) continues on business days."""
    freq = pd.infer_freq(index[-12:]) or pd.infer_freq(index)
    if freq is None:
        typical_days = pd.Series(index[-60:]).diff().dt.days.median()
        if typical_days <= 1.5:
            freq = "B"  # business days; exchange holidays are not excluded
        else:
            step = index[-1] - index[-2]
            return pd.DatetimeIndex([index[-1] + step * (i + 1) for i in range(horizon)])
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


def short_freq(index: pd.DatetimeIndex) -> str | None:
    """'M', 'Q', 'W', 'D' or None, for models that use a season length."""
    f = pd.infer_freq(index[-24:]) or ""
    for code in ("Q", "M", "W", "D"):
        if f.startswith(code):
            return code
    return None
