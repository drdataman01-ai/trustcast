"""Rolling-origin back-test: every model gets the same history at the same origins."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..models.base import ForecastModel
from . import metrics


def rolling_origin(
    model: ForecastModel,
    y: np.ndarray,
    horizon: int,
    n_origins: int,
    quantiles: list[float],
    freq: str | None = None,
    level: float = 0.9,
    m: int = 1,
    history_at: Callable[[int], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Forecast from the last `n_origins` cut points and score each one.

    history_at(t): optional function returning the data *as known* at origin t
    (use data.vintages for real-time macro data). Defaults to y[:t].
    """
    y = np.asarray(y, dtype=float)
    rows = []
    for t in range(len(y) - horizon - n_origins + 1, len(y) - horizon + 1):
        hist = history_at(t) if history_at else y[:t]
        truth = y[t : t + horizon]
        fc = model.predict(hist, horizon, quantiles, freq=freq)
        lo, hi = fc.interval(level)
        rows.append({
            "model": model.name,
            "origin": t,
            "mase": metrics.mase(truth, fc.mean, hist, m),
            "smape": metrics.smape(truth, fc.mean),
            "crps": metrics.crps_from_quantiles(truth, fc.quantiles),
            f"coverage_{int(level * 100)}": metrics.coverage(truth, lo, hi),
            "width": metrics.interval_width(lo, hi),
        })
    return pd.DataFrame(rows)
