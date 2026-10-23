"""Conformal correction: widen or narrow a model's intervals using its own
recent errors so they reach their stated coverage."""

from __future__ import annotations

import numpy as np

from ..models.base import Forecast


def conformity_scores(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CQR score: how far outside (positive) or inside (negative) the interval each truth fell."""
    return np.maximum(lo - y_true, y_true - hi)


def adjust(forecast: Forecast, past_scores: np.ndarray, level: float) -> Forecast:
    """Shift the `level` interval by the conformal quantile of past scores."""
    lo, hi = forecast.interval(level)
    n = len(past_scores)
    k = min(1.0, np.ceil((n + 1) * level) / n) if n else 1.0
    q = np.quantile(past_scores, k) if n else 0.0
    qs = dict(forecast.quantiles)
    a = round((1 - level) / 2, 4)
    qs[a], qs[round(1 - a, 4)] = lo - q, hi + q
    return Forecast(mean=forecast.mean, quantiles=qs, model=f"{forecast.model}+conformal")
