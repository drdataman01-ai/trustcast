"""Combine several models' forecasts; the spread captures uncertainty about which model is right."""

from __future__ import annotations

import numpy as np

from ..models.base import Forecast


def quantile_average(forecasts: list[Forecast], weights: list[float] | None = None) -> Forecast:
    """Weighted average of each quantile across models (simple, robust default)."""
    w = np.asarray(weights or [1.0] * len(forecasts), dtype=float)
    w = w / w.sum()
    levels = set.intersection(*(set(f.quantiles) for f in forecasts))
    qs = {q: sum(wi * f.quantiles[q] for wi, f in zip(w, forecasts)) for q in sorted(levels)}
    mean = sum(wi * f.mean for wi, f in zip(w, forecasts))
    return Forecast(mean=mean, quantiles=qs, model="ensemble")
