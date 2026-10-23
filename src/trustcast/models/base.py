"""One interface for every model, so all run under identical conditions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Forecast:
    """Point forecast plus quantiles for each step ahead.

    mean:      shape (horizon,)
    quantiles: {level: array of shape (horizon,)}, e.g. {0.05: ..., 0.95: ...}
    """

    mean: np.ndarray
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    model: str = ""

    def interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central interval, e.g. level=0.9 -> (q0.05, q0.95)."""
        lo, hi = round((1 - level) / 2, 4), round(1 - (1 - level) / 2, 4)
        return self.quantiles[lo], self.quantiles[hi]


class ForecastModel:
    """Base class. Subclasses implement `predict`; `fit` is optional
    (zero-shot foundation models need no fitting)."""

    name: str = "base"
    commercial_ok: bool = True

    def fit(self, y: np.ndarray, freq: str | None = None) -> "ForecastModel":
        return self

    def predict(
        self,
        y: np.ndarray,
        horizon: int,
        quantiles: list[float],
        freq: str | None = None,
        covariates: dict | None = None,
    ) -> Forecast:
        raise NotImplementedError


DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def from_quantile_grid(grid: np.ndarray, levels, quantiles: list[float], model: str) -> Forecast:
    """Build a Forecast from a model's native quantile grid.

    grid:   shape (horizon, len(levels)), the model's own quantile forecasts
    levels: the quantile levels of grid's columns, e.g. DECILES
    quantiles: the levels the caller asked for

    Levels inside the model's range are interpolated. Levels outside it (e.g. 0.025 or
    0.975 from a deciles-only model) are extended with a normal tail fitted to the
    outermost levels; the scorecard's conformal step then checks and corrects the width.
    """
    from statistics import NormalDist

    grid = np.sort(np.asarray(grid, dtype=float), axis=1)  # guard against quantile crossing
    levels = np.asarray(levels, dtype=float)
    nd = NormalDist()
    lo, hi = levels[0], levels[-1]
    med = np.array([np.interp(0.5, levels, row) for row in grid])
    sigma = (grid[:, -1] - grid[:, 0]) / (nd.inv_cdf(hi) - nd.inv_cdf(lo))
    out = {}
    for q in quantiles:
        if lo <= q <= hi:
            out[q] = np.array([np.interp(q, levels, row) for row in grid])
        else:
            out[q] = med + nd.inv_cdf(q) * sigma
    return Forecast(mean=med, quantiles=out, model=model)
