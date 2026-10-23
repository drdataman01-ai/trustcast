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
