"""Classical baselines: seasonal naive (built in), ARIMA and ETS (statsforecast)."""

from __future__ import annotations

import numpy as np

from .base import Forecast, ForecastModel

_SEASON = {"M": 12, "Q": 4, "W": 52, "D": 7, "H": 24}


class SeasonalNaive(ForecastModel):
    """Repeat the last season. Intervals from the spread of past seasonal errors."""

    name = "seasonal-naive"

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        y = np.asarray(y, dtype=float)
        m = _SEASON.get(freq or "", 1)
        m = m if len(y) > m else 1
        idx = [len(y) - m + (h % m) for h in range(horizon)]
        mean = y[idx]
        errors = y[m:] - y[:-m]
        qs = {}
        for q in quantiles:
            k = np.arange(1, horizon + 1)
            steps = np.ceil(k / m)  # error grows with number of seasons ahead
            qs[q] = mean + np.quantile(errors, q) * np.sqrt(steps)
        return Forecast(mean=mean, quantiles=qs, model=self.name)


class StatsForecastModel(ForecastModel):
    """AutoARIMA / AutoETS via the statsforecast package."""

    def __init__(self, method: str = "auto_arima"):
        self.method = method
        self.name = method.replace("auto_", "")

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        from statsforecast.models import AutoARIMA, AutoETS

        m = _SEASON.get(freq or "", 1)
        model = AutoARIMA(season_length=m) if self.method == "auto_arima" else AutoETS(season_length=m)
        levels = sorted({round(abs(1 - 2 * q) * 100) for q in quantiles if q != 0.5})
        out = model.forecast(y=np.asarray(y, dtype=float), h=horizon, level=levels)
        qs = {}
        for q in quantiles:
            if q == 0.5:
                qs[q] = out["mean"]
            else:
                lvl = round(abs(1 - 2 * q) * 100)
                qs[q] = out[f"lo-{lvl}"] if q < 0.5 else out[f"hi-{lvl}"]
        return Forecast(mean=out["mean"], quantiles=qs, model=self.name)
