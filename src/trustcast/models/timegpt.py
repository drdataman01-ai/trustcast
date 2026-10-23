"""TimeGPT adapter (Nixtla, hosted API). Install: pip install nixtla
Needs NIXTLA_API_KEY in .env. The series is sent to Nixtla's servers, so don't use
it on client data in data/private/ without the client's permission."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecast, ForecastModel

# Our short frequency codes -> pandas frequencies for the placeholder timestamps
_FREQ = {"M": "MS", "Q": "QS", "W": "W", "D": "D", "H": "h"}


class TimeGPT(ForecastModel):
    def __init__(self, model_name: str = "timegpt-2.1", commercial_ok: bool = True,
                 name: str = "timegpt"):
        self.model_name, self.commercial_ok, self.name = model_name, commercial_ok, name
        self._client = None

    def _load(self):
        if self._client is None:
            from dotenv import load_dotenv
            from nixtla import NixtlaClient

            load_dotenv()
            self._client = NixtlaClient()  # reads NIXTLA_API_KEY
        return self._client

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        y = np.asarray(y, dtype=float)
        pfreq = _FREQ.get(freq or "", "B")  # unknown/daily trading data -> business days
        ds = pd.date_range(end="2025-12-31", periods=len(y), freq=pfreq)
        df = pd.DataFrame({"unique_id": "series", "ds": ds, "y": y})
        qs = sorted(set(quantiles) | {0.5})
        out = self._load().forecast(df=df, h=horizon, freq=pfreq, quantiles=qs, model=self.model_name)
        # Nixtla names quantile columns TimeGPT-q-{int(q * 100)}
        result = {q: out[f"TimeGPT-q-{int(q * 100)}"].to_numpy() for q in quantiles}
        median = out[f"TimeGPT-q-{int(0.5 * 100)}"].to_numpy()
        return Forecast(mean=median, quantiles=result, model=self.name)
