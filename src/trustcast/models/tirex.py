"""TiRex-2 adapter (NX-AI, Apache-2.0). Install: pip install tirex-2"""

from __future__ import annotations

import numpy as np

from .base import DECILES, ForecastModel, from_quantile_grid


class TiRex(ForecastModel):
    def __init__(self, hf_id: str = "NX-AI/TiRex-2", revision: str | None = None,
                 device: str = "cpu", commercial_ok: bool = True, name: str = "tirex-2"):
        self.hf_id, self.revision, self.device = hf_id, revision, device
        self.commercial_ok, self.name = commercial_ok, name
        self._model = None

    def _load(self):
        if self._model is None:
            from tirex2 import load_model

            hf_kwargs = {"revision": self.revision} if self.revision else None
            self._model = load_model(self.hf_id, device=self.device, hf_kwargs=hf_kwargs)
        return self._model

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        import torch
        from tirex2 import TimeseriesType

        target = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(0)  # (1 variate, T)
        ts = TimeseriesType(target=target, past_covariates=None, future_covariates=None)
        out = self._load().forecast([ts], prediction_length=horizon, output_type="numpy")[0]
        grid = np.asarray(out)[0].T          # (1, 9 quantiles, horizon) -> (horizon, 9)
        return from_quantile_grid(grid, DECILES, quantiles, self.name)
