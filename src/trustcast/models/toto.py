"""Toto 2.0 adapter (Datadog, Apache-2.0). Install: pip install toto-models  (Python >= 3.12)
Sizes: Datadog/Toto-2.0-4m, -22m, -313m, -1B, -2.5B (set hf_id in configs/models.yaml)."""

from __future__ import annotations

import numpy as np

from .base import DECILES, ForecastModel, from_quantile_grid


class Toto(ForecastModel):
    def __init__(self, hf_id: str = "Datadog/Toto-2.0-313m", revision: str | None = None,
                 device: str = "cpu", commercial_ok: bool = True, name: str = "toto-2.0"):
        self.hf_id, self.revision, self.device = hf_id, revision, device
        self.commercial_ok, self.name = commercial_ok, name
        self._model = None

    def _load(self):
        if self._model is None:
            from toto2 import Toto2Model

            kw = {"revision": self.revision} if self.revision else {}
            self._model = Toto2Model.from_pretrained(self.hf_id, **kw).to(self.device).eval()
        return self._model

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        import torch

        model = self._load()
        target = torch.tensor(np.asarray(y, dtype=np.float32), device=self.device).reshape(1, 1, -1)
        inputs = {
            "target": target,                                        # (batch, variates, time)
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long, device=self.device),
        }
        with torch.no_grad():
            q = model.forecast(inputs, horizon=horizon, has_missing_values=False)
        grid = q[:, 0, 0, :].detach().cpu().numpy().T   # (9, batch, variates, horizon) -> (horizon, 9)
        return from_quantile_grid(grid, DECILES, quantiles, self.name)
