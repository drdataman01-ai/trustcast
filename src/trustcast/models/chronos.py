"""Chronos adapter (Amazon): Chronos-2, Chronos-Bolt and original Chronos.
Weights are downloaded from Hugging Face and cached. pip install chronos-forecasting"""

from __future__ import annotations

import numpy as np

from .base import Forecast, ForecastModel


class Chronos(ForecastModel):
    def __init__(self, hf_id: str, revision: str | None = None, device: str = "cpu",
                 commercial_ok: bool = True, name: str = "chronos"):
        self.hf_id, self.revision, self.device = hf_id, revision, device
        self.commercial_ok, self.name = commercial_ok, name
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from chronos import BaseChronosPipeline

            self._pipe = BaseChronosPipeline.from_pretrained(
                self.hf_id, revision=self.revision, device_map=self.device
            )
        return self._pipe

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        import torch

        pipe = self._load()
        context = np.asarray(y, dtype=np.float32)
        if type(pipe).__name__ == "Chronos2Pipeline":
            # Chronos-2 takes a list of series; each output is (n_variates, horizon, n_quantiles)
            inputs = [context]
        else:
            # Chronos / Chronos-Bolt take a tensor; output is (batch, horizon, n_quantiles)
            inputs = torch.tensor(context)
        q, mean = pipe.predict_quantiles(inputs, prediction_length=horizon,
                                         quantile_levels=list(quantiles))
        q, mean = np.asarray(q[0], dtype=float), np.asarray(mean[0], dtype=float)
        while q.ndim > 2:      # drop the variate / batch axis -> (horizon, n_quantiles)
            q = q[0]
        while mean.ndim > 1:   # -> (horizon,)
            mean = mean[0]
        return Forecast(mean=mean, quantiles={ql: q[:, i] for i, ql in enumerate(quantiles)},
                        model=self.name)
