"""Chronos adapter (Amazon). Weights are downloaded from Hugging Face and cached."""

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
            from chronos import BaseChronosPipeline  # pip install chronos-forecasting

            self._pipe = BaseChronosPipeline.from_pretrained(
                self.hf_id, revision=self.revision, device_map=self.device
            )
        return self._pipe

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        import torch

        pipe = self._load()
        context = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        q, mean = pipe.predict_quantiles(context, prediction_length=horizon,
                                         quantile_levels=list(quantiles))
        q, mean = q[0].numpy(), mean[0].numpy()  # (horizon, n_quantiles), (horizon,)
        return Forecast(mean=mean, quantiles={ql: q[:, i] for i, ql in enumerate(quantiles)},
                        model=self.name)
