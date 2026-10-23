"""TimesFM adapter (Google Research): 2.5 (Apache-2.0) and 3.0 (non-commercial).

Install (both versions, from Google's GitHub):
  pip install 'timesfm[mlx] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688'
Drop [mlx] off Apple Silicon; 3.0 then runs on PyTorch.
"""

from __future__ import annotations

import platform
import sys

import numpy as np

from .base import DECILES, ForecastModel, from_quantile_grid


class TimesFM(ForecastModel):
    def __init__(self, hf_id: str, version: str = "2.5", revision: str | None = None,
                 device: str = "cpu", commercial_ok: bool = True, name: str = "timesfm",
                 max_context: int = 1024):
        self.hf_id, self.version, self.revision, self.device = hf_id, str(version), revision, device
        self.commercial_ok, self.name, self.max_context = commercial_ok, name, max_context
        self._model = None
        self._compiled_horizon = 0

    # ---- 3.0: MLX on Apple Silicon, PyTorch elsewhere ----
    def _load_v3(self):
        kw = {"revision": self.revision} if self.revision else {}
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                from timesfm3.mlx import TimesFM3Forecaster

                return TimesFM3Forecaster.from_pretrained(self.hf_id, **kw)
            except ImportError:
                pass
        from timesfm3 import TimesFM3Forecaster

        return TimesFM3Forecaster.from_pretrained(self.hf_id, device=self.device, **kw)

    # ---- 2.5: compiled for a maximum horizon; recompiled if a longer one is asked for ----
    def _load_v25(self, horizon: int):
        import timesfm

        if self._model is None:
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.hf_id)
        if horizon > self._compiled_horizon:
            self._compiled_horizon = max(256, int(np.ceil(horizon / 128) * 128))
            self._model.compile(timesfm.ForecastConfig(
                max_context=self.max_context, max_horizon=self._compiled_horizon,
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
            ))
        return self._model

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        y = np.asarray(y, dtype=np.float32)
        if self.version.startswith("3"):
            if self._model is None:
                self._model = self._load_v3()
            out = self._model.predict(y, horizon=horizon, return_quantiles=True)
            grid = np.asarray(out.quantiles)                 # (horizon, 9): 0.1 ... 0.9
        else:
            model = self._load_v25(horizon)
            _, q = model.forecast(horizon=horizon, inputs=[y])
            grid = np.asarray(q)[0, :, 1:10]                 # index 0 is the mean; 1..9 = 0.1 ... 0.9
        return from_quantile_grid(grid, DECILES, quantiles, self.name)
