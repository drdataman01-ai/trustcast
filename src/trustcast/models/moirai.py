"""Moirai 2.0 adapter (Salesforce, CC-BY-NC-4.0: research lab only).

Install in a SEPARATE environment: uni2ts pins torch<2.5 and numpy 1.26, which
conflict with the other models. Use Python 3.11:
  python3.11 -m venv .venv-moirai && source .venv-moirai/bin/activate
  pip install -e . uni2ts pandas pyyaml pyarrow matplotlib
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel, from_quantile_grid


class Moirai(ForecastModel):
    def __init__(self, hf_id: str = "Salesforce/moirai-2.0-R-small", revision: str | None = None,
                 device: str = "cpu", commercial_ok: bool = False, name: str = "moirai-2.0",
                 max_context: int = 1680):
        self.hf_id, self.revision, self.device = hf_id, revision, device
        self.commercial_ok, self.name, self.max_context = commercial_ok, name, max_context
        self._module = None

    def _load(self):
        if self._module is None:
            from uni2ts.model.moirai2 import Moirai2Module

            kw = {"revision": self.revision} if self.revision else {}
            self._module = Moirai2Module.from_pretrained(self.hf_id, **kw)
        return self._module

    def predict(self, y, horizon, quantiles, freq=None, covariates=None):
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast

        y = np.asarray(y, dtype=np.float32)
        model = Moirai2Forecast(
            module=self._load(), prediction_length=horizon,
            context_length=min(len(y), self.max_context),
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1, device=self.device)
        # Moirai 2.0 doesn't use the calendar; a placeholder start date is enough.
        ds = ListDataset([{"start": pd.Period("2000-01-01", freq="D"), "target": y}], freq="D")
        fc = next(iter(predictor.predict(ds)))
        keys = [(float(k), i) for i, k in enumerate(fc.forecast_keys) if k != "mean"]
        keys.sort()
        levels = [k for k, _ in keys]
        grid = np.asarray(fc.forecast_array)[[i for _, i in keys]].T   # (horizon, n_levels)
        return from_quantile_grid(grid, levels, quantiles, self.name)
