"""Adapters still to write. Each should subclass ForecastModel (see base.py),
load weights lazily by hf_id + pinned revision, and return a Forecast with
quantiles (use base.from_quantile_grid for models with a fixed quantile grid).

- nowcasts.py -> GDPNow and other published nowcasts, as baselines

Done: chronos.py, timesfm.py (2.5 and 3.0), moirai.py, toto.py, tirex.py, timegpt.py
"""
