"""Adapters still to write. Each should subclass ForecastModel (see base.py)
and follow chronos.py: load weights lazily by hf_id + pinned revision, and
return a Forecast with quantiles.

- timesfm.py  -> TimesFM 2.5 (Apache-2.0) and 3.0 (research only)
- moirai.py   -> Moirai 2.0 (CC-BY-NC-4.0, research only)
- toto.py     -> Toto 2.0
- tirex.py    -> TiRex-2
- timegpt.py  -> TimeGPT via Nixtla API (key in .env)
- nowcasts.py -> GDPNow and other published nowcasts, as baselines
"""
