"""Data preparation: turn a clean long table into model-ready series.

Every model in a scorecard gets exactly the same prepared series, so preparation
choices are made once, in config, and recorded in a report.

Steps, per series:
  1. sort, drop duplicate dates, cut at `end` (no look-ahead)
  2. trim leading/trailing gaps (e.g. a stock that listed after the start date)
  3. put on a calendar: a regular one if `freq` is set, or with `align`, the dates seen
     across the whole dataset (e.g. trading days), so a missing day shows up as a gap
  4. fill short interior gaps (`fill`, up to `fill_limit` steps); drop series with longer gaps
  5. optionally resample, e.g. to weekly (each week's last value), after daily gaps are handled
  6. transform (log, log returns, % change, differences), keeping what's needed to undo it
  7. optionally clip outliers on the transformed scale
  8. drop series shorter than `min_obs`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

TRANSFORMS = ("none", "log", "log_return", "pct_change", "diff")


@dataclass
class PrepConfig:
    freq: str | None = None          # "B", "D", "W", "MS", "QS"... None = keep observed dates
    align: bool = False              # with freq None: use every date seen in the dataset (trading days)
    resample: str | None = None      # e.g. "W-FRI": each week's last value (weekly close)
    fill: str = "ffill"              # ffill | interpolate | none
    fill_limit: int = 5              # longest gap (in steps) that may be filled
    transform: str = "none"          # one of TRANSFORMS
    clip_mad: float | None = None    # clip transformed values beyond median ± k·MAD
    min_obs: int = 64
    end: str | None = None           # last date to use (e.g. a back-test cut-off)

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        if self.fill not in ("ffill", "interpolate", "none"):
            raise ValueError("fill must be ffill, interpolate or none")


@dataclass
class Prepared:
    """Model-ready series plus everything needed to explain and undo the preparation."""

    series: dict[str, pd.Series]
    report: pd.DataFrame
    config: PrepConfig
    last_level: dict[str, float] = field(default_factory=dict)  # last raw value, for inverting

    def arrays(self) -> dict[str, np.ndarray]:
        """{series_id: float32 array}, the form foundation models take."""
        return {k: s.to_numpy(dtype="float32") for k, s in self.series.items()}

    def to_levels(self, series_id: str, path: np.ndarray) -> np.ndarray:
        """Undo the transform for a forecast path (e.g. a median forecast).
        Quantile paths of returns don't sum exactly to level quantiles; for intervals
        on levels, simulate paths or forecast levels directly."""
        t, last = self.config.transform, self.last_level[series_id]
        path = np.asarray(path, dtype=float)
        if t == "none":
            return path
        if t == "log":
            return np.exp(path)
        if t == "log_return":
            return last * np.exp(np.cumsum(path))
        if t == "pct_change":
            return last * np.cumprod(1 + path)
        return last + np.cumsum(path)  # diff

    def to_darts(self):
        """Optional: Darts TimeSeries objects (pip install darts)."""
        from darts import TimeSeries

        if self.config.freq:
            return {k: TimeSeries.from_series(s, freq=self.config.freq) for k, s in self.series.items()}
        return {k: TimeSeries.from_values(s.to_numpy()) for k, s in self.series.items()}


def _transform(s: pd.Series, how: str) -> pd.Series:
    if how == "none":
        return s
    if how == "log":
        return np.log(s)
    if how == "log_return":
        return np.log(s).diff().iloc[1:]
    if how == "pct_change":
        return s.pct_change().iloc[1:]
    return s.diff().iloc[1:]


def _longest_gap(s: pd.Series) -> int:
    isna = s.isna().to_numpy()
    longest = run = 0
    for v in isna:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return longest


def prepare_one(s: pd.Series, cfg: PrepConfig,
                calendar: pd.DatetimeIndex | None = None) -> tuple[pd.Series | None, dict]:
    info = {"n_raw": int(s.notna().sum())}
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if cfg.end:
        s = s[s.index <= pd.Timestamp(cfg.end)]
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return None, {**info, "status": "dropped", "reason": "no data"}
    s = s.loc[first:last]
    if cfg.freq:
        s = s.asfreq(cfg.freq)
    elif calendar is not None:
        s = s.reindex(calendar[(calendar >= first) & (calendar <= last)])
    info["gaps"] = int(s.isna().sum())
    info["longest_gap"] = _longest_gap(s)
    if cfg.fill == "ffill":
        s = s.ffill(limit=cfg.fill_limit)
    elif cfg.fill == "interpolate":
        s = s.interpolate(limit=cfg.fill_limit, limit_area="inside")
    if s.isna().any():
        return None, {**info, "status": "dropped",
                      "reason": f"gap longer than {cfg.fill_limit} steps"}
    if cfg.resample:
        raw_last = s.index[-1]
        s = s.resample(cfg.resample).last().dropna()
        if len(s) and raw_last < s.index[-1]:
            s = s.iloc[:-1]  # drop an unfinished final period (e.g. a week still in progress)
    if cfg.transform in ("log", "log_return") and (s <= 0).any():
        return None, {**info, "status": "dropped", "reason": "non-positive values; can't take logs"}
    info["last_level"] = float(s.iloc[-1])
    y = _transform(s, cfg.transform).replace([np.inf, -np.inf], np.nan).dropna()
    if cfg.clip_mad:
        med = y.median()
        mad = 1.4826 * (y - med).abs().median()
        if mad > 0:
            lo, hi = med - cfg.clip_mad * mad, med + cfg.clip_mad * mad
            info["clipped"] = int(((y < lo) | (y > hi)).sum())
            y = y.clip(lo, hi)
    if len(y) < cfg.min_obs:
        return None, {**info, "status": "dropped", "reason": f"fewer than {cfg.min_obs} obs"}
    info.update(status="kept", reason="", n_final=len(y),
                start=y.index[0].date(), end=y.index[-1].date())
    return y.astype(float).rename(s.name), info


def prepare(long: pd.DataFrame, cfg: PrepConfig | dict | None = None) -> Prepared:
    """long: columns series_id, date, value (the output of the ETL step)."""
    cfg = cfg if isinstance(cfg, PrepConfig) else PrepConfig(**(cfg or {}))
    df = long.assign(date=pd.to_datetime(long["date"]))
    calendar = pd.DatetimeIndex(sorted(df["date"].unique())) if cfg.align and not cfg.freq else None
    out, rows, levels = {}, [], {}
    for sid, g in df.groupby("series_id", sort=True):
        y, info = prepare_one(g.set_index("date")["value"].rename(sid), cfg, calendar)
        rows.append({"series_id": sid, **info})
        if y is not None:
            out[sid] = y
            levels[sid] = info["last_level"]
    cols = ["series_id", "status", "reason", "n_raw", "n_final", "start", "end",
            "gaps", "longest_gap", "clipped"]
    report = pd.DataFrame(rows).reindex(columns=cols)
    return Prepared(series=out, report=report, config=cfg, last_level=levels)


def describe(cfg: PrepConfig) -> dict:
    return asdict(cfg)
