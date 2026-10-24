"""Walk-forward (rolling-origin) back-test with multi-step forecasts.

At each forecast origin, every model sees only data up to that origin, forecasts the
next `horizon` steps, and the origin then moves forward by `step` observations:

    origin 1:  [=== history ===]|--- 21 steps ---|
    origin 2:  [===== history =====]|--- 21 steps ---|          (+5 steps = 1 trading week)
    origin 3:  [======= history =======]|--- 21 steps ---|

Practices built in (see docs in scripts/backtest.py):
- No look-ahead: a model only ever receives y[:t]; preparation choices are fixed in
  config, not tuned on the test period.
- Same conditions for all: identical origins, context length and horizon per model.
- Every forecast is kept (origin, step, actual, quantiles), so any metric can be
  recomputed later and the run audited.
- Point and probabilistic scores: MAE/RMSE/MASE, pinball and CRPS, interval coverage
  and the interval (Winkler) score; compared with a naive benchmark.
- Overlapping forecasts (21-step horizons, 5-step moves) make scores autocorrelated, so
  significance uses a Diebold-Mariano test with Newey-West variance and the
  Harvey-Leybourne-Newbold small-sample correction.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..models.base import ForecastModel

DEFAULT_QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


@dataclass
class BacktestConfig:
    horizon: int = 21                 # steps forecast at each origin
    step: int = 5                     # steps the origin moves each time (5 trading days = 1 week)
    context: int = 512                # most recent observations given to each model
    min_history: int = 252            # first origin needs at least this much history
    start: str | None = None          # first forecast target date (e.g. after the models' release dates)
    end: str | None = None            # last origin date
    max_origins: int | None = None    # keep only the most recent N origins
    quantiles: list[float] = field(default_factory=lambda: list(DEFAULT_QUANTILES))
    levels: list[float] = field(default_factory=lambda: [0.8, 0.95])

    def to_dict(self) -> dict:
        return asdict(self)


def origin_positions(index: pd.DatetimeIndex, cfg: BacktestConfig) -> list[int]:
    """Positions t such that the model sees y[:t] and is scored on y[t : t + horizon].
    Only origins with a full horizon of actuals are used."""
    n = len(index)
    first = max(cfg.min_history, 1)
    if cfg.start:
        first = max(first, int(index.searchsorted(pd.Timestamp(cfg.start))))
    last = n - cfg.horizon
    if cfg.end:
        last = min(last, int(index.searchsorted(pd.Timestamp(cfg.end), side="right")))
    # Anchor the grid on the most recent origin so the latest data is always tested.
    positions = list(range(last, first - 1, -cfg.step))[::-1]
    if cfg.max_origins:
        positions = positions[-cfg.max_origins:]
    return positions


def run_backtest(models: dict[str, ForecastModel], y: pd.Series, cfg: BacktestConfig,
                 freq: str | None = None, progress: bool = True) -> pd.DataFrame:
    """Return every forecast as one row per (model, origin, step).

    Columns: model, origin (date of the last observation the model saw), target_date,
    step, actual, q<level> for each quantile, scale (in-sample naive MAE at the origin).
    """
    values = y.to_numpy(dtype=float)
    index = y.index
    positions = origin_positions(index, cfg)
    if not positions:
        raise ValueError("No origins: the series is too short for this horizon, history and start date.")

    rows = []
    for name, model in models.items():
        t0 = time.time()
        for i, t in enumerate(positions):
            history = values[max(0, t - cfg.context):t]          # the model sees only y[:t]
            actual = values[t:t + cfg.horizon]
            fc = model.predict(history.copy(), cfg.horizon, cfg.quantiles, freq=freq)
            scale = float(np.mean(np.abs(np.diff(values[:t])))) if t > 1 else np.nan
            block = pd.DataFrame({
                "model": name,
                "origin": index[t - 1],
                "target_date": index[t:t + cfg.horizon],
                "step": np.arange(1, cfg.horizon + 1),
                "actual": actual,
                "scale": scale,
                **{f"q{q:g}": np.asarray(fc.quantiles[q], dtype=float) for q in cfg.quantiles},
            })
            rows.append(block)
            if progress and (i + 1) % 25 == 0:
                print(f"  {name}: {i + 1}/{len(positions)} origins ({time.time() - t0:.0f}s)")
        if progress:
            print(f"  {name}: {len(positions)} origins done in {time.time() - t0:.0f}s")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- scoring

def _qcols(df: pd.DataFrame) -> dict[float, str]:
    return {float(c[1:]): c for c in df.columns if c.startswith("q") and c[1:].replace(".", "").isdigit()}


def score_rows(fc: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    """Add per-row scores: error, abs/squared error, CRPS (from quantiles),
    and for each interval level: covered, width, interval score."""
    df = fc.copy()
    q = _qcols(df)
    y = df["actual"].to_numpy()
    med = df[q[0.5]].to_numpy()
    df["error"] = med - y
    df["abs_error"] = np.abs(df["error"])
    df["sq_error"] = df["error"] ** 2
    df["scaled_abs_error"] = df["abs_error"] / df["scale"]
    pin = []
    for lev, col in q.items():
        d = y - df[col].to_numpy()
        pin.append(np.maximum(lev * d, (lev - 1) * d))
    df["crps"] = 2 * np.mean(pin, axis=0)                     # quantile approximation of CRPS
    for lev in levels:
        alpha = 1 - lev
        lo, hi = q[round(alpha / 2, 6)], q[round(1 - alpha / 2, 6)]
        l, u = df[lo].to_numpy(), df[hi].to_numpy()
        tag = int(round(lev * 100))
        df[f"covered_{tag}"] = (y >= l) & (y <= u)
        df[f"width_{tag}"] = u - l
        df[f"interval_score_{tag}"] = (u - l) + (2 / alpha) * (l - y) * (y < l) + (2 / alpha) * (y - u) * (y > u)
    return df


def summarize(scored: pd.DataFrame, levels: list[float], by: list[str]) -> pd.DataFrame:
    """Average scores by the given keys (e.g. ['model'] or ['model', 'step'])."""
    agg = {
        "origins": ("origin", "nunique"),
        "mae": ("abs_error", "mean"),
        "rmse": ("sq_error", lambda s: float(np.sqrt(s.mean()))),
        "mase": ("scaled_abs_error", "mean"),
        "bias": ("error", "mean"),
        "crps": ("crps", "mean"),
    }
    for lev in levels:
        tag = int(round(lev * 100))
        agg[f"coverage_{tag}"] = (f"covered_{tag}", "mean")
        agg[f"width_{tag}"] = (f"width_{tag}", "mean")
        agg[f"interval_score_{tag}"] = (f"interval_score_{tag}", "mean")
    return scored.groupby(by).agg(**agg).reset_index()


def quantile_calibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Share of actuals at or below each forecast quantile; a calibrated model's share
    equals the quantile level (e.g. 10% of actuals below q0.1)."""
    q = _qcols(scored)
    out = {lev: (scored["actual"] <= scored[col]).groupby(scored["model"]).mean() for lev, col in q.items()}
    return pd.DataFrame(out).rename_axis("model").reset_index()


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal accuracy with the Harvey-Leybourne-Newbold correction.

    loss_a, loss_b: losses of two models at the same origins, in time order.
    h: how many consecutive losses overlap (for 21-step forecasts moved 5 steps: ceil(21/5)).
    Uses a Newey-West long-run variance with h-1 lags. Negative statistic: a is more accurate.
    Returns (statistic, two-sided p-value from Student's t with n-1 df).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 8:
        return float("nan"), float("nan")
    dbar = d.mean()
    dc = d - dbar
    lrv = float(dc @ dc) / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * float(dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = dbar / math.sqrt(lrv / n)
    stat *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    try:
        from scipy.stats import t as student_t

        p = float(2 * student_t.sf(abs(stat), n - 1))
    except ImportError:
        p = float(math.erfc(abs(stat) / math.sqrt(2)))
    return float(stat), p


def compare_to_baseline(scored: pd.DataFrame, baseline: str, cfg: BacktestConfig) -> pd.DataFrame:
    """Skill and significance of each model against the baseline, on two losses:
    CRPS averaged over the whole horizon, and absolute error at the final step."""
    h_overlap = max(1, math.ceil(cfg.horizon / cfg.step))
    per_origin = scored.groupby(["model", "origin"]).agg(crps=("crps", "mean")).reset_index()
    last = scored[scored["step"] == cfg.horizon][["model", "origin", "abs_error"]]
    per_origin = per_origin.merge(last, on=["model", "origin"])
    wide = {m: g.set_index("origin").sort_index() for m, g in per_origin.groupby("model")}
    if baseline not in wide:
        return pd.DataFrame()
    base = wide[baseline]
    rows = []
    for m, g in wide.items():
        if m == baseline:
            continue
        g = g.reindex(base.index)
        s_crps, p_crps = dm_test(g["crps"].to_numpy(), base["crps"].to_numpy(), h_overlap)
        s_ae, p_ae = dm_test(g["abs_error"].to_numpy(), base["abs_error"].to_numpy(), h_overlap)
        rows.append({
            "model": m,
            "crps_skill_vs_baseline": 1 - g["crps"].mean() / base["crps"].mean(),
            "dm_stat_crps": s_crps, "p_value_crps": p_crps,
            f"mae_step{cfg.horizon}_ratio": g["abs_error"].mean() / base["abs_error"].mean(),
            f"dm_stat_step{cfg.horizon}": s_ae, f"p_value_step{cfg.horizon}": p_ae,
        })
    return pd.DataFrame(rows)
