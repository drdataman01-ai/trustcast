"""Walk-forward back-test: forecast `horizon` steps from many past origins, moving
the origin `step` observations at a time, and score every forecast against what happened.

Usage:
  python scripts/backtest.py --dataset sp500 --series LMT --models chronos-2 timesfm-2.5 arima seasonal-naive
  python scripts/backtest.py --dataset sp500 --series LMT --models chronos-2 seasonal-naive --start 2025-11-01
  python scripts/backtest.py --series unemployment --horizon 12 --step 1 --models chronos-2 ets seasonal-naive
  python scripts/backtest.py --dataset sp500_weekly --series LMT --horizon 4 --step 1 --models chronos-2 seasonal-naive

Compare runs at different frequencies on one axis with scripts/compare_backtests.py.

Defaults: 21-step forecasts (about one trading month), origin moved 5 steps (one trading
week), 512 steps of context, at least 252 steps of history before the first origin.

Best practices this follows:
- Rolling origin, no look-ahead: each model sees only data before the origin.
- A naive benchmark (seasonal-naive; for daily prices that's the random walk, "tomorrow =
  today"). A model is only useful if it beats it.
- Point and probabilistic scores, by horizon step: MAE, RMSE, MASE, CRPS, interval
  coverage and the interval (Winkler) score; quantile calibration.
- Overlapping forecasts are autocorrelated: Diebold-Mariano tests use Newey-West
  variance and the Harvey-Leybourne-Newbold correction.
- Pretraining leakage: foundation models may have been trained on data from the test
  period (including correlated stocks). For a clean test, use --start after the latest
  release date of the models compared, and treat earlier origins as possibly optimistic.
- Reproducibility: every forecast, the config, model versions and git commit are saved.

Outputs: outputs/backtests/<series>_h<horizon>_s<step>_<timestamp>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trustcast.data.select import load_one, trading_days_per_step
from trustcast.evaluation.walkforward import (
    BacktestConfig, compare_to_baseline, origin_positions, quantile_calibration,
    run_backtest, score_rows, summarize,
)
from trustcast.models import LicenseError, get_model, load_roster

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#C98A1E", "#0E6E6B", "#7A4EAB", "#3A6EA5", "#B5533C", "#6B8E23", "#D16BA5", "#5A6B75"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward back-test of one series.")
    p.add_argument("--series", required=True)
    p.add_argument("--dataset", help="Dataset built by scripts/etl.py, e.g. sp500")
    p.add_argument("--models", nargs="+", required=True, help="Names from configs/models.yaml")
    p.add_argument("--baseline", default="seasonal-naive", help="Benchmark model (added if missing)")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--step", type=int, default=5, help="Steps between origins (5 = one trading week)")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--min-history", type=int, default=None,
                   help="History required before the first forecast (default: dataset setting, else 252)")
    p.add_argument("--start", help="Forecast targets start on or after this date, e.g. 2025-11-01")
    p.add_argument("--end", help="Last origin date")
    p.add_argument("--max-origins", type=int, help="Use only the most recent N origins")
    p.add_argument("--transform", choices=["log", "none"], default="log")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    try:
        sel = load_one(args.series, args.dataset, args.transform)
    except (KeyError, ValueError, FileNotFoundError) as e:
        sys.exit(e.args[0] if e.args else str(e))

    min_history = args.min_history
    if min_history is None:
        from trustcast.data.etl import load_spec
        min_history = (load_spec(args.dataset).get("min_history") if args.dataset else None) or 252
    cfg = BacktestConfig(horizon=args.horizon, step=args.step, context=args.context,
                         min_history=min_history, start=args.start, end=args.end,
                         max_origins=args.max_origins)
    positions = origin_positions(sel.y.index, cfg)
    if not positions:
        sys.exit("No origins fit: try an earlier --start or a smaller --min-history.")
    first, last = sel.y.index[positions[0] - 1], sel.y.index[positions[-1] - 1]
    print(f"{sel.label}: {len(sel.y)} observations, {sel.y.index[0]:%Y-%m-%d} to {sel.y.index[-1]:%Y-%m-%d}")
    print(f"{len(positions)} origins from {first:%Y-%m-%d} to {last:%Y-%m-%d}, "
          f"every {cfg.step} steps; {cfg.horizon}-step forecasts; context {cfg.context}.\n")

    names = list(dict.fromkeys(args.models + [args.baseline]))
    models = {}
    for name in names:
        try:
            models[name] = get_model(name, device=args.device)
        except KeyError:
            print(f"Skipping {name}: not in configs/models.yaml")
        except (NotImplementedError, LicenseError) as e:
            print(f"Skipping {name}: {e}")

    freq = sel.freq  # None for non-seasonal data such as stock prices
    forecasts = []
    for name, model in models.items():
        try:
            forecasts.append(run_backtest({name: model}, sel.y, cfg, freq=freq))
        except ImportError as e:
            print(f"Skipping {name}: its library isn't installed ({e.name}). "
                  "See src/trustcast/models/README.md")
    if not forecasts:
        sys.exit("No model ran.")
    fc = pd.concat(forecasts, ignore_index=True)

    scored = score_rows(fc, cfg.levels)
    if sel.log_scale:  # errors in log price ≈ percentage errors; also report exact % error of price
        scored["abs_pct_error_price"] = np.abs(np.expm1(scored["error"]))
    overall = summarize(scored, cfg.levels, ["model"])
    by_step = summarize(scored, cfg.levels, ["model", "step"])
    if sel.log_scale:
        pct = scored.groupby("model")["abs_pct_error_price"].mean().rename("mape_price")
        overall = overall.merge(pct, on="model")
    calib = quantile_calibration(scored)
    vs_base = compare_to_baseline(scored, args.baseline, cfg)

    # ---------- save ----------
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = ROOT / "outputs" / "backtests" / f"{sel.label.replace('/', '_')}_h{cfg.horizon}_s{cfg.step}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "forecasts.csv", index=False)
    overall.to_csv(out / "summary.csv", index=False)
    by_step.to_csv(out / "by_step.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    vs_base.to_csv(out / "vs_baseline.csv", index=False)
    roster = load_roster()
    meta = {
        "series": sel.label, "transform": args.transform, "config": cfg.to_dict(),
        "origins": len(positions), "first_origin": str(first.date()), "last_origin": str(last.date()),
        "baseline": args.baseline, "git_commit": git_commit(), "run_at": stamp,
        "trading_days_per_step": trading_days_per_step(sel.y.index), "seasonal_freq": freq,
        "data_sha256": hashlib.sha256(sel.y.to_numpy().tobytes()).hexdigest()[:16],
        "models": {m: {k: roster[m].get(k) for k in ("hf_id", "revision", "license", "version")}
                   for m in models if m in roster},
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---------- report ----------
    pd.set_option("display.width", 160)
    show = overall.set_index("model")
    cols = ["origins", "mae", "mase", "crps", "coverage_80", "coverage_95", "width_80", "bias"]
    if sel.log_scale:
        cols.insert(2, "mape_price")
    print("\nOverall (log-price errors: 0.04 means about 4%):")
    print(show[cols].sort_values("crps").round(4).to_string())
    if not vs_base.empty:
        print(f"\nAgainst {args.baseline} (skill > 0 = better; p < 0.05 = difference unlikely to be chance):")
        print(vs_base.set_index("model").round(3).to_string())
    print("\nIntervals: coverage_80 should be near 0.80 and coverage_95 near 0.95.")
    print(f"\nSaved to {out}")

    # ---------- charts ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip([m for m in models if m != args.baseline], COLORS))
    colors[args.baseline] = "#8A9AA2"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m, g in by_step.groupby("model"):
        axes[0].plot(g["step"], g["mae"], color=colors.get(m), lw=2 if m != args.baseline else 1.5,
                     ls="--" if m == args.baseline else "-", label=m)
        axes[1].plot(g["step"], g["coverage_80"], color=colors.get(m), lw=2 if m != args.baseline else 1.5,
                     ls="--" if m == args.baseline else "-", label=m)
    axes[0].set_title("Median forecast error by step ahead (MAE)")
    axes[0].set_xlabel("Steps ahead")
    axes[1].axhline(0.8, color="#8A9AA2", lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("80% interval coverage by step ahead (target 0.80)")
    axes[1].set_xlabel("Steps ahead")
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "by_step.png", dpi=150)

    # Final-step forecasts against what happened, for each model
    levels = sel.levels
    fig, ax = plt.subplots(figsize=(12, 4.5))
    hist = levels[levels.index >= first - pd.Timedelta(days=30)]
    ax.plot(hist.index, hist.values, color="#13212B", lw=1.4, label="Actual")
    for m in models:
        g = scored[(scored["model"] == m) & (scored["step"] == cfg.horizon)]
        conv = np.exp if sel.log_scale else (lambda v: v)
        ax.plot(g["target_date"], conv(g["q0.5"]), color=colors.get(m), lw=1.4,
                ls="--" if m == args.baseline else "-", label=f"{m} ({cfg.horizon}-step forecast)")
        if m != args.baseline and len(models) <= 3:
            ax.fill_between(g["target_date"], conv(g["q0.1"]), conv(g["q0.9"]), color=colors.get(m), alpha=0.12, lw=0)
    ax.set_title(f"{sel.label}: forecasts made {cfg.horizon} steps earlier vs actual")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "forecasts_vs_actual.png", dpi=150)
    print("Charts: by_step.png, forecasts_vs_actual.png")


if __name__ == "__main__":
    main()
