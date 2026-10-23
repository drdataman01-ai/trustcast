"""Run Google's TimesFM 3.0 locally (MLX on Apple Silicon, PyTorch on CPU elsewhere).

Forecasts one series and reports the median with an 80% interval (10th to 90th percentile).

Single series in data/<name>.csv:
  python scripts/run_timesfm3.py --series unemployment --horizon 12
  python scripts/run_timesfm3.py --series gdp --horizon 4

One series from a dataset built by scripts/etl.py (e.g. a stock in data/sp500.parquet):
  python scripts/run_timesfm3.py --dataset sp500 --series LMT --horizon 21

  For datasets, the series is prepared with the dataset's settings from
  configs/datasets.yaml (calendar alignment, gap filling), then modeled on the log
  scale (--transform log, the default) so the forecast and its interval come back
  as prices exactly. --transform none models raw prices.

  python scripts/run_timesfm3.py --list                  # single series in data/
  python scripts/run_timesfm3.py --dataset sp500 --list  # series in a dataset

Outputs go to outputs/timesfm3/.

TimesFM 3.0 weights are under Google's timesfm-non-commercial-license-v1.0:
research and non-production use only (research lab, not the commercial arm).
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trustcast.data import store
from trustcast.data.select import future_dates, load_one

CHECKPOINT = "google/timesfm-3.0-pytorch"
ROOT = Path(__file__).resolve().parents[1]
Q10, Q50, Q90 = 0, 4, 8  # TimesFM 3.0 returns 9 deciles: 0.1, 0.2, ..., 0.9


def load_model():
    """MLX on Apple Silicon (fast, no PyTorch); PyTorch on CPU otherwise."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            from timesfm3.mlx import TimesFM3Forecaster

            return TimesFM3Forecaster.from_pretrained(CHECKPOINT), "MLX (Apple Silicon)"
        except ImportError:
            print("MLX backend not installed; falling back to PyTorch on CPU.")
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(CHECKPOINT, device="cpu"), "PyTorch (CPU)"


def main() -> None:
    p = argparse.ArgumentParser(description="Forecast one series with TimesFM 3.0.")
    p.add_argument("--series", default="unemployment",
                   help="Series name: a file in data/ (without .csv), or a series_id in --dataset")
    p.add_argument("--dataset", help="Dataset in data/ built by scripts/etl.py, e.g. sp500")
    p.add_argument("--transform", choices=["log", "none"], default="log",
                   help="Scale to model a dataset series on (default: log)")
    p.add_argument("--horizon", type=int, default=12, help="Steps to forecast (default: 12)")
    p.add_argument("--context", type=int, default=512, help="Most recent points to use (default: 512)")
    p.add_argument("--list", action="store_true", help="List the series available and exit")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG chart")
    args = p.parse_args()

    if args.list:
        if args.dataset:
            names = sorted(store.load_dataset(args.dataset)["series_id"].unique())
        else:
            names = store.available()
        print("\n".join(names) if len(names) else f"No series in {store.data_dir()}")
        return

    try:
        sel = load_one(args.series, args.dataset, args.transform)
    except (KeyError, ValueError, FileNotFoundError) as e:
        sys.exit(e.args[0] if e.args else str(e))
    if sel.filled:
        print(f"Note: filled {sel.filled} missing value(s) during preparation.")
    y, levels, label, log_scale = sel.y, sel.levels, sel.label, sel.log_scale

    if len(y) < 32:
        sys.exit(f"Only {len(y)} observations; TimesFM needs at least 32.")
    context = y.iloc[-args.context:].astype("float32")
    print(f"{label}: {len(y)} observations, {y.index[0]:%Y-%m-%d} to {y.index[-1]:%Y-%m-%d}")
    print(f"Using the last {len(context)} as context; forecasting {args.horizon} steps"
          f"{' on the log scale' if log_scale else ''}.")

    t0 = time.time()
    model, backend = load_model()
    print(f"Loaded TimesFM 3.0 with {backend} in {time.time() - t0:.1f}s")

    t0 = time.time()
    out = model.predict(context.to_numpy(), horizon=args.horizon, return_quantiles=True)
    print(f"Forecast took {time.time() - t0:.2f}s\n")

    q = np.asarray(out.quantiles)  # (horizon, 9)
    cols = {"q10": q[:, Q10], "median": np.asarray(out.forecast), "q90": q[:, Q90]}
    if log_scale:  # exp is monotonic, so quantiles of log price map exactly to quantiles of price
        cols = {k: np.exp(v) for k, v in cols.items()}
    result = pd.DataFrame(cols, index=future_dates(y.index, args.horizon))
    result.index.name = "date"
    print(f"Last value: {levels.iloc[-1]:.2f} on {levels.index[-1]:%Y-%m-%d}")
    print(result.round(2).to_string())

    out_dir = ROOT / "outputs" / "timesfm3"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label.replace('/', '_')}_h{args.horizon}"
    result.to_csv(out_dir / f"{stem}.csv")
    print(f"\nSaved {out_dir / f'{stem}.csv'}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = levels.iloc[-max(4 * args.horizon, 120):]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(hist.index, hist.values, color="#13212B", lw=1.6, label="History")
        ax.fill_between(result.index, result["q10"], result["q90"], color="#0E6E6B",
                        alpha=0.25, lw=0, label="80% interval (q10 to q90)")
        ax.plot(result.index, result["median"], color="#C98A1E", lw=2.2, label="Median forecast")
        ax.axvline(levels.index[-1], color="#8A9AA2", ls=":", lw=1)
        ax.set_title(f"TimesFM 3.0 forecast: {label}")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        print(f"Saved {out_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
