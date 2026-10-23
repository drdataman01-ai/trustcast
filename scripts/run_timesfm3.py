"""Run Google's TimesFM 3.0 locally (MLX on Apple Silicon, PyTorch on CPU elsewhere).

Reads a series from data/<domain>/<series>.csv and reports the median forecast
with an 80% interval (10th to 90th percentile).

Usage:
  python scripts/run_timesfm3.py --series unemployment --horizon 12
  python scripts/run_timesfm3.py --domain macro --series gdp --horizon 4
  python scripts/run_timesfm3.py --list                # show series in data/<domain>/

Fill data/ first with: python scripts/fetch_data.py --domain macro
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

CHECKPOINT = "google/timesfm-3.0-pytorch"
ROOT = Path(__file__).resolve().parents[1]
Q10, Q50, Q90 = 0, 4, 8  # TimesFM 3.0 returns 9 deciles: 0.1, 0.2, ..., 0.9


def future_dates(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Dates for the forecast steps, using the data's own frequency."""
    freq = pd.infer_freq(index[-12:]) or pd.infer_freq(index)
    if freq is None:
        step = index[-1] - index[-2]
        return pd.DatetimeIndex([index[-1] + step * (i + 1) for i in range(horizon)])
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


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
    p.add_argument("--domain", default="macro", help="Folder under data/ (default: macro)")
    p.add_argument("--series", default="unemployment", help="File name in data/<domain>/, without .csv")
    p.add_argument("--horizon", type=int, default=12, help="Steps to forecast (default: 12)")
    p.add_argument("--context", type=int, default=512, help="Most recent points to use (default: 512)")
    p.add_argument("--list", action="store_true", help="List the series available and exit")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG chart")
    args = p.parse_args()

    if args.list:
        names = store.available(args.domain)
        print("\n".join(names) if names else f"No series in {store.data_dir() / args.domain}")
        return

    series = store.load_series(args.domain, args.series)
    if len(series) < 32:
        sys.exit(f"Only {len(series)} observations; TimesFM needs at least 32.")
    context = series.iloc[-args.context:].astype("float32")
    print(f"{args.domain}/{args.series}: {len(series)} observations, "
          f"{series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d}")
    print(f"Using the last {len(context)} as context; forecasting {args.horizon} steps.")

    t0 = time.time()
    model, backend = load_model()
    print(f"Loaded TimesFM 3.0 with {backend} in {time.time() - t0:.1f}s")

    t0 = time.time()
    out = model.predict(context.to_numpy(), horizon=args.horizon, return_quantiles=True)
    print(f"Forecast took {time.time() - t0:.2f}s\n")

    q = np.asarray(out.quantiles)  # (horizon, 9)
    result = pd.DataFrame(
        {"q10": q[:, Q10], "median": np.asarray(out.forecast), "q90": q[:, Q90]},
        index=future_dates(series.index, args.horizon),
    )
    result.index.name = "date"
    print(result.round(3).to_string())

    out_dir = ROOT / "outputs" / "timesfm3"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.domain}_{args.series}_h{args.horizon}"
    result.to_csv(out_dir / f"{stem}.csv")
    print(f"\nSaved {out_dir / f'{stem}.csv'}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = series.iloc[-max(4 * args.horizon, 60):]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(hist.index, hist.values, color="#13212B", lw=1.8, label="History")
        ax.fill_between(result.index, result["q10"], result["q90"], color="#0E6E6B",
                        alpha=0.25, lw=0, label="80% interval (q10 to q90)")
        ax.plot(result.index, result["median"], color="#C98A1E", lw=2.2, label="Median forecast")
        ax.axvline(series.index[-1], color="#8A9AA2", ls=":", lw=1)
        ax.set_title(f"TimesFM 3.0 forecast: {args.domain}/{args.series}")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        print(f"Saved {out_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
