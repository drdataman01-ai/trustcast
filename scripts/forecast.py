"""Forecast one series with one or more models from configs/models.yaml, side by side.

Usage:
  python scripts/forecast.py --dataset sp500 --series LMT --horizon 21 --models chronos-2
  python scripts/forecast.py --dataset sp500 --series LMT --horizon 21 --models chronos-2 arima seasonal-naive
  python scripts/forecast.py --series unemployment --horizon 12 --models chronos-2 ets

Models: timesfm-2.5, timesfm-3.0, chronos-2, moirai-2.0, toto-2.0, tirex-2, timegpt,
arima, ets, seasonal-naive (for daily stock data, seasonal-naive is the random walk:
tomorrow = today). Install notes: src/trustcast/models/README.md. A model whose
library isn't installed is skipped with a message.

Each model returns the median and an 80% interval (10th to 90th percentile). With
--dataset, models run on log prices and results are converted back to prices.
Outputs: outputs/forecasts/<series>_h<horizon>.csv and .png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trustcast.data.select import future_dates, load_one, short_freq
from trustcast.models import LicenseError, get_model

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [0.1, 0.5, 0.9]
COLORS = ["#C98A1E", "#0E6E6B", "#7A4EAB", "#B5533C", "#3A6EA5"]


def main() -> None:
    p = argparse.ArgumentParser(description="Forecast one series with several models.")
    p.add_argument("--series", required=True, help="Ticker/series in --dataset, or a file in data/")
    p.add_argument("--dataset", help="Dataset built by scripts/etl.py, e.g. sp500")
    p.add_argument("--models", nargs="+", default=["chronos-2"], help="Names from configs/models.yaml")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--context", type=int, default=512, help="Most recent points given to each model")
    p.add_argument("--transform", choices=["log", "none"], default="log",
                   help="For datasets: model log prices (default) or raw prices")
    p.add_argument("--device", default="cpu", help="cpu, or mps on Apple Silicon (Chronos only)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    try:
        sel = load_one(args.series, args.dataset, args.transform)
    except (KeyError, ValueError, FileNotFoundError) as e:
        sys.exit(e.args[0] if e.args else str(e))
    if sel.filled:
        print(f"Note: filled {sel.filled} missing value(s) during preparation.")
    y = sel.y.iloc[-args.context:]
    freq = short_freq(sel.y.index)
    dates = future_dates(sel.y.index, args.horizon)
    print(f"{sel.label}: {len(sel.y)} observations to {sel.y.index[-1]:%Y-%m-%d}; "
          f"last value {sel.levels.iloc[-1]:.2f}. Using {len(y)} as context, {args.horizon} steps ahead.\n")

    results = {}
    for name in args.models:
        try:
            model = get_model(name, device=args.device)
        except KeyError:
            print(f"Skipping {name}: not in configs/models.yaml")
            continue
        except NotImplementedError as e:
            print(f"Skipping {name}: {e}")
            continue
        except LicenseError as e:
            print(f"Skipping {name}: {e}")
            continue
        t0 = time.time()
        try:
            fc = model.predict(y.to_numpy(), args.horizon, QUANTILES, freq=freq)
        except ImportError as e:
            print(f"Skipping {name}: its library isn't installed ({e.name}). "
                  "See src/trustcast/models/README.md\n")
            continue
        cols = {"q10": fc.quantiles[0.1], "median": fc.quantiles[0.5], "q90": fc.quantiles[0.9]}
        if sel.log_scale:  # exp is monotonic: quantiles of log price -> quantiles of price
            cols = {k: np.exp(v) for k, v in cols.items()}
        results[name] = pd.DataFrame(cols, index=dates)
        print(f"{name} ({time.time() - t0:.1f}s)")
        print(results[name].round(2).to_string(), "\n")

    if not results:
        sys.exit("No model ran.")

    summary = pd.DataFrame({
        name: {"median at horizon": r["median"].iloc[-1], "q10 at horizon": r["q10"].iloc[-1],
               "q90 at horizon": r["q90"].iloc[-1],
               "80% band width": r["q90"].iloc[-1] - r["q10"].iloc[-1]}
        for name, r in results.items()
    }).T
    print(f"At {dates[-1]:%Y-%m-%d} (step {args.horizon}):")
    print(summary.round(2).to_string())

    out_dir = ROOT / "outputs" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sel.label.replace('/', '_')}_h{args.horizon}"
    table = pd.concat(results, axis=1)
    table.columns = [f"{m}_{c}" for m, c in table.columns]
    table.index.name = "date"
    table.to_csv(out_dir / f"{stem}.csv")
    print(f"\nSaved {out_dir / f'{stem}.csv'}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = sel.levels.iloc[-max(4 * args.horizon, 120):]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(hist.index, hist.values, color="#13212B", lw=1.5, label="History")
        for (name, r), c in zip(results.items(), COLORS):
            ax.fill_between(r.index, r["q10"], r["q90"], color=c, alpha=0.15, lw=0)
            ax.plot(r.index, r["median"], color=c, lw=2, label=f"{name} median (band: 80%)")
        ax.axvline(sel.levels.index[-1], color="#8A9AA2", ls=":", lw=1)
        ax.set_title(f"Forecasts for {sel.label}")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        print(f"Saved {out_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
