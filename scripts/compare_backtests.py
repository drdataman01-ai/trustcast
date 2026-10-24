"""Compare back-test runs made at different data frequencies on one horizon axis.

Each run's steps are converted to trading days ahead (1 per daily step, 5 per weekly
step), so a weekly 4-step forecast lines up with a daily 20-step forecast.

Usage:
  python scripts/compare_backtests.py outputs/backtests/sp500_LMT_h21_s5_<time> \\
                                      outputs/backtests/sp500_weekly_LMT_h4_s1_<time>
  python scripts/compare_backtests.py RUN_A RUN_B --labels daily weekly

Output: comparison.png and comparison.csv in the folder of the last run given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLORS = ["#C98A1E", "#0E6E6B", "#7A4EAB", "#3A6EA5", "#B5533C", "#6B8E23", "#D16BA5", "#5A6B75"]
STYLES = ["--", "-", ":", "-."]
MARKERS = [None, "o", "s", "^"]


def load_run(path: Path, label: str | None) -> pd.DataFrame:
    meta = json.loads((path / "config.json").read_text())
    by_step = pd.read_csv(path / "by_step.csv")
    tdps = meta.get("trading_days_per_step", 1)
    by_step["days_ahead"] = by_step["step"] * tdps
    by_step["run"] = label or f"{meta['series']} ({'daily' if tdps == 1 else 'weekly' if tdps == 5 else f'{tdps}d steps'})"
    by_step["origins_in_run"] = meta.get("origins")
    return by_step


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay back-tests on a trading-days-ahead axis.")
    p.add_argument("runs", nargs="+", help="Back-test output folders")
    p.add_argument("--labels", nargs="+", help="A label per run, e.g. daily weekly")
    args = p.parse_args()
    labels = args.labels or [None] * len(args.runs)
    data = pd.concat([load_run(Path(r), l) for r, l in zip(args.runs, labels)], ignore_index=True)

    runs = list(dict.fromkeys(data["run"]))
    models = list(dict.fromkeys(data["model"]))
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    if "seasonal-naive" in color:
        color["seasonal-naive"] = "#8A9AA2"

    # Side-by-side table at the horizons the runs share
    shared = sorted(set.intersection(*[set(g["days_ahead"]) for _, g in data.groupby("run")]))
    cols = ["run", "model", "days_ahead", "origins", "mae", "crps", "coverage_80", "coverage_95"]
    table = data[data["days_ahead"].isin(shared)][cols].sort_values(["days_ahead", "model", "run"])
    out_dir = Path(args.runs[-1])
    table.to_csv(out_dir / "comparison.csv", index=False)
    pd.set_option("display.width", 160)
    if shared:
        print(f"At {shared[-1]} trading days ahead (log-price errors: 0.05 means about 5%):")
        print(table[table["days_ahead"] == shared[-1]].round(4).to_string(index=False))
    else:
        print("The runs share no horizon in trading days; see comparison.png.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (run, model), g in data.groupby(["run", "model"], sort=False):
        k = runs.index(run)
        kw = dict(color=color[model], ls=STYLES[k % len(STYLES)], marker=MARKERS[k % len(MARKERS)],
                  ms=5, lw=1.8 if k else 1.3)
        axes[0].plot(g["days_ahead"], g["mae"], **kw)
        axes[1].plot(g["days_ahead"], g["coverage_80"], **kw)
    axes[0].set_title("Median forecast error by trading days ahead (MAE)")
    axes[1].axhline(0.8, color="#8A9AA2", lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("80% interval coverage by trading days ahead (target 0.80)")
    for ax in axes:
        ax.set_xlabel("Trading days ahead")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
    model_keys = [Line2D([], [], color=color[m], lw=2, label=m) for m in models]
    run_keys = [Line2D([], [], color="#13212B", ls=STYLES[k % len(STYLES)], marker=MARKERS[k % len(MARKERS)],
                       label=r) for k, r in enumerate(runs)]
    axes[0].legend(handles=model_keys, frameon=False, fontsize=9, loc="upper left")
    axes[1].legend(handles=run_keys, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    print(f"\nSaved {out_dir / 'comparison.png'} and comparison.csv")


if __name__ == "__main__":
    main()
