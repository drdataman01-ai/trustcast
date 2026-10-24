"""Predicted vs actual change for a finished back-test: did any model call the direction?

Usage:
  python scripts/direction_check.py outputs/backtests/sp500_weekly_LMT_h4_s1_<time>
  python scripts/direction_check.py outputs/backtests/sp500_LMT_h21_s5_<time> --step 5

--step defaults to the run's horizon. Works on runs made before this check existed:
the last value each model saw is looked up again from data/.
Writes direction.png and direction.csv into the run folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trustcast.data.select import load_one
from trustcast.evaluation.walkforward import direction_check, plot_direction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", help="Back-test output folder")
    p.add_argument("--step", type=int, help="Steps ahead to check (default: the run's horizon)")
    args = p.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "config.json").read_text())
    scored = pd.read_csv(run / "forecasts.csv", parse_dates=["origin", "target_date"])
    step = args.step or meta["config"]["horizon"]
    log_scale = meta.get("transform", "log") == "log"

    if "origin_value" not in scored.columns:  # runs from before origin_value was recorded
        dataset, _, series = meta["series"].rpartition("/")
        sel = load_one(series, dataset or None, meta.get("transform", "log"))
        scored["origin_value"] = scored["origin"].map(sel.y)
        missing = scored["origin_value"].isna().sum()
        if missing:
            print(f"Note: {missing} rows had no matching origin in the current data and were skipped.")
            scored = scored.dropna(subset=["origin_value"])

    changes, summary = direction_check(scored, step, log_scale)
    summary.to_csv(run / "direction.csv", index=False)
    plot_direction(changes, summary, step, meta["series"], run / "direction.png",
                   "%" if log_scale else "units")
    pd.set_option("display.width", 160)
    print(f"{meta['series']}: change over {step} steps, {summary['forecasts'].max()} forecasts per model\n")
    print(summary.set_index("model").round(3).to_string())
    print("\nSkill would show as corr clearly above 0 and a hit rate above up_share "
          "(what 'always predict up' would score).")
    print(f"Saved {run / 'direction.png'} and direction.csv")


if __name__ == "__main__":
    main()
