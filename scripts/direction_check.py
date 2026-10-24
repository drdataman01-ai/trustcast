"""Predicted vs actual change for a finished back-test: did any model call the direction?

Usage:
  python scripts/direction_check.py sp500_weekly_LMT     # newest weekly LMT run
  python scripts/direction_check.py sp500_LMT --step 20  # newest daily LMT run, 20 days ahead
  python scripts/direction_check.py                      # newest run of any kind
  python scripts/direction_check.py outputs/backtests/sp500_weekly_LMT_h4_s1_20261023-1412

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


ROOT = Path(__file__).resolve().parents[1]
BACKTESTS = ROOT / "outputs" / "backtests"


def find_run(arg: str) -> Path:
    """A folder path as given, else the newest run in outputs/backtests whose name starts with arg."""
    if arg and (Path(arg) / "config.json").exists():
        return Path(arg)
    prefix = Path(arg).name if arg else ""
    runs = sorted((d for d in BACKTESTS.glob(f"{prefix}*") if (d / "config.json").exists()),
                  key=lambda d: d.name[-13:])  # names end in _YYYYMMDD-HHMM
    if not runs:
        have = "\n  ".join(sorted(d.name for d in BACKTESTS.glob("*") if d.is_dir())) or "(none)"
        raise SystemExit(f"No back-test run matches '{arg}'. Runs in {BACKTESTS}:\n  {have}")
    return runs[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="",
                   help="Back-test folder, or the start of its name (e.g. sp500_weekly_LMT); "
                        "the newest match is used. Omit for the newest run.")
    p.add_argument("--step", type=int, help="Steps ahead to check (default: the run's horizon)")
    args = p.parse_args()

    run = find_run(args.run)
    print(f"Run: {run}")
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
