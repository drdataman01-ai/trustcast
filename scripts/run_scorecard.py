"""Monthly scorecard run for one domain.

Usage: python scripts/run_scorecard.py --domain macro --models seasonal-naive arima chronos-2

Steps: load each series from data/ (as known at each origin) -> back-test every model under
the same conditions -> summarize -> save to archive/<domain>/<YYYY-MM>/.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from trustcast.data import store
from trustcast.evaluation.backtest import rolling_origin
from trustcast.evaluation.scorecard import publish, summarize
from trustcast.models import get_model

ROOT = Path(__file__).resolve().parents[1]
SEASON = {"M": 12, "Q": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="macro")
    p.add_argument("--models", nargs="+", default=["seasonal-naive"])
    p.add_argument("--origins", type=int, default=24)
    args = p.parse_args()

    defaults = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    domain = yaml.safe_load(open(ROOT / "configs" / "domains" / f"{args.domain}.yaml"))
    quantiles = defaults["quantiles"]

    results = []
    for series, spec in domain["series"].items():
        y = store.load_series(series).values  # from data/<series>.csv
        # TODO: pass history_at=... built from data.vintages for real-time scoring
        for h in spec["horizons"]:
            for name in args.models:
                df = rolling_origin(get_model(name), y, h, args.origins, quantiles,
                                    freq=spec["freq"], m=SEASON.get(spec["freq"], 1))
                df["series"], df["horizon"] = series, h
                results.append(df)

    table = summarize(pd.concat(results))
    print(table.to_string(index=False))
    print("saved:", publish(table, args.domain))


if __name__ == "__main__":
    main()
