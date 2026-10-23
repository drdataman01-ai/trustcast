"""Download a domain's series into data/<domain>/.

Usage:
  python scripts/fetch_data.py --domain macro              # latest values, no API key needed
  python scripts/fetch_data.py --domain macro --vintages   # also every release (needs FRED_API_KEY)

Series come from configs/domains/<domain>.yaml. Re-run any time to refresh.
"""

import argparse
from pathlib import Path

import yaml

from trustcast.data import store

ROOT = Path(__file__).resolve().parents[1]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="macro")
    p.add_argument("--vintages", action="store_true", help="also download ALFRED vintages")
    args = p.parse_args()

    domain = yaml.safe_load(open(ROOT / "configs" / "domains" / f"{args.domain}.yaml"))
    if domain.get("source") != "fred":
        raise SystemExit(f"Only FRED downloads are set up so far; put {args.domain} files "
                         f"in {store.data_dir() / args.domain} by hand.")

    if args.vintages:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        from trustcast.data.sources import fred

    for name, spec in domain["series"].items():
        s = store.read_csv_series(FRED_CSV.format(spec["fred_id"]))
        path = store.save_series(s, args.domain, name)
        print(f"{name:<16} {len(s):>5} obs  {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d}  -> {path}")
        if args.vintages:
            v = fred.all_vintages(spec["fred_id"])
            print(f"{'':<16} {len(v):>5} releases -> {store.save_vintages(v, args.domain, name)}")


if __name__ == "__main__":
    main()
