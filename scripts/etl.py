"""Run the ETL and data-preparation pipeline for one dataset.

Usage:
  python scripts/etl.py sp500                       # download, clean, save data/sp500.parquet, prepare
  python scripts/etl.py sp500 --import-pickle ~/Downloads/data/stocks.pkl   # reuse a saved download
  python scripts/etl.py sp500 --skip-extract        # re-prepare from data/ without downloading
  python scripts/etl.py macro

Datasets are defined in configs/datasets.yaml. Reports go to outputs/etl/:
  <name>_validation.csv  coverage per series after extraction
  <name>_prep.csv        what preparation kept, dropped and why
"""

import argparse
from pathlib import Path

from trustcast.data import etl, store
from trustcast.data.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="name in configs/datasets.yaml")
    p.add_argument("--import-pickle", help="use a pickle saved from yfinance instead of downloading")
    p.add_argument("--skip-extract", action="store_true", help="prepare the copy already in data/")
    args = p.parse_args()

    spec = etl.load_spec(args.dataset)
    if args.import_pickle:
        spec["import_pickle"] = str(Path(args.import_pickle).expanduser())

    out = ROOT / "outputs" / "etl"
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        df = store.load_dataset(args.dataset)
        print(f"Loaded {len(df):,} rows from data/")
    else:
        df, report = etl.run_etl(args.dataset, spec)
        report.to_csv(out / f"{args.dataset}_validation.csv", index=False)
        print(f"Extracted {df['series_id'].nunique()} series, {len(df):,} rows "
              f"-> {store.dataset_path(args.dataset, spec.get('private', False))}")

    prepared = prepare(df, spec.get("prepare"))
    prepared.report.to_csv(out / f"{args.dataset}_prep.csv", index=False)
    kept = (prepared.report["status"] == "kept").sum()
    print(f"Prepared {kept} of {len(prepared.report)} series "
          f"(transform: {prepared.config.transform}).")
    dropped = prepared.report[prepared.report["status"] == "dropped"]
    if len(dropped):
        print("Dropped:")
        for reason, n in dropped["reason"].value_counts().items():
            print(f"  {n:>4}  {reason}")
    print(f"Reports in {out}")


if __name__ == "__main__":
    main()
