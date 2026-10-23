"""ETL: extract from a source, transform to one long format, validate, load into data/.

Every source ends up in the same shape, so everything downstream (preparation,
models, scorecards) works the same whatever the data came from:

    series_id   date        value
    AAPL        2020-01-02  72.71
    unemployment 2020-01-01 3.6

Add a source by writing one extract function and registering it in EXTRACTORS.
Datasets are described in configs/datasets.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from . import store

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ROOT / "configs" / "datasets.yaml"


# ---------- Extract: one function per source, each returning a long table ----------

def extract_yahoo(spec: dict) -> pd.DataFrame:
    from .sources import yahoo

    raw = (yahoo.read_pickle(spec["import_pickle"]) if spec.get("import_pickle")
           else yahoo.download(spec["tickers"], spec["start"], spec.get("end")))
    return yahoo.to_long(raw, spec.get("field", "Close"))


def extract_fred(spec: dict) -> pd.DataFrame:
    from .sources import fred

    ids = spec.get("ids")
    if ids is None:  # take the series list from a domain config
        domain = yaml.safe_load(open(ROOT / "configs" / "domains" / f"{spec['domain']}.yaml"))
        ids = {name: s["fred_id"] for name, s in domain["series"].items()}
    return fred.download(ids)


def extract_csv(spec: dict) -> pd.DataFrame:
    from .sources import csvfile

    return csvfile.read(spec["path"], spec.get("date_col"), spec.get("id_col"), spec.get("value_col"))


EXTRACTORS: dict[str, Callable[[dict], pd.DataFrame]] = {
    "yahoo": extract_yahoo,
    "fred": extract_fred,
    "csv": extract_csv,
}


# ---------- Transform + validate ----------

def standardize(long: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    """Enforce the schema: types, no duplicate (series_id, date), sorted, no empty rows."""
    missing = {"series_id", "date", "value"} - set(long.columns)
    if missing:
        raise ValueError(f"extract step must return series_id, date, value; missing {missing}")
    df = long[["series_id", "date", "value"]].copy()
    df["series_id"] = df["series_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    df = df.drop_duplicates(["series_id", "date"], keep="last")
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per series: coverage and basic checks. Raises if the table is unusable."""
    if df.empty:
        raise ValueError("No rows after extraction; check the source settings.")
    g = df.groupby("series_id")
    report = pd.DataFrame({
        "n_obs": g.size(),
        "start": g["date"].min().dt.date,
        "end": g["date"].max().dt.date,
        "min": g["value"].min(),
        "max": g["value"].max(),
        "n_nonpositive": g["value"].apply(lambda v: int((v <= 0).sum())),
    }).reset_index()
    return report


# ---------- Pipeline ----------

def load_spec(name: str, path: Path = DATASETS) -> dict:
    specs = yaml.safe_load(open(path))
    if name not in specs:
        raise KeyError(f"No dataset '{name}' in {path}. Known: {', '.join(specs)}")
    return specs[name]


def run_etl(name: str, spec: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract -> standardize -> validate -> save to data/<name>.parquet."""
    spec = spec or load_spec(name)
    source = spec["source"]
    if source not in EXTRACTORS:
        raise KeyError(f"Unknown source '{source}'. Add an extract function to EXTRACTORS.")
    df = standardize(EXTRACTORS[source](spec), spec.get("start"))
    report = validate(df)
    store.save_dataset(df, name, private=spec.get("private", False))
    return df, report
