"""Yahoo Finance prices via yfinance (pip install yfinance)."""

from __future__ import annotations

from datetime import date

import pandas as pd

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


USER_AGENT = "TrustCast/0.1 (research; https://github.com/drdataman01-ai/trustcast)"


def sp500_symbols() -> list[str]:
    """Current S&P 500 members from Wikipedia, in Yahoo's format (BRK.B -> BRK-B).

    Wikipedia refuses requests without a User-Agent, so one is sent. Each successful
    fetch is saved to data/sp500_symbols.txt. If Wikipedia can't be reached, that file is
    used, or failing that the tickers already in the sp500 dataset.
    Note: today's members applied to past dates causes survivorship bias.
    """
    from io import StringIO

    from .. import store

    cache = store.data_dir() / "sp500_symbols.txt"
    try:
        import requests

        resp = requests.get(WIKI_SP500, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        symbols = sorted(table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(symbols) + "\n")
        return symbols
    except Exception as e:
        if cache.exists():
            print(f"Couldn't fetch the S&P 500 list from Wikipedia ({e}); using {cache}.")
            return [t for t in cache.read_text().split() if t]
        try:
            ids = sorted(store.load_dataset("sp500")["series_id"].unique())
            print(f"Couldn't fetch the S&P 500 list from Wikipedia ({e}); "
                  f"using the {len(ids)} tickers in the sp500 dataset.")
            return ids
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Couldn't fetch the S&P 500 list from Wikipedia ({e}). Set `tickers:` to a list "
            f"in configs/datasets.yaml, or put one ticker per line in {cache}."
        ) from e


def download(tickers: list[str] | str, start: str, end: str | None = None,
             auto_adjust: bool = True) -> pd.DataFrame:
    """Daily OHLCV, split- and dividend-adjusted when auto_adjust=True.
    Returns yfinance's wide frame: columns (field, ticker), index = trading days."""
    import yfinance as yf

    if tickers == "sp500":
        tickers = sp500_symbols()
    return yf.download(tickers, start=start, end=end or date.today().isoformat(),
                       auto_adjust=auto_adjust, progress=False, group_by="column", threads=True)


def read_pickle(path: str) -> pd.DataFrame:
    """A frame saved earlier with data.to_pickle(...)."""
    return pd.read_pickle(path)


def to_long(raw: pd.DataFrame, field: str = "Close") -> pd.DataFrame:
    """Wide yfinance frame -> long table: series_id (ticker), date, value."""
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [list(raw.columns.get_level_values(i).unique()) for i in range(2)]
        lvl = 0 if field in levels[0] else 1 if field in levels[1] else None
        if lvl is None:
            raise KeyError(f"Field '{field}' not in the data; found {levels[0][:6]}")
        wide = raw.xs(field, axis=1, level=lvl)
    else:  # a single ticker comes back with plain columns
        wide = raw[[field]].rename(columns={field: getattr(raw, "name", None) or "series"})
    wide = wide.copy()
    wide.index.name, wide.columns.name = "date", None
    long = wide.reset_index().melt(id_vars="date", var_name="series_id", value_name="value")
    return long[["series_id", "date", "value"]]
