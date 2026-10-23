"""Yahoo Finance prices via yfinance (pip install yfinance)."""

from __future__ import annotations

from datetime import date

import pandas as pd

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_symbols() -> list[str]:
    """Current S&P 500 members from Wikipedia, in Yahoo's format (BRK.B -> BRK-B).
    Note: today's members applied to past dates causes survivorship bias."""
    table = pd.read_html(WIKI_SP500)[0]
    return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())


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
