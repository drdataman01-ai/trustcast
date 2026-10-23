"""Real-time data: the series as it was known on a given date.

Macro data gets revised. A fair scorecard gives every model only the data
available at each forecast origin, and scores against a stated target
(first release or latest revision)."""

from __future__ import annotations

import pandas as pd


def as_of(releases: pd.DataFrame, when: str | pd.Timestamp) -> pd.Series:
    """releases: columns date, realtime_start, value (as from fred.all_vintages).
    Returns, for each observation date, the latest value published on or before `when`."""
    when = pd.Timestamp(when)
    known = releases[pd.to_datetime(releases["realtime_start"]) <= when]
    known = known.sort_values("realtime_start").groupby("date").last()
    return known["value"].astype(float).sort_index()


def first_release(releases: pd.DataFrame) -> pd.Series:
    """The first published value of each observation (a common scoring target)."""
    first = releases.sort_values("realtime_start").groupby("date").first()
    return first["value"].astype(float).sort_index()
