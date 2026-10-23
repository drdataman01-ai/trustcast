"""Containers for richer problems: covariates, related series, hierarchies.

To build as domains are added:
- Panel: many related series with shared frequency (retail, energy)
- Covariates: known future inputs (prices, holidays, weather)
- Hierarchy: store -> region -> national, with reconciliation so totals add up
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SeriesSet:
    target: pd.DataFrame                      # columns: series_id, date, value
    past_covariates: pd.DataFrame | None = None
    future_covariates: pd.DataFrame | None = None
    hierarchy: dict[str, list[str]] = field(default_factory=dict)  # parent -> children
