"""Build models by name from configs/models.yaml, enforcing license rules."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .base import ForecastModel
from .classical import SeasonalNaive, StatsForecastModel

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


class LicenseError(RuntimeError):
    pass


def load_roster(path: Path = CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {**cfg.get("foundation", {}), **cfg.get("baselines", {})}


def get_model(name: str, mode: str | None = None, device: str = "cpu") -> ForecastModel:
    """mode: 'research' (all models) or 'commercial' (commercially licensed only)."""
    mode = mode or os.getenv("TRUSTCAST_MODE", "research")
    spec = load_roster()[name]

    if mode == "commercial" and not spec.get("commercial_ok", True):
        raise LicenseError(f"{name} ({spec.get('license')}) is research-only.")

    adapter = spec["adapter"]
    if adapter == "classical":
        return SeasonalNaive() if spec["method"] == "seasonal_naive" else StatsForecastModel(spec["method"])
    if adapter == "chronos":
        from .chronos import Chronos

        return Chronos(spec["hf_id"], spec.get("revision"), device,
                       spec.get("commercial_ok", True), name)
    raise NotImplementedError(f"Adapter '{adapter}' not written yet (see adapters_todo.py).")
