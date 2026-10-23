"""Ongoing calibration checks: are a model's intervals too narrow or too wide?"""

from __future__ import annotations

import numpy as np


def coverage_report(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float,
                    tolerance: float = 0.05) -> dict:
    inside = (y_true >= lo) & (y_true <= hi)
    cov = float(inside.mean())
    status = "ok"
    if cov < level - tolerance:
        status = "too narrow"
    elif cov > level + tolerance:
        status = "too wide"
    return {"level": level, "coverage": cov, "n": int(len(y_true)), "status": status}
