"""Accuracy and interval-reliability metrics used in every scorecard."""

from __future__ import annotations

import numpy as np


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """Mean absolute scaled error vs. the in-sample seasonal naive forecast."""
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def interval_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def pinball(y_true, q_pred, q: float) -> float:
    diff = np.asarray(y_true) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def crps_from_quantiles(y_true, quantiles: dict[float, np.ndarray]) -> float:
    """Approximate CRPS as twice the mean pinball loss over the quantile levels."""
    return float(2 * np.mean([pinball(y_true, v, q) for q, v in quantiles.items()]))
