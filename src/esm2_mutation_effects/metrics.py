"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        raise ValueError("Spearman correlation requires at least two observations")
    result = spearmanr(y_true, y_pred, nan_policy="raise")
    value = float(result.statistic)
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is undefined (possibly constant predictions)")
    return value
