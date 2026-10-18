"""Metrics for separating between-position and within-position ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Return Spearman correlation or None when it is mathematically undefined."""

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if len(truth) < 2:
        return None
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Spearman inputs must be finite")
    if np.ptp(truth) <= 1e-12 or np.ptp(prediction) <= 1e-12:
        return None
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else None


def position_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positions: np.ndarray,
    minimum_group_size: int = 3,
) -> dict[str, float | int | None]:
    """Compute overall, between-position, and within-position ranking metrics.

    Pairwise accuracy pools all within-position pairs with unequal true scores.
    A prediction tie receives half credit. Per-position Spearman is summarized
    only for groups with at least ``minimum_group_size`` variants and variation
    in both truth and predictions.
    """

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    position = np.asarray(positions)
    if truth.ndim != 1 or prediction.ndim != 1 or position.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if not (len(truth) == len(prediction) == len(position)):
        raise ValueError("Metric inputs must have equal lengths")
    if len(truth) == 0:
        raise ValueError("At least one prediction is required")
    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")

    frame = pd.DataFrame(
        {"true": truth, "predicted": prediction, "position": position}
    )
    position_means = frame.groupby("position", sort=True)[["true", "predicted"]].mean()
    centered_true = frame["true"] - frame.groupby("position")["true"].transform("mean")
    centered_prediction = frame["predicted"] - frame.groupby("position")[
        "predicted"
    ].transform("mean")

    concordance_credit = 0.0
    pair_count = 0
    positions_with_pairs = 0
    within_spearman = []
    for _, group in frame.groupby("position", sort=True):
        group_truth = group["true"].to_numpy(dtype=np.float64)
        group_prediction = group["predicted"].to_numpy(dtype=np.float64)
        group_pairs = 0
        group_credit = 0.0
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                true_difference = group_truth[left] - group_truth[right]
                if np.isclose(true_difference, 0.0, rtol=1e-12, atol=1e-12):
                    continue
                predicted_difference = group_prediction[left] - group_prediction[right]
                group_pairs += 1
                if np.isclose(predicted_difference, 0.0, rtol=1e-12, atol=1e-12):
                    group_credit += 0.5
                elif np.sign(true_difference) == np.sign(predicted_difference):
                    group_credit += 1.0
        if group_pairs:
            positions_with_pairs += 1
            pair_count += group_pairs
            concordance_credit += group_credit

        if len(group) >= minimum_group_size:
            correlation = safe_spearman(group_truth, group_prediction)
            if correlation is not None:
                within_spearman.append(correlation)

    return {
        "overall_spearman": safe_spearman(truth, prediction),
        "between_position_spearman": safe_spearman(
            position_means["true"].to_numpy(),
            position_means["predicted"].to_numpy(),
        ),
        "within_position_centered_spearman": safe_spearman(
            centered_true.to_numpy(),
            centered_prediction.to_numpy(),
        ),
        "within_position_pairwise_accuracy": (
            float(concordance_credit / pair_count) if pair_count else None
        ),
        "within_position_spearman_mean": (
            float(np.mean(within_spearman)) if within_spearman else None
        ),
        "within_position_spearman_median": (
            float(np.median(within_spearman)) if within_spearman else None
        ),
        "n_test_positions": int(frame["position"].nunique()),
        "n_within_position_pairs": int(pair_count),
        "n_positions_with_pairs": int(positions_with_pairs),
        "n_positions_with_defined_spearman": int(len(within_spearman)),
    }
