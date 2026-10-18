"""Deterministic nested subsampling for label-efficiency experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def nested_training_subsets(
    training_indices: Sequence[int] | np.ndarray,
    fractions: Sequence[float],
    seed: int,
) -> dict[float, np.ndarray]:
    """Return deterministic, nested subsets of the available training rows.

    A single permutation is created for each seed. Each requested fraction takes
    a prefix of that permutation, which guarantees that smaller subsets are
    contained in larger ones.
    """

    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("training_indices must be a non-empty one-dimensional sequence")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("training_indices must be unique")

    values = [float(fraction) for fraction in fractions]
    if not values:
        raise ValueError("At least one training fraction is required")
    if len(set(values)) != len(values):
        raise ValueError("Training fractions must be unique")
    if any(not 0.0 < fraction <= 1.0 for fraction in values):
        raise ValueError("Training fractions must be greater than 0 and at most 1")

    shuffled = np.random.default_rng(seed).permutation(indices)
    subsets = {}
    for fraction in sorted(values):
        size = min(len(indices), max(1, round(len(indices) * fraction)))
        subsets[fraction] = shuffled[:size].copy()
    return subsets
