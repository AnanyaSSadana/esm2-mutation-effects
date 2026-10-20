"""Deterministic nested subsampling for label-efficiency experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def nested_training_subsets_by_size(
    training_indices: Sequence[int] | np.ndarray,
    sizes: Sequence[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """Return deterministic, nested subsets with exact requested label counts."""

    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("training_indices must be a non-empty one-dimensional sequence")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("training_indices must be unique")

    requested_sizes = [int(size) for size in sizes]
    if not requested_sizes:
        raise ValueError("At least one subset size is required")
    if len(set(requested_sizes)) != len(requested_sizes):
        raise ValueError("Subset sizes must be unique")
    if any(size < 1 or size > len(indices) for size in requested_sizes):
        raise ValueError("Subset sizes must be between 1 and the number of training indices")

    shuffled = np.random.default_rng(seed).permutation(indices)
    return {size: shuffled[:size].copy() for size in sorted(requested_sizes)}


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

    values = [float(fraction) for fraction in fractions]
    if not values:
        raise ValueError("At least one training fraction is required")
    if len(set(values)) != len(values):
        raise ValueError("Training fractions must be unique")
    if any(not 0.0 < fraction <= 1.0 for fraction in values):
        raise ValueError("Training fractions must be greater than 0 and at most 1")

    indices = np.asarray(training_indices, dtype=np.int64)
    sizes = [min(len(indices), max(1, round(len(indices) * fraction))) for fraction in values]
    if len(set(sizes)) != len(sizes):
        raise ValueError("Training fractions resolve to duplicate subset sizes")
    by_size = nested_training_subsets_by_size(indices, sizes, seed)
    return {
        fraction: by_size[min(len(indices), max(1, round(len(indices) * fraction)))]
        for fraction in sorted(values)
    }
