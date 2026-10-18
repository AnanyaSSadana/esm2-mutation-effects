"""Feature validation and selection for supervised ESM-2 baselines."""

from __future__ import annotations

import numpy as np

CACHED_FEATURE_DEFINITION = "concat(wild_type_site,mutant_site,mutant_minus_wild_type)"
MODEL_FEATURE_DEFINITION = "concat(wild_type_site,mutant_minus_wild_type)"


def select_nonredundant_site_features(features: np.ndarray) -> np.ndarray:
    """Return wild-type and delta blocks as finite float64 model features.

    The cache contains three equally sized blocks: wild type, mutant, and
    mutant minus wild type. Using all three in a linear model is exactly
    redundant, so the mutant block is omitted without losing information.
    """

    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix; got shape {values.shape}")
    if values.shape[1] == 0 or values.shape[1] % 3 != 0:
        raise ValueError("Expected concatenated wild-type, mutant, and delta blocks of equal width")
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains non-finite values")

    block_width = values.shape[1] // 3
    wild_type = values[:, :block_width]
    delta = values[:, 2 * block_width :]
    return np.concatenate([wild_type, delta], axis=1).astype(np.float64, copy=False)
