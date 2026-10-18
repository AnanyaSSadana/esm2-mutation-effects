"""Shared supervised baseline definitions and validation-based selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import spearman

MUTATION_FEATURE_COLUMNS = ("wild_type", "alternate", "position_fraction")
MUTATION_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FROZEN_ESM2_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)
POSITION_CONTROL_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)


def make_mutation_feature_model(alpha: float) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                "amino_acids",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["wild_type", "alternate"],
            ),
            ("position", StandardScaler(), ["position_fraction"]),
        ]
    )
    return Pipeline([("features", features), ("regressor", Ridge(alpha=alpha))])


def make_frozen_esm2_model(alpha: float) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10_000),
    )


def make_categorical_position_model(alpha: float) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["wild_type", "alternate", "position"],
            )
        ]
    )
    return Pipeline([("features", features), ("regressor", Ridge(alpha=alpha))])


def select_alpha(
    model_factory: Callable[[float], Any],
    alphas: Sequence[float],
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Choose alpha using validation Spearman correlation only."""

    candidates = []
    for alpha in alphas:
        model = model_factory(float(alpha))
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_spearman": spearman(y_validation, prediction),
            }
        )
    selected = max(candidates, key=lambda item: item["validation_spearman"])
    return selected, candidates
