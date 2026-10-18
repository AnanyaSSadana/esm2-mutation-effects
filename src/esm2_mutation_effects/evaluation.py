"""Prediction validation and evaluation-subset selection."""

from __future__ import annotations

import pandas as pd

SPLIT_STRATEGIES = ("random", "position")
SUBSETS = ("train", "validation", "test")


def select_evaluation_subset(
    predictions: pd.DataFrame,
    score_column: str,
    *,
    split_manifest: pd.DataFrame | None = None,
    split_strategy: str | None = None,
    subset: str | None = None,
) -> pd.DataFrame:
    """Validate predictions and optionally retain one named split subset."""

    required = {"mutant", score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions["mutant"].duplicated().any():
        raise ValueError("Prediction file contains duplicate mutants")
    if predictions[score_column].isna().any():
        raise ValueError(f"Prediction column {score_column!r} contains missing values")

    filtering_requested = split_strategy is not None or subset is not None
    if filtering_requested and (split_strategy is None or subset is None):
        raise ValueError("--split-strategy and --subset must be provided together")
    if not filtering_requested:
        return predictions[["mutant", score_column]].copy()

    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy: {split_strategy!r}")
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset!r}")
    if split_manifest is None:
        raise ValueError("A split manifest is required for subset evaluation")

    split_required = {"mutant", split_strategy}
    split_missing = split_required.difference(split_manifest.columns)
    if split_missing:
        raise ValueError(f"Missing split-manifest columns: {sorted(split_missing)}")
    if split_manifest["mutant"].duplicated().any():
        raise ValueError("Split manifest contains duplicate mutants")

    selected = predictions[["mutant", score_column]].merge(
        split_manifest[["mutant", split_strategy]],
        on="mutant",
        how="left",
        validate="one_to_one",
    )
    if selected[split_strategy].isna().any():
        unknown = selected.loc[selected[split_strategy].isna(), "mutant"].tolist()[:5]
        raise ValueError(f"Predictions are missing from the split manifest, for example: {unknown}")
    selected = selected.loc[selected[split_strategy] == subset, ["mutant", score_column]]
    if selected.empty:
        raise ValueError(f"No predictions remain after selecting {split_strategy}:{subset}")
    return selected.reset_index(drop=True)
