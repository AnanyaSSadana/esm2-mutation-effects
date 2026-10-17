"""Deterministic train/validation/test split construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import mutation_table


def _assign(items: Sequence[object], seed: int) -> dict[object, str]:
    values = list(items)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    n_total = len(values)
    n_test = round(n_total * 0.10)
    n_validation = round(n_total * 0.10)
    labels = (
        ["test"] * n_test
        + ["validation"] * n_validation
        + ["train"] * (n_total - n_test - n_validation)
    )
    return dict(zip(values, labels))


def make_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    mutations = mutation_table(frame)

    random_mapping = _assign(mutations["mutant"].tolist(), seed)
    positions = sorted(mutations["position"].unique().tolist())
    position_mapping = _assign(positions, seed)

    result = mutations[["mutant", "position"]].rename(columns={"position": "residue_position"})
    result["random"] = result["mutant"].map(random_mapping)
    result["position"] = result["residue_position"].map(position_mapping)
    return result


def validate_splits(splits: pd.DataFrame, expected_mutants: set[str]) -> None:
    if set(splits["mutant"]) != expected_mutants or splits["mutant"].duplicated().any():
        raise ValueError("Split manifest does not contain every mutant exactly once")
    expected_labels = {"train", "validation", "test"}
    for strategy in ("random", "position"):
        if set(splits[strategy]) != expected_labels:
            raise ValueError(f"Unexpected labels for {strategy} split")
    if splits[["residue_position", "position"]].isna().any().any():
        raise ValueError("Split manifest contains missing residue positions")

    # Every substitution at the same residue must share one held-out label.
    if (splits.groupby("residue_position")["position"].nunique() != 1).any():
        raise ValueError("Position split leaks a residue across subsets")
    if (splits["residue_position"] < 1).any():
        raise ValueError("Positions must be one-based positive integers")
