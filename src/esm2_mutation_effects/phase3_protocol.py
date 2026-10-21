"""Strict target-label manifests and protocol checks for the Phase 3A pilot."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .data import mutation_table


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_sha256(manifest: pd.DataFrame) -> str:
    required = {"mutant", "position", "subset", "target_label_visible"}
    if set(manifest.columns) != required:
        raise ValueError(f"Unexpected manifest columns: {list(manifest.columns)}")
    records = manifest.sort_values("mutant", kind="stable").to_dict(orient="records")
    return canonical_sha256(records)


def make_strict_target_manifest(
    assay: pd.DataFrame,
    *,
    split_seed: int = 142,
    subset_seed: int = 11,
    label_budget: int = 401,
) -> pd.DataFrame:
    """Hold out whole positions and expose exactly ``label_budget`` train labels."""

    mutations = mutation_table(assay)
    positions = np.asarray(sorted(mutations["position"].unique()), dtype=np.int64)
    test_count = round(len(positions) * 0.10)
    if test_count < 1 or test_count >= len(positions):
        raise ValueError("Position split cannot form both adaptation and test partitions")
    shuffled_positions = np.random.default_rng(split_seed).permutation(positions)
    test_positions = set(shuffled_positions[:test_count].tolist())
    result = mutations.copy()
    result["subset"] = np.where(result["position"].isin(test_positions), "test", "adaptation")
    candidate_rows = np.flatnonzero(result["subset"].to_numpy() == "adaptation")
    if label_budget < 1 or label_budget > len(candidate_rows):
        raise ValueError("label_budget must fit within adaptation-candidate mutations")
    visible_rows = np.random.default_rng(subset_seed).permutation(candidate_rows)[:label_budget]
    visible = np.zeros(len(result), dtype=bool)
    visible[visible_rows] = True
    result["target_label_visible"] = visible
    result = result[["mutant", "position", "subset", "target_label_visible"]]
    validate_strict_target_manifest(result, label_budget=label_budget)
    return result


def validate_strict_target_manifest(manifest: pd.DataFrame, *, label_budget: int) -> None:
    required = ["mutant", "position", "subset", "target_label_visible"]
    if list(manifest.columns) != required:
        raise ValueError(f"Manifest must have columns in this order: {required}")
    if manifest["mutant"].duplicated().any() or manifest[required].isna().any().any():
        raise ValueError("Manifest contains duplicate mutants or missing values")
    if set(manifest["subset"]) != {"adaptation", "test"}:
        raise ValueError("Manifest must contain adaptation and test rows")
    if (manifest.groupby("position")["subset"].nunique() != 1).any():
        raise ValueError("Position leakage between adaptation and test")
    if manifest.loc[manifest["subset"] == "test", "target_label_visible"].any():
        raise ValueError("A target-test label is marked visible")
    visible_count = int(manifest["target_label_visible"].sum())
    if visible_count != label_budget:
        raise ValueError(f"Expected {label_budget} visible labels; found {visible_count}")
    if not (manifest.loc[manifest["target_label_visible"], "subset"] == "adaptation").all():
        raise ValueError("Visible labels must be adaptation mutations")
