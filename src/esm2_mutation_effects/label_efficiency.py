"""Deterministic mutation manifests for label-efficiency experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .sampling import nested_training_subsets_by_size
from .splits import make_splits, validate_splits
from .training import training_subset_fingerprint


def combined_sampling_seed(split_seed: int, sampling_seed: int) -> int:
    """Derive an RNG seed from distinct outer-split and subset seeds."""

    sequence = np.random.SeedSequence([split_seed, sampling_seed])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def nested_mutation_manifests(
    assay: pd.DataFrame,
    *,
    split_seed: int,
    sampling_seed: int,
    budgets: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    """Build exact-size nested manifests containing only training-position mutants."""

    if "mutant" not in assay.columns:
        raise ValueError("Assay must contain a mutant column")
    if assay["mutant"].duplicated().any():
        raise ValueError("Assay mutants must be unique")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be a non-empty sequence of unique values")

    split = make_splits(assay, seed=split_seed)
    validate_splits(split, set(assay["mutant"]))
    labels = assay[["mutant"]].merge(
        split[["mutant", "position"]].rename(columns={"position": "split"}),
        on="mutant",
        validate="one_to_one",
    )
    training_rows = np.flatnonzero(labels["split"].to_numpy() == "train")
    subsets = nested_training_subsets_by_size(
        training_rows,
        budgets,
        seed=combined_sampling_seed(split_seed, sampling_seed),
    )
    return {
        budget: pd.DataFrame({"mutant": assay.iloc[rows]["mutant"].to_numpy()})
        for budget, rows in subsets.items()
    }


def manifest_metadata(manifest: pd.DataFrame) -> dict[str, object]:
    """Validate a manifest and return its immutable identity metadata."""

    if list(manifest.columns) != ["mutant"]:
        raise ValueError("Subset manifest must contain exactly one column named 'mutant'")
    mutants = manifest["mutant"].astype(str).tolist()
    return {
        "n_labeled_train": len(mutants),
        "training_subset_sha256": training_subset_fingerprint(mutants),
    }


def write_manifest(path: Path, manifest: pd.DataFrame) -> dict[str, object]:
    """Persist a manifest once; reject a conflicting existing manifest."""

    metadata = manifest_metadata(manifest)
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        if manifest_metadata(existing) != metadata:
            raise RuntimeError(f"Existing manifest does not match the requested subset: {path}")
        return metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(path)
    return metadata


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using a stable serialization."""

    import json

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
