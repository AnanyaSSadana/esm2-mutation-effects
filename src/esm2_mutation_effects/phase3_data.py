"""Raw-table integrity validation for Phase 3A ProteinGym pilot tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import parse_mutation
from .phase3_pilot import PilotDataRequirements


def validate_pilot_assay(
    frame: pd.DataFrame,
    metadata: dict[str, object],
    *,
    requirements: PilotDataRequirements | None = None,
) -> dict[str, Any]:
    """Validate one downloaded ProteinGym substitution table at row level.

    ProteinGym's processed ``DMS_score`` is direction-normalized; the caller
    records that provenance separately. This function confirms the mechanical
    conditions that make the table safe for one-assay ranking losses.
    """

    requirements = requirements or PilotDataRequirements()
    requirements.validate()
    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Assay table is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Assay table contains missing required values")
    if frame["mutant"].duplicated().any():
        raise ValueError("Assay table contains duplicate mutation identifiers")
    if bool(metadata["includes_multiple_mutants"]):
        raise ValueError("Metadata declares multiple mutants")

    target = str(metadata["target_seq"]).upper()
    expected_length = int(metadata["seq_len"])
    if len(target) != expected_length:
        raise ValueError("Metadata target sequence length is inconsistent")
    if expected_length > requirements.maximum_sequence_length:
        raise ValueError("Target sequence exceeds ESM-2 input-length requirement")
    scores = pd.to_numeric(frame["DMS_score"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("DMS_score contains non-finite values")

    positions: list[int] = []
    for row in frame.itertuples(index=False):
        mutation = parse_mutation(str(row.mutant))
        site = mutation.position - 1
        if not 0 <= site < expected_length:
            raise ValueError(f"Out-of-range mutation position in {row.mutant}")
        if target[site] != mutation.wild_type:
            raise ValueError(f"Wild-type residue mismatch for {row.mutant}")
        mutated_sequence = str(row.mutated_sequence).upper()
        if len(mutated_sequence) != expected_length:
            raise ValueError(f"Sequence length mismatch for {row.mutant}")
        differences = [
            index
            for index, (reference, observed) in enumerate(zip(target, mutated_sequence))
            if reference != observed
        ]
        if differences != [site] or mutated_sequence[site] != mutation.alternate:
            raise ValueError(f"mutated_sequence is inconsistent with {row.mutant}")
        positions.append(mutation.position)

    unique_positions = len(set(positions))
    if len(frame) < requirements.minimum_valid_single_substitutions:
        raise ValueError("Assay has too few valid single substitutions")
    if unique_positions < requirements.minimum_mutated_positions:
        raise ValueError("Assay has too few distinct mutated positions")
    distinct_score_values = len(np.unique(scores))
    if distinct_score_values < requirements.list_size:
        raise ValueError("Assay has too few distinct score values for ListMLE")
    return {
        "assay_id": str(metadata["DMS_id"]),
        "valid_single_substitutions": len(frame),
        "unique_mutated_positions": unique_positions,
        "mutated_position_coverage": unique_positions / expected_length,
        "position_min": min(positions),
        "position_max": max(positions),
        "sequence_length": expected_length,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "distinct_dms_score_values": distinct_score_values,
        "dms_score_direction": "higher_is_better_by_ProteinGym_processed_score_convention",
    }
