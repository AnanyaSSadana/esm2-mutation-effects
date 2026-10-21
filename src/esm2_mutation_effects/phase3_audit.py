"""Metadata-level safeguards for Phase 3 ProteinGym task selection.

This module deliberately does *not* select a final meta-learning cohort.  The
authoritative metadata can rule out clearly unsuitable assays, group related
assays, and identify likely TEM-1 homologs before a download.  Final inclusion
also requires row-level checks of the downloaded substitution tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .data import AMINO_ACIDS, ASSAY_ID

METADATA_REQUIRED_COLUMNS = frozenset(
    {
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "target_seq",
        "seq_len",
        "includes_multiple_mutants",
        "DMS_number_single_mutants",
        "raw_DMS_directionality",
        "molecule_name",
        "title",
    }
)


@dataclass(frozen=True)
class AuditThresholds:
    """Outcome-blind Phase 3 metadata-screen thresholds."""

    minimum_single_substitutions: int = 500
    maximum_sequence_length: int = 1022
    tem_identity_threshold: float = 0.30
    tem_coverage_threshold: float = 0.70
    smith_waterman_match_score: int = 2
    smith_waterman_mismatch_score: int = -1
    smith_waterman_gap_score: int = -2

    def validate(self) -> None:
        if self.minimum_single_substitutions < 1:
            raise ValueError("minimum_single_substitutions must be positive")
        if self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")
        for name, value in (
            ("tem_identity_threshold", self.tem_identity_threshold),
            ("tem_coverage_threshold", self.tem_coverage_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.smith_waterman_match_score <= 0:
            raise ValueError("smith_waterman_match_score must be positive")
        if self.smith_waterman_mismatch_score >= 0:
            raise ValueError("smith_waterman_mismatch_score must be negative")
        if self.smith_waterman_gap_score >= 0:
            raise ValueError("smith_waterman_gap_score must be negative")


@dataclass(frozen=True)
class LocalAlignment:
    """Summary of a deterministic Smith-Waterman local alignment."""

    score: int
    aligned_pairs: int
    identical_pairs: int
    reference_residues_aligned: int
    candidate_residues_aligned: int
    reference_length: int

    @property
    def identity(self) -> float:
        return self.identical_pairs / self.aligned_pairs if self.aligned_pairs else 0.0

    @property
    def reference_coverage(self) -> float:
        if self.reference_length < 1:
            raise ValueError("reference_length must be positive")
        return self.reference_residues_aligned / self.reference_length


def sequence_sha256(sequence: str) -> str:
    """Return a canonical checksum for an uppercase amino-acid sequence."""

    return sha256(sequence.encode("ascii")).hexdigest()


def smith_waterman_local_alignment(
    reference: str,
    candidate: str,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> LocalAlignment:
    """Compute a small, dependency-free local-alignment screen.

    The output is used only for the pre-specified TEM-1 exclusion screen. It
    is not a substitute for evolutionary-homology annotation or a BLAST-based
    analysis. ``reference`` is TEM-1, so coverage is relative to all 286 TEM-1
    residues as required by the Phase 3 design.
    """

    if not reference or not candidate:
        raise ValueError("Sequences must be non-empty")
    if min(match_score, -mismatch_score, -gap_score) < 1:
        raise ValueError("Alignment scores must reward matches and penalize mismatches and gaps")

    n_reference = len(reference)
    n_candidate = len(candidate)
    scores = np.zeros((n_reference + 1, n_candidate + 1), dtype=np.int32)
    trace = np.zeros((n_reference + 1, n_candidate + 1), dtype=np.uint8)
    # Trace values: 0 stop, 1 diagonal, 2 consume reference, 3 consume candidate.
    maximum_score = 0
    maximum_i = 0
    maximum_j = 0
    for i in range(1, n_reference + 1):
        for j in range(1, n_candidate + 1):
            diagonal = scores[i - 1, j - 1] + (
                match_score if reference[i - 1] == candidate[j - 1] else mismatch_score
            )
            consume_reference = scores[i - 1, j] + gap_score
            consume_candidate = scores[i, j - 1] + gap_score
            best = max(0, int(diagonal), int(consume_reference), int(consume_candidate))
            scores[i, j] = best
            if best == 0:
                direction = 0
            elif best == diagonal:
                direction = 1
            elif best == consume_reference:
                direction = 2
            else:
                direction = 3
            trace[i, j] = direction
            if best > maximum_score:
                maximum_score = best
                maximum_i = i
                maximum_j = j

    aligned_pairs = 0
    identical_pairs = 0
    reference_residues_aligned = 0
    candidate_residues_aligned = 0
    i, j = maximum_i, maximum_j
    while i > 0 and j > 0 and trace[i, j] != 0:
        direction = int(trace[i, j])
        if direction == 1:
            aligned_pairs += 1
            reference_residues_aligned += 1
            candidate_residues_aligned += 1
            identical_pairs += int(reference[i - 1] == candidate[j - 1])
            i -= 1
            j -= 1
        elif direction == 2:
            reference_residues_aligned += 1
            i -= 1
        else:
            candidate_residues_aligned += 1
            j -= 1

    return LocalAlignment(
        score=maximum_score,
        aligned_pairs=aligned_pairs,
        identical_pairs=identical_pairs,
        reference_residues_aligned=reference_residues_aligned,
        candidate_residues_aligned=candidate_residues_aligned,
        reference_length=n_reference,
    )


def _normalized_sequence(value: object) -> str | None:
    if pd.isna(value):
        return None
    sequence = str(value).strip().upper()
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        return None
    return sequence


def _protein_group_key(uniprot_id: object, sequence: str) -> str:
    """Group exact sequence duplicates even when a UniProt identifier is absent."""

    if not pd.isna(uniprot_id) and str(uniprot_id).strip():
        return f"uniprot:{str(uniprot_id).strip()}"
    return f"sequence:{sequence_sha256(sequence)}"


def _is_tem_family_annotation(row: pd.Series) -> bool:
    """Conservatively flag explicitly beta-lactamase/TEM-labelled metadata."""

    text = " ".join(
        str(row.get(column, ""))
        for column in ("DMS_id", "UniProt_ID", "molecule_name", "title", "selection_assay")
        if not pd.isna(row.get(column, ""))
    ).lower()
    return "beta-lactamase" in text or "β-lactamase" in text or "blat_" in text


def _metadata_failure_reason(row: pd.Series, thresholds: AuditThresholds) -> str | None:
    if bool(row["includes_multiple_mutants"]):
        return "metadata_includes_multiple_mutants"
    try:
        n_single = int(row["DMS_number_single_mutants"])
    except (TypeError, ValueError):
        return "metadata_invalid_single_mutant_count"
    if n_single < thresholds.minimum_single_substitutions:
        return "metadata_too_few_single_substitutions"
    try:
        length = int(row["seq_len"])
    except (TypeError, ValueError):
        return "metadata_invalid_sequence_length"
    if length > thresholds.maximum_sequence_length:
        return "metadata_sequence_too_long"
    sequence = _normalized_sequence(row["target_seq"])
    if sequence is None:
        return "metadata_invalid_target_sequence"
    if len(sequence) != length:
        return "metadata_sequence_length_mismatch"
    try:
        direction = float(row["raw_DMS_directionality"])
    except (TypeError, ValueError):
        return "metadata_missing_score_direction"
    if direction not in {-1.0, 1.0}:
        return "metadata_invalid_score_direction"
    return None


def audit_substitution_metadata(
    metadata: pd.DataFrame,
    *,
    target_assay_id: str = ASSAY_ID,
    thresholds: AuditThresholds | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit ProteinGym substitution metadata without opening assay score tables.

    Rows that pass this function are *candidates*, not final tasks. Every such
    row remains ``requires_raw_integrity`` until a future download verifies
    mutation notation, row-level uniqueness, sequence consistency, finite
    direction-normalized scores, and residue-position coverage.
    """

    thresholds = thresholds or AuditThresholds()
    thresholds.validate()
    missing = METADATA_REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if metadata["DMS_id"].isna().any() or metadata["DMS_id"].duplicated().any():
        raise ValueError("DMS_id must be present and unique")

    target_rows = metadata.loc[metadata["DMS_id"] == target_assay_id]
    if len(target_rows) != 1:
        raise ValueError(f"Expected exactly one target assay {target_assay_id!r}")
    target = target_rows.iloc[0]
    target_sequence = _normalized_sequence(target["target_seq"])
    if target_sequence is None:
        raise ValueError("Target assay has an invalid target sequence")
    target_uniprot = str(target["UniProt_ID"])

    records: list[dict[str, Any]] = []
    for _, row in metadata.sort_values("DMS_id", kind="stable").iterrows():
        assay_id = str(row["DMS_id"])
        sequence = _normalized_sequence(row["target_seq"])
        reason = _metadata_failure_reason(row, thresholds)
        exact_target_sequence = sequence == target_sequence
        same_uniprot = str(row["UniProt_ID"]) == target_uniprot
        tem_annotation = _is_tem_family_annotation(row)
        # The dynamic-programming screen is only meaningful for otherwise
        # admissible, non-TEM candidates. Known exclusions do not need an
        # expensive alignment merely to confirm an exclusion already recorded
        # from authoritative metadata.
        alignment = (
            smith_waterman_local_alignment(
                target_sequence,
                sequence,
                match_score=thresholds.smith_waterman_match_score,
                mismatch_score=thresholds.smith_waterman_mismatch_score,
                gap_score=thresholds.smith_waterman_gap_score,
            )
            if (
                sequence is not None
                and reason is None
                and not exact_target_sequence
                and not same_uniprot
                and not tem_annotation
            )
            else None
        )
        identity_screen = bool(
            alignment
            and alignment.identity >= thresholds.tem_identity_threshold
            and alignment.reference_coverage >= thresholds.tem_coverage_threshold
        )
        target_role = "non_target"
        if assay_id == target_assay_id:
            target_role = "target"
        elif exact_target_sequence or same_uniprot:
            target_role = "same_protein_positive_control"
        elif tem_annotation or identity_screen:
            target_role = "tem_related_excluded_from_strict"

        records.append(
            {
                "DMS_id": assay_id,
                "DMS_filename": str(row["DMS_filename"]),
                "UniProt_ID": str(row["UniProt_ID"]),
                "protein_group": _protein_group_key(row["UniProt_ID"], sequence)
                if sequence is not None
                else None,
                "target_sequence_sha256": sequence_sha256(sequence) if sequence is not None else None,
                "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                "metadata_single_substitutions": (
                    int(row["DMS_number_single_mutants"])
                    if pd.notna(row["DMS_number_single_mutants"])
                    else None
                ),
                "raw_score_direction": (
                    float(row["raw_DMS_directionality"])
                    if pd.notna(row["raw_DMS_directionality"])
                    else None
                ),
                "molecule_name": str(row.get("molecule_name", "")),
                "selection_type": str(row.get("selection_type", "")),
                "coarse_selection_type": str(row.get("coarse_selection_type", "")),
                "metadata_eligibility": "eligible" if reason is None else "excluded",
                "metadata_reason": reason,
                "target_role": target_role,
                "same_target_sequence": exact_target_sequence,
                "same_uniprot_as_tem1": same_uniprot,
                "tem_family_metadata_flag": tem_annotation,
                "tem_local_alignment_identity": alignment.identity if alignment else None,
                "tem_local_alignment_coverage": alignment.reference_coverage if alignment else None,
                "tem_similarity_screen_flag": identity_screen,
                "raw_integrity_status": "requires_download_and_row_level_validation"
                if reason is None
                else "not_applicable",
                "final_cohort_status": "not_selected",
            }
        )

    registry = pd.DataFrame.from_records(records)
    eligible = registry.loc[registry["metadata_eligibility"] == "eligible"]
    strict_candidates = eligible.loc[eligible["target_role"] == "non_target"]
    summary: dict[str, Any] = {
        "audit_name": "phase3_proteingym_metadata_preflight",
        "status": "metadata_only_not_a_final_cohort",
        "target_assay_id": target_assay_id,
        "metadata_rows": len(registry),
        "metadata_eligible_assays": len(eligible),
        "metadata_eligible_protein_groups": int(eligible["protein_group"].nunique()),
        "strict_metadata_candidates": len(strict_candidates),
        "strict_metadata_candidate_protein_groups": int(strict_candidates["protein_group"].nunique()),
        "same_protein_positive_controls": int(
            (registry["target_role"] == "same_protein_positive_control").sum()
        ),
        "tem_related_excluded_from_strict": int(
            (registry["target_role"] == "tem_related_excluded_from_strict").sum()
        ),
        "thresholds": asdict(thresholds),
        "required_before_final_cohort": [
            "download each candidate from the authoritative ProteinGym substitutions release",
            "validate single-substitution mutation notation and mutated-sequence consistency",
            "verify finite direction-normalized DMS_score values and documented higher-is-better use",
            "measure valid unique substitutions and mutated-position coverage from rows",
            "select one raw-integrity-passing assay per protein group by the locked tie-break rule",
            "record source archive and file checksums, license, biological annotation, and task partition",
        ],
    }
    return registry, summary
