"""Dataset loading and biological-integrity checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MUTATION_RE = re.compile(rf"^([{AMINO_ACIDS}])(\d+)([{AMINO_ACIDS}])$")
ASSAY_ID = "BLAT_ECOLX_Stiffler_2015"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_assay_path() -> Path:
    return project_root() / "data" / "raw" / f"{ASSAY_ID}.csv"


def default_reference_path() -> Path:
    return project_root() / "data" / "provenance" / "DMS_substitutions.csv"


@dataclass(frozen=True)
class Mutation:
    wild_type: str
    position: int
    alternate: str


def parse_mutation(value: str) -> Mutation:
    match = MUTATION_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Expected a single substitution such as H24C, got {value!r}")
    wild_type, position, alternate = match.groups()
    if wild_type == alternate:
        raise ValueError(f"Mutation does not change the residue: {value!r}")
    return Mutation(wild_type, int(position), alternate)


def load_assay_metadata(path: Path | None = None) -> dict[str, object]:
    reference_path = path or default_reference_path()
    reference = pd.read_csv(reference_path)
    row = reference.loc[reference["DMS_id"] == ASSAY_ID]
    if len(row) != 1:
        raise ValueError(f"Expected one metadata row for {ASSAY_ID}; found {len(row)}")
    return row.iloc[0].to_dict()


def load_assay(path: Path | None = None, validate: bool = True) -> pd.DataFrame:
    assay_path = path or default_assay_path()
    frame = pd.read_csv(assay_path)
    if validate:
        validate_assay(frame, load_assay_metadata())
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assay(frame: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    expected_columns = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing assay columns: {sorted(missing)}")
    if frame[list(expected_columns)].isna().any().any():
        raise ValueError("Assay contains missing values")
    if frame["mutant"].duplicated().any():
        raise ValueError("Assay contains duplicate mutant identifiers")

    target = str(metadata["target_seq"])
    sequence_length = int(metadata["seq_len"])
    expected_rows = int(metadata["DMS_total_number_mutants"])
    if len(target) != sequence_length:
        raise ValueError("Metadata target sequence length is inconsistent")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} variants; found {len(frame)}")

    positions: list[int] = []
    for row in frame.itertuples(index=False):
        mutation = parse_mutation(row.mutant)
        index = mutation.position - 1
        if not 0 <= index < sequence_length:
            raise ValueError(f"Out-of-range position in {row.mutant}")
        if target[index] != mutation.wild_type:
            raise ValueError(
                f"Wild-type residue mismatch for {row.mutant}: target has {target[index]}"
            )
        mutant_sequence = str(row.mutated_sequence)
        if len(mutant_sequence) != sequence_length:
            raise ValueError(f"Sequence length mismatch for {row.mutant}")
        differences = [i for i, (a, b) in enumerate(zip(target, mutant_sequence)) if a != b]
        if differences != [index] or mutant_sequence[index] != mutation.alternate:
            raise ValueError(f"mutated_sequence is inconsistent with {row.mutant}")
        positions.append(mutation.position)

    scores = pd.to_numeric(frame["DMS_score"], errors="raise")
    return {
        "assay_id": ASSAY_ID,
        "variants": len(frame),
        "unique_positions": len(set(positions)),
        "position_min": min(positions),
        "position_max": max(positions),
        "sequence_length": sequence_length,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "higher_is_better": True,
    }


def mutation_table(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_mutation(value) for value in frame["mutant"]]
    return pd.DataFrame(
        {
            "mutant": frame["mutant"].to_numpy(),
            "wild_type": [item.wild_type for item in parsed],
            "position": [item.position for item in parsed],
            "alternate": [item.alternate for item in parsed],
        }
    )
