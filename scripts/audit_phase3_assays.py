"""Create a metadata-only Phase 3 ProteinGym assay-audit report.

This command opens only the checked-in authoritative substitutions metadata.
It never downloads assay score tables, trains a model, or opens a target test
partition. The output is an eligibility preflight, not a frozen cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import default_reference_path, project_root, sha256
from esm2_mutation_effects.phase3_audit import AuditThresholds, audit_substitution_metadata


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=default_reference_path())
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "phase3_assay_audit"
    )
    parser.add_argument("--minimum-single-substitutions", type=int, default=500)
    parser.add_argument("--maximum-sequence-length", type=int, default=1022)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    thresholds = AuditThresholds(
        minimum_single_substitutions=args.minimum_single_substitutions,
        maximum_sequence_length=args.maximum_sequence_length,
    )
    metadata = pd.read_csv(args.metadata)
    registry, summary = audit_substitution_metadata(metadata, thresholds=thresholds)
    summary["metadata_path"] = str(args.metadata.resolve())
    summary["metadata_sha256"] = sha256(args.metadata)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_dir / "metadata_registry.csv", index=False)
    (output_dir / "metadata_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote registry: {(output_dir / 'metadata_registry.csv').resolve()}")


if __name__ == "__main__":
    main()
