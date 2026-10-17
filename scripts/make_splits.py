#!/usr/bin/env python3
"""Create deterministic random and position-held-out splits."""

import argparse
import json
from pathlib import Path

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.splits import make_splits, validate_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=project_root() / "data" / "splits" / "splits.csv"
    )
    args = parser.parse_args()

    assay = load_assay()
    splits = make_splits(assay, seed=args.seed)
    validate_splits(splits, set(assay["mutant"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)

    summary = {
        "seed": args.seed,
        "output": str(args.output),
        "random": splits["random"].value_counts().sort_index().to_dict(),
        "position": splits["position"].value_counts().sort_index().to_dict(),
        "position_unique_residues": (
            splits.groupby("position")["residue_position"].nunique().sort_index().to_dict()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
