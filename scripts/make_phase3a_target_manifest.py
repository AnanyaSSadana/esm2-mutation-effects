"""Create the locked strict-budget TEM-1 manifest for Phase 3A."""

from __future__ import annotations

import argparse
from pathlib import Path

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.phase3_protocol import make_strict_target_manifest, manifest_sha256


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=142)
    parser.add_argument("--subset-seed", type=int, default=11)
    parser.add_argument("--label-budget", type=int, default=401)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "phase3a_pilot" / "target_manifest.csv",
    )
    args = parser.parse_args()
    manifest = make_strict_target_manifest(
        load_assay(),
        split_seed=args.split_seed,
        subset_seed=args.subset_seed,
        label_budget=args.label_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(
        {
            "path": str(args.output.resolve()),
            "manifest_sha256": manifest_sha256(manifest),
            "visible_target_labels": int(manifest.target_label_visible.sum()),
            "test_positions": int(manifest.loc[manifest.subset == "test", "position"].nunique()),
        }
    )


if __name__ == "__main__":
    main()
