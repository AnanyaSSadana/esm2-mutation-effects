"""Cache masked-marginal ESM-2 scores for Phase 3A pilot assays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import (
    default_assay_path,
    default_reference_path,
    load_assay_metadata,
    project_root,
)
from esm2_mutation_effects.esm_utils import choose_device, load_model
from esm2_mutation_effects.phase3_pilot import PILOT_TASKS
from esm2_mutation_effects.phase3_zero_shot import score_masked_marginals


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--positions-per-batch", type=int, default=16)
    parser.add_argument("--metadata", type=Path, default=default_reference_path())
    parser.add_argument("--auxiliary-raw-dir", type=Path, default=root / "data" / "phase3" / "raw")
    parser.add_argument("--target-assay", type=Path, default=default_assay_path())
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "phase3a_pilot" / "zero_shot"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.positions_per_batch < 1:
        raise ValueError("--positions-per-batch must be positive")
    metadata = pd.read_csv(args.metadata).set_index("DMS_id", verify_integrity=True)
    target_metadata = load_assay_metadata()
    inputs = [("BLAT_ECOLX_Stiffler_2015", args.target_assay, target_metadata, "target")]
    for task in PILOT_TASKS:
        path = args.auxiliary_raw_dir / f"{task.dms_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Pilot raw table is missing: {path}")
        inputs.append((task.dms_id, path, metadata.loc[task.dms_id].to_dict(), task.role))

    device = choose_device(args.device)
    model, alphabet = load_model(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for assay_id, path, assay_metadata, role in inputs:
        output = args.output_dir / f"{assay_id}.csv"
        if output.exists() and not args.overwrite:
            predicted = pd.read_csv(output)
            if set(predicted.columns) >= {"mutant", "DMS_score", "predicted_score"}:
                summaries.append({"DMS_id": assay_id, "role": role, "status": "reused", "rows": len(predicted)})
                continue
            raise ValueError(f"Existing zero-shot file is malformed: {output}")
        scored = score_masked_marginals(
            pd.read_csv(path),
            str(assay_metadata["target_seq"]),
            model,
            alphabet,
            device,
            positions_per_batch=args.positions_per_batch,
        )
        scored.to_csv(output, index=False)
        summaries.append({"DMS_id": assay_id, "role": role, "status": "scored", "rows": len(scored)})
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    print(json.dumps({"device": str(device), "assays": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
