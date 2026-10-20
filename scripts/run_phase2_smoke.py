#!/usr/bin/env python3
"""Run the validation-only, 20-label Phase 2 smoke check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.label_efficiency import (
    manifest_metadata,
    nested_mutation_manifests,
    write_manifest,
)

SPLIT_SEED = 42
SAMPLING_SEED = 1
LABEL_BUDGET = 20
TRAINING_SEED = 1
CONFIGURATION = ("lora_listmle", "lora", "listmle")
MAX_EPOCHS = 100
PATIENCE = 5
LISTS_PER_BATCH = 16
LIST_MICROBATCH_SIZE = 4


def run_directory(output_root: Path) -> Path:
    return (
        output_root
        / CONFIGURATION[0]
        / f"split_{SPLIT_SEED}"
        / f"sample_{SAMPLING_SEED}"
        / f"labels_{LABEL_BUDGET}"
    )


def build_command(*, device: str, output_root: Path, resume: bool) -> list[str]:
    directory = run_directory(output_root)
    command = [
        sys.executable,
        str(project_root() / "scripts" / "train_masked_adaptation.py"),
        "--backbone",
        CONFIGURATION[1],
        "--loss",
        CONFIGURATION[2],
        "--split-seed",
        str(SPLIT_SEED),
        "--training-seed",
        str(TRAINING_SEED),
        "--device",
        device,
        "--max-epochs",
        str(MAX_EPOCHS),
        "--patience",
        str(PATIENCE),
        "--lists-per-batch",
        str(LISTS_PER_BATCH),
        "--list-microbatch-size",
        str(LIST_MICROBATCH_SIZE),
        "--training-subset",
        str(directory / "subset_manifest.csv"),
        "--output-dir",
        str(directory),
    ]
    if resume:
        command.append("--resume")
    return command


def load_complete_record(output_root: Path, metadata: dict[str, object]) -> dict[str, object] | None:
    directory = run_directory(output_root)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    config = summary.get("config", {})
    expected = {
        "backbone_mode": "lora",
        "loss_name": "listmle",
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lists_per_batch": LISTS_PER_BATCH,
        "list_microbatch_size": LIST_MICROBATCH_SIZE,
        **metadata,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Phase 2 smoke configuration mismatch: {mismatches}")
    if summary.get("test_evaluated") is not False:
        raise RuntimeError("Phase 2 smoke run must not evaluate test positions")
    if summary.get("status") not in {"stopped_early", "completed_max_epochs"}:
        return None
    for forbidden in (directory / "test_predictions.csv", directory / "test_metrics.json"):
        if forbidden.exists():
            raise RuntimeError(f"Phase 2 smoke run contains forbidden test artifact: {forbidden}")
    return {
        "configuration": CONFIGURATION[0],
        "split_seed": SPLIT_SEED,
        "sampling_seed": SAMPLING_SEED,
        "n_labeled_train": metadata["n_labeled_train"],
        "training_subset_sha256": metadata["training_subset_sha256"],
        "best_epoch": summary["best_epoch"],
        "best_validation_spearman": summary["best_validation_spearman"],
        "status": summary["status"],
        "test_evaluated": False,
        "run_directory": str(directory.resolve()),
    }


def write_summary(output_root: Path, record: dict[str, object]) -> None:
    destination = output_root / "smoke_summary.json"
    payload = {
        "experiment": "phase_2_validation_only_20_label_smoke",
        "record": record,
        "test_evaluated": False,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "phase2_development",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests = nested_mutation_manifests(
        load_assay(),
        split_seed=SPLIT_SEED,
        sampling_seed=SAMPLING_SEED,
        budgets=(LABEL_BUDGET,),
    )
    directory = run_directory(args.output_root)
    metadata = write_manifest(directory / "subset_manifest.csv", manifests[LABEL_BUDGET])
    if metadata != manifest_metadata(manifests[LABEL_BUDGET]):
        raise RuntimeError("Saved manifest metadata differs from requested metadata")
    record = load_complete_record(args.output_root, metadata)
    if record is not None:
        write_summary(args.output_root, record)
        print(json.dumps({"status": "already_complete", **record}, indent=2, sort_keys=True))
        return

    resume = (directory / "config.json").exists()
    if resume and not (directory / "last.pt").exists():
        raise RuntimeError(f"No resumable checkpoint in {directory}")
    command = build_command(device=args.device, output_root=args.output_root, resume=resume)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=project_root(), check=True)
    record = load_complete_record(args.output_root, metadata)
    if record is None:
        raise RuntimeError("Phase 2 smoke run did not produce a complete summary")
    write_summary(args.output_root, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
