#!/usr/bin/env python3
"""Benchmark Phase 2 LoRA ListMLE microbatch sizes without test evaluation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.label_efficiency import (
    manifest_metadata,
    nested_mutation_manifests,
    write_manifest,
)

DEFAULT_MICROBATCH_SIZES = (4, 8, 16)
SPLIT_SEED = 42
SAMPLING_SEED = 1
LABEL_BUDGET = 2004
TRAINING_SEED = 1
LISTS_PER_BATCH = 16
MAX_EPOCHS = 1
PATIENCE = 1


def run_directory(
    output_root: Path,
    microbatch_size: int,
    *,
    split_seed: int,
    sampling_seed: int,
    label_budget: int,
    training_seed: int,
) -> Path:
    return (
        output_root
        / f"microbatch_{microbatch_size}"
        / f"split_{split_seed}"
        / f"sample_{sampling_seed}"
        / f"labels_{label_budget}"
        / f"seed_{training_seed}"
    )


def build_command(
    microbatch_size: int,
    *,
    device: str,
    split_seed: int,
    sampling_seed: int,
    label_budget: int,
    training_seed: int,
    output_root: Path,
    training_subset: Path,
    resume: bool,
) -> list[str]:
    output_dir = run_directory(
        output_root,
        microbatch_size,
        split_seed=split_seed,
        sampling_seed=sampling_seed,
        label_budget=label_budget,
        training_seed=training_seed,
    )
    command = [
        sys.executable,
        str(project_root() / "scripts" / "train_masked_adaptation.py"),
        "--backbone",
        "lora",
        "--loss",
        "listmle",
        "--split-seed",
        str(split_seed),
        "--training-seed",
        str(training_seed),
        "--device",
        device,
        "--max-epochs",
        str(MAX_EPOCHS),
        "--patience",
        str(PATIENCE),
        "--lists-per-batch",
        str(LISTS_PER_BATCH),
        "--list-microbatch-size",
        str(microbatch_size),
        "--training-subset",
        str(training_subset),
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def load_completed_result(
    run_dir: Path,
    *,
    microbatch_size: int,
    split_seed: int,
    training_seed: int,
    metadata: dict[str, object],
) -> dict[str, object] | None:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.json"
    if not summary_path.exists() or not history_path.exists():
        return None

    summary = json.loads(summary_path.read_text())
    config = summary.get("config", {})
    expected = {
        "backbone_mode": "lora",
        "loss_name": "listmle",
        "split_seed": split_seed,
        "training_seed": training_seed,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lists_per_batch": LISTS_PER_BATCH,
        "list_microbatch_size": microbatch_size,
        "ranking_target_representation": "ordinal_rank_from_float64",
        **metadata,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Benchmark configuration mismatch in {run_dir}: {mismatches}")
    if summary.get("test_evaluated") is not False:
        raise RuntimeError("Batching benchmark unexpectedly evaluated test positions")
    for forbidden in (run_dir / "test_predictions.csv", run_dir / "test_metrics.json"):
        if forbidden.exists():
            raise RuntimeError(f"Batching benchmark contains forbidden test artifact: {forbidden}")

    history = json.loads(history_path.read_text())
    epochs = [record for record in history if record.get("epoch") == 1]
    if len(epochs) != 1:
        return None
    epoch = epochs[0]
    numeric_values = (
        epoch.get("epoch_seconds"),
        epoch.get("train_loss"),
        epoch.get("validation_metrics", {}).get("overall_spearman"),
    )
    if any(value is None or not math.isfinite(float(value)) for value in numeric_values):
        raise RuntimeError(f"Benchmark produced a missing or non-finite metric in {run_dir}")
    return {
        "microbatch_size": microbatch_size,
        "status": "completed",
        "epoch_seconds": epoch["epoch_seconds"],
        "reported_allocated_memory_mib_at_epoch_end": epoch["allocated_memory_mib"],
        "train_loss": epoch["train_loss"],
        "validation_spearman": epoch["validation_metrics"]["overall_spearman"],
        "maximum_gradient_norm": epoch["maximum_gradient_norm"],
        "maximum_lora_gradient_norm": epoch["maximum_lora_gradient_norm"],
        "test_evaluated": False,
        "run_directory": str(run_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LoRA ListMLE microbatch sizes on the same Phase 2 labelled subset. "
            "Each candidate runs one validation-only epoch and never evaluates test positions."
        )
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--sampling-seed", type=int, default=SAMPLING_SEED)
    parser.add_argument("--label-budget", type=int, default=LABEL_BUDGET)
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument(
        "--microbatch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MICROBATCH_SIZES),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "phase2_microbatch_benchmark",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.microbatch_sizes)) != len(args.microbatch_sizes):
        raise ValueError("Microbatch sizes must be unique")
    if any(size < 1 or size > LISTS_PER_BATCH for size in args.microbatch_sizes):
        raise ValueError(
            f"Microbatch sizes must be between 1 and the effective batch size of {LISTS_PER_BATCH}"
        )
    if args.label_budget < 2:
        raise ValueError("Label budget must be at least two")

    manifest = nested_mutation_manifests(
        load_assay(),
        split_seed=args.split_seed,
        sampling_seed=args.sampling_seed,
        budgets=(args.label_budget,),
    )[args.label_budget]
    manifest_path = args.output_root / "subset_manifest.csv"
    metadata = write_manifest(manifest_path, manifest)
    if metadata != manifest_metadata(manifest):
        raise RuntimeError("Saved benchmark manifest differs from the requested manifest")

    results = []
    for microbatch_size in args.microbatch_sizes:
        run_dir = run_directory(
            args.output_root,
            microbatch_size,
            split_seed=args.split_seed,
            sampling_seed=args.sampling_seed,
            label_budget=args.label_budget,
            training_seed=args.training_seed,
        )
        completed_result = load_completed_result(
            run_dir,
            microbatch_size=microbatch_size,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            metadata=metadata,
        )
        if completed_result is not None:
            print(f"\n=== LoRA ListMLE microbatch {microbatch_size}: complete ===", flush=True)
            results.append(completed_result)
            continue

        resume = (run_dir / "config.json").exists()
        if resume and not (run_dir / "last.pt").exists():
            raise RuntimeError(f"No resumable checkpoint in {run_dir}")
        command = build_command(
            microbatch_size,
            device=args.device,
            split_seed=args.split_seed,
            sampling_seed=args.sampling_seed,
            label_budget=args.label_budget,
            training_seed=args.training_seed,
            output_root=args.output_root,
            training_subset=manifest_path,
            resume=resume,
        )
        print(f"\n=== LoRA ListMLE microbatch {microbatch_size} ===", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=project_root(), check=False)
        if completed.returncode != 0:
            results.append(
                {
                    "microbatch_size": microbatch_size,
                    "status": "failed",
                    "return_code": completed.returncode,
                    "test_evaluated": False,
                    "run_directory": str(run_dir.resolve()),
                }
            )
            continue
        result = load_completed_result(
            run_dir,
            microbatch_size=microbatch_size,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            metadata=metadata,
        )
        if result is None:
            raise RuntimeError(f"Benchmark did not produce a complete epoch in {run_dir}")
        results.append(result)

    if args.dry_run:
        return
    completed_results = [result for result in results if result["status"] == "completed"]
    fastest = (
        min(completed_results, key=lambda result: float(result["epoch_seconds"]))
        if completed_results
        else None
    )
    benchmark = {
        "experiment": "phase2_validation_only_lora_listmle_microbatch_benchmark",
        "scientific_result": False,
        "ranking_target_representation": "ordinal_rank_from_float64",
        "effective_lists_per_batch": LISTS_PER_BATCH,
        "split_seed": args.split_seed,
        "sampling_seed": args.sampling_seed,
        "label_budget": args.label_budget,
        "training_seed": args.training_seed,
        "training_subset_sha256": metadata["training_subset_sha256"],
        "test_evaluated": False,
        "fastest_completed_microbatch": (
            fastest["microbatch_size"] if fastest is not None else None
        ),
        "results": results,
        "notes": (
            "Epoch time includes one training epoch plus validation. CUDA reports peak allocated "
            "memory; MPS reports current allocated memory at epoch end, not peak memory."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "benchmark_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
