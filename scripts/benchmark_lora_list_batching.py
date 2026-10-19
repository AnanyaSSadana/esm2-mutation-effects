#!/usr/bin/env python3
"""Benchmark validation-only LoRA ListMLE microbatch sizes without test evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root


def build_command(
    microbatch_size: int,
    *,
    device: str,
    split_seed: int,
    training_seed: int,
    output_root: Path,
    resume: bool,
) -> list[str]:
    output_dir = (
        output_root
        / f"microbatch_{microbatch_size}"
        / f"split_{split_seed}"
        / f"seed_{training_seed}"
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
        "1",
        "--patience",
        "1",
        "--lists-per-batch",
        "16",
        "--list-microbatch-size",
        str(microbatch_size),
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one validation-only LoRA ListMLE epoch at larger microbatch sizes. "
            "This runner never evaluates test positions."
        )
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--microbatch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "lora_batch_benchmark",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.microbatch_sizes)) != len(args.microbatch_sizes):
        raise ValueError("Microbatch sizes must be unique")
    if any(size < 1 or size > 16 for size in args.microbatch_sizes):
        raise ValueError("Microbatch sizes must be between 1 and the effective batch size of 16")

    results = []
    for microbatch_size in args.microbatch_sizes:
        run_dir = (
            args.output_root
            / f"microbatch_{microbatch_size}"
            / f"split_{args.split_seed}"
            / f"seed_{args.training_seed}"
        )
        command = build_command(
            microbatch_size,
            device=args.device,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            output_root=args.output_root,
            resume=(run_dir / "config.json").exists(),
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

        summary = json.loads((run_dir / "summary.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        if summary.get("test_evaluated") is not False:
            raise RuntimeError("Batching benchmark unexpectedly evaluated test positions")
        epoch = next(record for record in history if record["epoch"] == 1)
        results.append(
            {
                "microbatch_size": microbatch_size,
                "status": "completed",
                "epoch_seconds": epoch["epoch_seconds"],
                "allocated_memory_mib_at_epoch_end": epoch["allocated_memory_mib"],
                "train_loss": epoch["train_loss"],
                "validation_spearman": epoch["validation_metrics"]["overall_spearman"],
                "maximum_gradient_norm": epoch["maximum_gradient_norm"],
                "maximum_lora_gradient_norm": epoch["maximum_lora_gradient_norm"],
                "test_evaluated": False,
                "run_directory": str(run_dir.resolve()),
            }
        )

    if args.dry_run:
        return
    benchmark = {
        "experiment": "validation_only_lora_listmle_microbatch_benchmark",
        "scientific_result": False,
        "effective_lists_per_batch": 16,
        "split_seed": args.split_seed,
        "training_seed": args.training_seed,
        "test_evaluated": False,
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "benchmark_summary.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
