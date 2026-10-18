#!/usr/bin/env python3
"""Run the four validation-only five-epoch adaptation pilot configurations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root

PILOT_CONFIGURATIONS = (
    ("frozen_mse", "frozen", "mse"),
    ("frozen_listmle", "frozen", "listmle"),
    ("lora_mse", "lora", "mse"),
    ("lora_listmle", "lora", "listmle"),
)


def build_command(
    configuration: tuple[str, str, str],
    *,
    device: str,
    split_seed: int,
    training_seed: int,
    output_root: Path,
    resume: bool,
) -> list[str]:
    name, backbone, loss_name = configuration
    output_dir = output_root / name / f"split_{split_seed}" / f"seed_{training_seed}"
    command = [
        sys.executable,
        str(project_root() / "scripts" / "train_masked_adaptation.py"),
        "--backbone",
        backbone,
        "--loss",
        loss_name,
        "--split-seed",
        str(split_seed),
        "--training-seed",
        str(training_seed),
        "--device",
        device,
        "--max-epochs",
        "5",
        "--patience",
        "5",
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the four five-epoch technical pilot configurations. This runner never evaluates "
            "test positions."
        )
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument(
        "--configuration",
        choices=("all",) + tuple(item[0] for item in PILOT_CONFIGURATIONS),
        default="all",
        help="Run all four configurations or one named configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "lora_pilot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting model training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [
        item
        for item in PILOT_CONFIGURATIONS
        if args.configuration == "all" or item[0] == args.configuration
    ]
    summaries = []
    for configuration in selected:
        name = configuration[0]
        run_dir = (
            args.output_root
            / name
            / f"split_{args.split_seed}"
            / f"seed_{args.training_seed}"
        )
        resume = (run_dir / "config.json").exists()
        command = build_command(
            configuration,
            device=args.device,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            output_root=args.output_root,
            resume=resume,
        )
        print(f"\n=== {name} ===", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=project_root(), check=True)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("test_evaluated") is not False:
            raise RuntimeError(f"Pilot run {name} unexpectedly evaluated test positions")
        history = json.loads((run_dir / "history.json").read_text())
        validation_metrics = json.loads((run_dir / "validation_metrics.json").read_text())
        trained_epochs = [record for record in history if record["epoch"] > 0]
        lora_gradient_norms = [
            record["maximum_lora_gradient_norm"]
            for record in trained_epochs
            if record["maximum_lora_gradient_norm"] is not None
        ]
        allocated_memory = [
            record["allocated_memory_mib"]
            for record in history
            if record["allocated_memory_mib"] is not None
        ]
        summaries.append(
            {
                "configuration": name,
                "status": summary["status"],
                "best_epoch": summary["best_epoch"],
                "best_validation_spearman": summary["best_validation_spearman"],
                "best_validation_loss": summary["best_validation_loss"],
                "epochs_completed": summary["epochs_completed"],
                "trainable_parameters": summary["trainable_parameters"],
                "total_epoch_seconds": sum(record["epoch_seconds"] for record in trained_epochs),
                "maximum_observed_allocated_memory_mib": (
                    max(allocated_memory) if allocated_memory else None
                ),
                "maximum_head_gradient_norm": max(
                    record["maximum_head_gradient_norm"] for record in trained_epochs
                ),
                "maximum_lora_gradient_norm": (
                    max(lora_gradient_norms) if lora_gradient_norms else None
                ),
                "validation_metrics": validation_metrics,
                "test_evaluated": False,
                "run_directory": str(run_dir.resolve()),
            }
        )

    if args.dry_run:
        return
    pilot_summary = {
        "experiment": "validation_only_five_epoch_technical_pilot",
        "scientific_result": False,
        "split_seed": args.split_seed,
        "training_seed": args.training_seed,
        "test_evaluated": False,
        "runs": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "pilot_summary.json"
    summary_path.write_text(json.dumps(pilot_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(pilot_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
