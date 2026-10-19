#!/usr/bin/env python3
"""Run the validation-only split-42 adaptation development matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root

DEVELOPMENT_SPLIT_SEED = 42
DEVELOPMENT_TRAINING_SEEDS = (1, 2, 3)
DEVELOPMENT_CONFIGURATIONS = (
    ("frozen_mse", "frozen", "mse"),
    ("frozen_listmle", "frozen", "listmle"),
    ("lora_mse", "lora", "mse"),
    ("lora_listmle", "lora", "listmle"),
)
MAX_EPOCHS = 100
PATIENCE = 15
LISTS_PER_BATCH = 16
LIST_MICROBATCH_SIZE = 8


def run_directory(
    output_root: Path,
    configuration_name: str,
    training_seed: int,
) -> Path:
    return (
        output_root
        / configuration_name
        / f"split_{DEVELOPMENT_SPLIT_SEED}"
        / f"seed_{training_seed}"
    )


def build_command(
    configuration: tuple[str, str, str],
    *,
    device: str,
    training_seed: int,
    output_root: Path,
    resume: bool,
) -> list[str]:
    name, backbone, loss_name = configuration
    command = [
        sys.executable,
        str(project_root() / "scripts" / "train_masked_adaptation.py"),
        "--backbone",
        backbone,
        "--loss",
        loss_name,
        "--split-seed",
        str(DEVELOPMENT_SPLIT_SEED),
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
        str(LIST_MICROBATCH_SIZE),
        "--output-dir",
        str(run_directory(output_root, name, training_seed)),
    ]
    if resume:
        command.append("--resume")
    return command


def _expected_configurations() -> dict[str, tuple[str, str]]:
    return {name: (backbone, loss_name) for name, backbone, loss_name in DEVELOPMENT_CONFIGURATIONS}


def _load_run_record(
    output_root: Path,
    configuration_name: str,
    training_seed: int,
) -> dict[str, object] | None:
    directory = run_directory(output_root, configuration_name, training_seed)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text())
    config = summary.get("config", {})
    expected_backbone, expected_loss = _expected_configurations()[configuration_name]
    expected_values = {
        "backbone_mode": expected_backbone,
        "loss_name": expected_loss,
        "split_seed": DEVELOPMENT_SPLIT_SEED,
        "training_seed": training_seed,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lists_per_batch": LISTS_PER_BATCH,
        "list_microbatch_size": LIST_MICROBATCH_SIZE,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in expected_values.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Development run has a protocol mismatch at {directory}: {mismatches}")
    if summary.get("test_evaluated") is not False:
        raise RuntimeError(f"Development run unexpectedly evaluated test positions: {directory}")
    if summary.get("status") not in {"stopped_early", "completed_max_epochs"}:
        raise RuntimeError(f"Development run has an incomplete status at {directory}")
    forbidden_test_artifacts = [
        path
        for path in (directory / "test_predictions.csv", directory / "test_metrics.json")
        if path.exists()
    ]
    if forbidden_test_artifacts:
        raise RuntimeError(
            "Development directory contains forbidden test artifacts: "
            + ", ".join(str(path) for path in forbidden_test_artifacts)
        )

    validation_metrics = json.loads((directory / "validation_metrics.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    trained_epochs = [record for record in history if record["epoch"] > 0]
    return {
        "configuration": configuration_name,
        "backbone": expected_backbone,
        "loss": expected_loss,
        "split_seed": DEVELOPMENT_SPLIT_SEED,
        "training_seed": training_seed,
        "status": summary["status"],
        "best_epoch": summary["best_epoch"],
        "best_validation_spearman": summary["best_validation_spearman"],
        "best_validation_loss": summary["best_validation_loss"],
        "epochs_completed": summary["epochs_completed"],
        "total_epoch_seconds": sum(record["epoch_seconds"] for record in trained_epochs),
        "trainable_parameters": summary["trainable_parameters"],
        "validation_metrics": validation_metrics,
        "test_evaluated": False,
        "run_directory": str(directory.resolve()),
    }


def collect_development_summary(output_root: Path) -> dict[str, object]:
    runs = []
    for configuration_name, _, _ in DEVELOPMENT_CONFIGURATIONS:
        for training_seed in DEVELOPMENT_TRAINING_SEEDS:
            record = _load_run_record(output_root, configuration_name, training_seed)
            if record is not None:
                runs.append(record)

    aggregates = []
    for configuration_name, _, _ in DEVELOPMENT_CONFIGURATIONS:
        configuration_runs = [
            record for record in runs if record["configuration"] == configuration_name
        ]
        correlations = [
            float(record["best_validation_spearman"]) for record in configuration_runs
        ]
        aggregates.append(
            {
                "configuration": configuration_name,
                "n_completed_seeds": len(configuration_runs),
                "completed_training_seeds": sorted(
                    int(record["training_seed"]) for record in configuration_runs
                ),
                "mean_best_validation_spearman": (
                    statistics.mean(correlations) if correlations else None
                ),
                "sample_std_best_validation_spearman": (
                    statistics.stdev(correlations) if len(correlations) > 1 else None
                ),
                "minimum_best_validation_spearman": min(correlations) if correlations else None,
                "maximum_best_validation_spearman": max(correlations) if correlations else None,
            }
        )

    expected_runs = len(DEVELOPMENT_CONFIGURATIONS) * len(DEVELOPMENT_TRAINING_SEEDS)
    return {
        "experiment": "validation_only_split_42_three_seed_development",
        "confirmatory_result": False,
        "split_seed": DEVELOPMENT_SPLIT_SEED,
        "training_seeds": list(DEVELOPMENT_TRAINING_SEEDS),
        "test_evaluated": False,
        "protocol": {
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lists_per_batch": LISTS_PER_BATCH,
            "list_microbatch_size": LIST_MICROBATCH_SIZE,
            "checkpoint_selection": "validation overall Spearman",
        },
        "expected_runs": expected_runs,
        "completed_runs": len(runs),
        "matrix_complete": len(runs) == expected_runs,
        "runs": runs,
        "summary": aggregates,
    }


def _write_development_summary(output_root: Path) -> dict[str, object]:
    summary = collect_development_summary(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "development_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed split-42, three-seed validation development matrix. "
            "This runner has no test-evaluation option."
        )
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--configuration",
        choices=("all",) + tuple(item[0] for item in DEVELOPMENT_CONFIGURATIONS),
        default="all",
        help="Run all configurations or one configuration across the selected seeds",
    )
    parser.add_argument(
        "--training-seed",
        choices=("all",) + tuple(str(seed) for seed in DEVELOPMENT_TRAINING_SEEDS),
        default="all",
        help="Run all three predeclared seeds or one seed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "lora_development",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed commands without starting model training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_configurations = [
        item
        for item in DEVELOPMENT_CONFIGURATIONS
        if args.configuration == "all" or item[0] == args.configuration
    ]
    selected_seeds = [
        seed
        for seed in DEVELOPMENT_TRAINING_SEEDS
        if args.training_seed == "all" or seed == int(args.training_seed)
    ]

    for training_seed in selected_seeds:
        for configuration in selected_configurations:
            name = configuration[0]
            directory = run_directory(args.output_root, name, training_seed)
            if (directory / "summary.json").exists():
                record = _load_run_record(args.output_root, name, training_seed)
                if record is None:
                    raise RuntimeError(f"Could not load completed run summary: {directory}")
                progress = _write_development_summary(args.output_root)
                print(
                    f"\n=== seed {training_seed}: {name} already complete "
                    f"({progress['completed_runs']}/{progress['expected_runs']}) ===",
                    flush=True,
                )
                continue
            config_exists = (directory / "config.json").exists()
            if config_exists and not (directory / "last.pt").exists():
                raise RuntimeError(
                    f"Run directory has a config but no resumable checkpoint: {directory}"
                )
            command = build_command(
                configuration,
                device=args.device,
                training_seed=training_seed,
                output_root=args.output_root,
                resume=config_exists,
            )
            print(f"\n=== seed {training_seed}: {name} ===", flush=True)
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue

            subprocess.run(command, cwd=project_root(), check=True)
            record = _load_run_record(args.output_root, name, training_seed)
            if record is None:
                raise RuntimeError(f"Completed process did not write a summary: {directory}")
            progress = _write_development_summary(args.output_root)
            print(
                json.dumps(
                    {
                        "completed_runs": progress["completed_runs"],
                        "expected_runs": progress["expected_runs"],
                        "latest_configuration": name,
                        "latest_training_seed": training_seed,
                        "latest_best_validation_spearman": record[
                            "best_validation_spearman"
                        ],
                        "test_evaluated": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.dry_run:
        return
    final_summary = _write_development_summary(args.output_root)
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
