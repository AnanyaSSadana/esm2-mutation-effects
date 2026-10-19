#!/usr/bin/env python3
"""Run the protocol-locked five-split confirmatory adaptation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root

CONFIGURATIONS = (
    ("frozen_mse", "frozen", "mse"),
    ("frozen_listmle", "frozen", "listmle"),
    ("lora_mse", "lora", "mse"),
    ("lora_listmle", "lora", "listmle"),
)
SPLIT_SEEDS = (42, 43, 44, 45, 46)
TRAINING_SEEDS = (1, 2, 3)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_lock(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    protocol = value.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError("Protocol lock is missing its protocol object")
    if canonical_sha256(protocol) != value.get("protocol_sha256"):
        raise ValueError("Protocol lock checksum does not match its contents")
    expected = {
        "split_seeds": list(SPLIT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "max_epochs": 100,
        "patience": 5,
        "lists_per_batch": 16,
        "list_microbatch_size": 4,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": protocol.get(key)}
        for key, expected_value in expected.items()
        if protocol.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Protocol lock does not match the confirmatory runner: {mismatches}")
    if value.get("test_evaluated_at_lock") is not False:
        raise ValueError("Protocol lock was not created before test evaluation")
    return value


def run_directory(
    confirmatory_root: Path,
    development_root: Path,
    configuration_name: str,
    split_seed: int,
    training_seed: int,
) -> Path:
    root = development_root if split_seed == 42 else confirmatory_root
    return root / configuration_name / f"split_{split_seed}" / f"seed_{training_seed}"


def build_command(
    configuration: tuple[str, str, str],
    *,
    device: str,
    split_seed: int,
    training_seed: int,
    confirmatory_root: Path,
    development_root: Path,
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
        str(split_seed),
        "--training-seed",
        str(training_seed),
        "--device",
        device,
        "--max-epochs",
        "100",
        "--patience",
        "5",
        "--lists-per-batch",
        "16",
        "--list-microbatch-size",
        "4",
        "--evaluate-test",
        "--output-dir",
        str(
            run_directory(
                confirmatory_root,
                development_root,
                name,
                split_seed,
                training_seed,
            )
        ),
    ]
    if resume:
        command.append("--resume")
    return command


def _load_test_record(
    directory: Path,
    configuration: tuple[str, str, str],
    split_seed: int,
    training_seed: int,
) -> dict[str, object] | None:
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("test_evaluated") is not True:
        return None
    for filename in ("test_predictions.csv", "test_metrics.json"):
        if not (directory / filename).exists():
            raise RuntimeError(f"Confirmatory run is missing {filename}: {directory}")
    name, backbone, loss_name = configuration
    config = summary["config"]
    expected = {
        "backbone_mode": backbone,
        "loss_name": loss_name,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "max_epochs": 100,
        "patience": 5,
        "lists_per_batch": 16,
        "list_microbatch_size": 4,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Confirmatory run configuration mismatch: {directory}")
    metrics = json.loads((directory / "test_metrics.json").read_text())
    return {
        "configuration": name,
        "backbone": backbone,
        "loss": loss_name,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "selected_epoch": summary["best_epoch"],
        "validation_spearman": summary["best_validation_spearman"],
        "test_metrics": metrics,
        "run_directory": str(directory.resolve()),
    }


def collect_confirmatory_summary(
    confirmatory_root: Path,
    development_root: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    runs = []
    for configuration in CONFIGURATIONS:
        for split_seed in SPLIT_SEEDS:
            for training_seed in TRAINING_SEEDS:
                directory = run_directory(
                    confirmatory_root,
                    development_root,
                    configuration[0],
                    split_seed,
                    training_seed,
                )
                record = _load_test_record(
                    directory,
                    configuration,
                    split_seed,
                    training_seed,
                )
                if record is not None:
                    runs.append(record)

    split_level = []
    for name, _, _ in CONFIGURATIONS:
        for split_seed in SPLIT_SEEDS:
            selected = [
                record
                for record in runs
                if record["configuration"] == name and record["split_seed"] == split_seed
            ]
            if not selected:
                continue
            correlations = [
                float(record["test_metrics"]["overall_spearman"]) for record in selected
            ]
            split_level.append(
                {
                    "configuration": name,
                    "split_seed": split_seed,
                    "n_optimization_seeds": len(selected),
                    "mean_test_spearman": statistics.mean(correlations),
                    "sample_std_test_spearman": (
                        statistics.stdev(correlations) if len(correlations) > 1 else None
                    ),
                }
            )

    aggregate = []
    for name, _, _ in CONFIGURATIONS:
        values = [
            float(record["mean_test_spearman"])
            for record in split_level
            if record["configuration"] == name and record["n_optimization_seeds"] == 3
        ]
        aggregate.append(
            {
                "configuration": name,
                "n_complete_splits": len(values),
                "mean_test_spearman": statistics.mean(values) if values else None,
                "sample_std_test_spearman": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
                "minimum_test_spearman": min(values) if values else None,
                "maximum_test_spearman": max(values) if values else None,
            }
        )

    expected_runs = len(CONFIGURATIONS) * len(SPLIT_SEEDS) * len(TRAINING_SEEDS)
    return {
        "experiment": "protocol_locked_five_split_confirmatory_matrix",
        "protocol_sha256": protocol_sha256,
        "expected_runs": expected_runs,
        "completed_runs": len(runs),
        "matrix_complete": len(runs) == expected_runs,
        "test_evaluated": bool(runs),
        "runs": runs,
        "split_level_summary": split_level,
        "aggregate_summary": aggregate,
    }


def _write_summary(
    confirmatory_root: Path,
    development_root: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    value = collect_confirmatory_summary(
        confirmatory_root,
        development_root,
        protocol_sha256,
    )
    confirmatory_root.mkdir(parents=True, exist_ok=True)
    destination = confirmatory_root / "confirmatory_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return value


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run the locked Phase 1 confirmatory matrix")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--split-seed",
        choices=("all",) + tuple(str(seed) for seed in SPLIT_SEEDS),
        default="all",
    )
    parser.add_argument(
        "--training-seed",
        choices=("all",) + tuple(str(seed) for seed in TRAINING_SEEDS),
        default="all",
    )
    parser.add_argument(
        "--configuration",
        choices=("all",) + tuple(item[0] for item in CONFIGURATIONS),
        default="all",
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=root / "results" / "lora_development" / "protocol_lock.json",
    )
    parser.add_argument(
        "--development-root",
        type=Path,
        default=root / "results" / "lora_development",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results" / "lora_confirmatory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = load_and_validate_lock(args.protocol_lock)
    selected_configurations = [
        item
        for item in CONFIGURATIONS
        if args.configuration == "all" or item[0] == args.configuration
    ]
    selected_splits = [
        seed for seed in SPLIT_SEEDS if args.split_seed == "all" or seed == int(args.split_seed)
    ]
    selected_training_seeds = [
        seed
        for seed in TRAINING_SEEDS
        if args.training_seed == "all" or seed == int(args.training_seed)
    ]

    for split_seed in selected_splits:
        for training_seed in selected_training_seeds:
            for configuration in selected_configurations:
                name = configuration[0]
                directory = run_directory(
                    args.output_root,
                    args.development_root,
                    name,
                    split_seed,
                    training_seed,
                )
                record = _load_test_record(
                    directory,
                    configuration,
                    split_seed,
                    training_seed,
                )
                if record is not None:
                    print(f"\n=== split {split_seed}, seed {training_seed}: {name} complete ===")
                    continue
                config_exists = (directory / "config.json").exists()
                if config_exists and not (directory / "last.pt").exists():
                    raise RuntimeError(f"No resumable checkpoint in {directory}")
                command = build_command(
                    configuration,
                    device=args.device,
                    split_seed=split_seed,
                    training_seed=training_seed,
                    confirmatory_root=args.output_root,
                    development_root=args.development_root,
                    resume=config_exists,
                )
                print(f"\n=== split {split_seed}, seed {training_seed}: {name} ===", flush=True)
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                subprocess.run(command, cwd=project_root(), check=True)
                progress = _write_summary(
                    args.output_root,
                    args.development_root,
                    str(lock["protocol_sha256"]),
                )
                print(
                    json.dumps(
                        {
                            "completed_runs": progress["completed_runs"],
                            "expected_runs": progress["expected_runs"],
                            "test_evaluated": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if args.dry_run:
        return
    final = _write_summary(
        args.output_root,
        args.development_root,
        str(lock["protocol_sha256"]),
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
