#!/usr/bin/env python3
"""Run the protocol-locked Phase 2 label-efficiency confirmation matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from esm2_mutation_effects.data import load_assay, mutation_table, project_root
from esm2_mutation_effects.diagnostics import position_ranking_metrics
from esm2_mutation_effects.label_efficiency import (
    canonical_sha256,
    manifest_metadata,
    nested_mutation_manifests,
    write_manifest,
)
from esm2_mutation_effects.splits import make_splits, validate_splits

SPLIT_SEEDS = (42, 43, 44, 45, 46)
SAMPLING_SEEDS = (1, 2, 3)
LABEL_BUDGETS = (20, 40, 200, 401, 1002, 2004, 4008)
FULL_LABEL_BUDGET = 4008
TRAINING_SEED = 1
CONFIGURATIONS = (
    ("frozen_listmle", "frozen", "listmle"),
    ("lora_listmle", "lora", "listmle"),
)
MAX_EPOCHS = 100
PATIENCE = 5
LISTS_PER_BATCH = 16
LIST_MICROBATCH_SIZE = 4


def run_directory(
    output_root: Path,
    configuration_name: str,
    split_seed: int,
    sampling_seed: int,
    label_budget: int,
) -> Path:
    return (
        output_root
        / configuration_name
        / f"split_{split_seed}"
        / f"sample_{sampling_seed}"
        / f"labels_{label_budget}"
    )


def expected_run_count() -> int:
    per_configuration = len(SPLIT_SEEDS) * (
        1 + len(SAMPLING_SEEDS) * (len(LABEL_BUDGETS) - 1)
    )
    return len(CONFIGURATIONS) * per_configuration


def canonical_protocol() -> dict[str, object]:
    return {
        "version": 1,
        "model_name": "esm2_t6_8M_UR50D",
        "configurations": [
            {"name": name, "backbone": backbone, "loss": loss_name}
            for name, backbone, loss_name in CONFIGURATIONS
        ],
        "split_seeds": list(SPLIT_SEEDS),
        "sampling_seeds": list(SAMPLING_SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "full_label_budget": FULL_LABEL_BUDGET,
        "sampling_unit": "individual labelled mutation rows within training positions",
        "nested_subsets": True,
        "training_seed": TRAINING_SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "head_learning_rate": 1e-3,
        "lora_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "maximum_gradient_norm": 3.0,
        "lora_rank": 16,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "list_size": 5,
        "lists_per_batch": LISTS_PER_BATCH,
        "list_microbatch_size": LIST_MICROBATCH_SIZE,
        "evaluation_positions_per_batch": 16,
        "ranking_target_representation": "ordinal_rank_from_float64",
        "checkpoint_selection": "validation overall Spearman",
        "validation_policy": "full fixed validation positions; labels excluded from budget",
        "test_policy": "one evaluation after validation-selected checkpoint per manifest",
    }


def load_and_validate_lock(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 protocol lock does not exist: {path}")
    lock = json.loads(path.read_text())
    protocol = lock.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError("Phase 2 protocol lock is missing its protocol")
    if canonical_sha256(protocol) != lock.get("protocol_sha256"):
        raise ValueError("Phase 2 protocol lock checksum does not match its protocol")
    expected = canonical_protocol()
    if protocol != expected:
        raise ValueError("Phase 2 protocol lock does not match the confirmation runner")
    if lock.get("test_evaluated_at_lock") is not False:
        raise ValueError("Phase 2 protocol lock was created after test evaluation")
    return lock


def build_command(
    configuration: tuple[str, str, str],
    *,
    device: str,
    split_seed: int,
    output_root: Path,
    sampling_seed: int,
    label_budget: int,
    resume: bool,
) -> list[str]:
    name, backbone, loss_name = configuration
    directory = run_directory(output_root, name, split_seed, sampling_seed, label_budget)
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
        "--evaluate-test",
        "--output-dir",
        str(directory),
    ]
    if resume:
        command.append("--resume")
    return command


def load_run_record(
    output_root: Path,
    configuration: tuple[str, str, str],
    *,
    split_seed: int,
    sampling_seed: int,
    label_budget: int,
    metadata: dict[str, object],
) -> dict[str, object] | None:
    name, backbone, loss_name = configuration
    directory = run_directory(output_root, name, split_seed, sampling_seed, label_budget)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("test_evaluated") is not True:
        return None
    for required in (directory / "test_predictions.csv", directory / "test_metrics.json"):
        if not required.exists():
            raise RuntimeError(f"Phase 2 run is missing {required.name}: {directory}")
    expected = {
        "backbone_mode": backbone,
        "loss_name": loss_name,
        "split_seed": split_seed,
        "training_seed": TRAINING_SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lists_per_batch": LISTS_PER_BATCH,
        "list_microbatch_size": LIST_MICROBATCH_SIZE,
        "ranking_target_representation": "ordinal_rank_from_float64",
        **metadata,
    }
    config = summary.get("config", {})
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Phase 2 run has a protocol mismatch at {directory}: {mismatches}")
    if summary.get("n_labeled_train") != label_budget:
        raise RuntimeError(f"Phase 2 run has an incorrect label count at {directory}")
    metrics = json.loads((directory / "test_metrics.json").read_text())
    return {
        "configuration": name,
        "backbone": backbone,
        "loss": loss_name,
        "split_seed": split_seed,
        "sampling_seed": None if label_budget == FULL_LABEL_BUDGET else sampling_seed,
        "label_budget": label_budget,
        "n_labeled_train": label_budget,
        "training_subset_sha256": metadata["training_subset_sha256"],
        "best_epoch": summary["best_epoch"],
        "validation_spearman": summary["best_validation_spearman"],
        "test_metrics": metrics,
        "run_directory": str(directory.resolve()),
    }


def zero_shot_records() -> list[dict[str, object]]:
    assay = load_assay()
    scores_path = project_root() / "results" / "esm2_masked_marginal.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Zero-shot predictions do not exist: {scores_path}")
    scores = pd.read_csv(scores_path)
    aligned = assay[["mutant", "DMS_score"]].merge(
        scores[["mutant", "predicted_score"]],
        on="mutant",
        validate="one_to_one",
    )
    positions = mutation_table(assay)["position"].to_numpy()
    records = []
    for split_seed in SPLIT_SEEDS:
        split = make_splits(assay, seed=split_seed)
        validate_splits(split, set(assay["mutant"]))
        selected = split["position"].to_numpy() == "test"
        metrics = position_ranking_metrics(
            aligned.loc[selected, "DMS_score"].to_numpy(),
            aligned.loc[selected, "predicted_score"].to_numpy(),
            positions[selected],
        )
        metrics["subset"] = "test"
        records.append(
            {
                "configuration": "zero_shot_esm2",
                "split_seed": split_seed,
                "n_labeled_train": 0,
                "test_metrics": metrics,
            }
        )
    return records


def aggregate_records(runs: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in runs:
        grouped[(str(record["configuration"]), int(record["label_budget"]))].append(record)

    split_level = []
    for (configuration, budget), group in sorted(grouped.items()):
        expected_samples = 1 if budget == FULL_LABEL_BUDGET else len(SAMPLING_SEEDS)
        for split_seed in SPLIT_SEEDS:
            values = [
                float(item["test_metrics"]["overall_spearman"])
                for item in group
                if item["split_seed"] == split_seed
            ]
            if not values:
                continue
            split_level.append(
                {
                    "configuration": configuration,
                    "label_budget": budget,
                    "split_seed": split_seed,
                    "n_subset_samples": len(values),
                    "expected_subset_samples": expected_samples,
                    "mean_test_spearman": statistics.mean(values),
                    "sample_std_test_spearman": (
                        statistics.stdev(values) if len(values) > 1 else None
                    ),
                }
            )

    aggregate = []
    for (configuration, budget), group in sorted(grouped.items()):
        selected = [
            item["mean_test_spearman"]
            for item in split_level
            if item["configuration"] == configuration
            and item["label_budget"] == budget
            and item["n_subset_samples"] == item["expected_subset_samples"]
        ]
        within_split = [
            item["sample_std_test_spearman"]
            for item in split_level
            if item["configuration"] == configuration
            and item["label_budget"] == budget
            and item["sample_std_test_spearman"] is not None
        ]
        aggregate.append(
            {
                "configuration": configuration,
                "label_budget": budget,
                "n_complete_splits": len(selected),
                "mean_test_spearman": statistics.mean(selected) if selected else None,
                "sample_std_test_spearman": (
                    statistics.stdev(selected) if len(selected) > 1 else None
                ),
                "minimum_test_spearman": min(selected) if selected else None,
                "maximum_test_spearman": max(selected) if selected else None,
                "mean_within_split_sampling_std": (
                    statistics.mean(within_split) if within_split else None
                ),
            }
        )
    return split_level, aggregate


def collect_summary(output_root: Path, protocol_sha256: str) -> dict[str, object]:
    assay = load_assay()
    runs = []
    for split_seed in SPLIT_SEEDS:
        for sampling_seed in SAMPLING_SEEDS:
            manifests = nested_mutation_manifests(
                assay,
                split_seed=split_seed,
                sampling_seed=sampling_seed,
                budgets=LABEL_BUDGETS,
            )
            for budget, manifest in manifests.items():
                if budget == FULL_LABEL_BUDGET and sampling_seed != SAMPLING_SEEDS[0]:
                    continue
                metadata = manifest_metadata(manifest)
                for configuration in CONFIGURATIONS:
                    record = load_run_record(
                        output_root,
                        configuration,
                        split_seed=split_seed,
                        sampling_seed=sampling_seed,
                        label_budget=budget,
                        metadata=metadata,
                    )
                    if record is not None:
                        runs.append(record)
    split_level, aggregate = aggregate_records(runs)
    return {
        "experiment": "phase_2_protocol_locked_label_efficiency",
        "protocol_sha256": protocol_sha256,
        "expected_supervised_runs": expected_run_count(),
        "completed_supervised_runs": len(runs),
        "matrix_complete": len(runs) == expected_run_count(),
        "test_evaluated": bool(runs),
        "runs": runs,
        "split_level_summary": split_level,
        "aggregate_summary": aggregate,
        "zero_shot_reference": zero_shot_records(),
    }


def write_summary(output_root: Path, protocol_sha256: str) -> dict[str, object]:
    payload = collect_summary(output_root, protocol_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "phase2_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return payload


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--configuration",
        choices=("all",) + tuple(item[0] for item in CONFIGURATIONS),
        default="all",
    )
    parser.add_argument(
        "--split-seed",
        choices=("all",) + tuple(str(value) for value in SPLIT_SEEDS),
        default="all",
    )
    parser.add_argument(
        "--sampling-seed",
        choices=("all",) + tuple(str(value) for value in SAMPLING_SEEDS),
        default="all",
    )
    parser.add_argument(
        "--label-budget",
        choices=("all",) + tuple(str(value) for value in LABEL_BUDGETS),
        default="all",
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=root / "results" / "phase2_label_efficiency" / "protocol_lock.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results" / "phase2_label_efficiency",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = load_and_validate_lock(args.protocol_lock)
    selected_configurations = [
        item for item in CONFIGURATIONS if args.configuration == "all" or item[0] == args.configuration
    ]
    selected_splits = [
        seed for seed in SPLIT_SEEDS if args.split_seed == "all" or seed == int(args.split_seed)
    ]
    selected_samples = [
        seed
        for seed in SAMPLING_SEEDS
        if args.sampling_seed == "all" or seed == int(args.sampling_seed)
    ]
    selected_budgets = [
        budget
        for budget in LABEL_BUDGETS
        if args.label_budget == "all" or budget == int(args.label_budget)
    ]

    assay = load_assay()
    for split_seed in selected_splits:
        manifests_by_sample = {
            sampling_seed: nested_mutation_manifests(
                assay,
                split_seed=split_seed,
                sampling_seed=sampling_seed,
                budgets=tuple(selected_budgets),
            )
            for sampling_seed in selected_samples
        }
        for budget in selected_budgets:
            applicable_samples = (
                [SAMPLING_SEEDS[0]] if budget == FULL_LABEL_BUDGET else selected_samples
            )
            if budget == FULL_LABEL_BUDGET and SAMPLING_SEEDS[0] not in selected_samples:
                continue
            for sampling_seed in applicable_samples:
                manifest = manifests_by_sample[sampling_seed][budget]
                for configuration in selected_configurations:
                    directory = run_directory(
                        args.output_root,
                        configuration[0],
                        split_seed,
                        sampling_seed,
                        budget,
                    )
                    metadata = write_manifest(directory / "subset_manifest.csv", manifest)
                    record = load_run_record(
                        args.output_root,
                        configuration,
                        split_seed=split_seed,
                        sampling_seed=sampling_seed,
                        label_budget=budget,
                        metadata=metadata,
                    )
                    if record is not None:
                        print(
                            f"\n=== split {split_seed}, sample {sampling_seed}, labels {budget}: "
                            f"{configuration[0]} complete ==="
                        )
                        continue
                    resume = (directory / "config.json").exists()
                    if resume and not (directory / "last.pt").exists():
                        raise RuntimeError(f"No resumable checkpoint in {directory}")
                    command = build_command(
                        configuration,
                        device=args.device,
                        split_seed=split_seed,
                        output_root=args.output_root,
                        sampling_seed=sampling_seed,
                        label_budget=budget,
                        resume=resume,
                    )
                    print(
                        f"\n=== split {split_seed}, sample {sampling_seed}, labels {budget}: "
                        f"{configuration[0]} ===",
                        flush=True,
                    )
                    print(" ".join(command), flush=True)
                    if args.dry_run:
                        continue
                    subprocess.run(command, cwd=project_root(), check=True)
                    progress = write_summary(args.output_root, str(lock["protocol_sha256"]))
                    print(
                        json.dumps(
                            {
                                "completed_supervised_runs": progress["completed_supervised_runs"],
                                "expected_supervised_runs": progress["expected_supervised_runs"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    if args.dry_run:
        return
    final = write_summary(args.output_root, str(lock["protocol_sha256"]))
    print(
        json.dumps(
            {
                "completed_supervised_runs": final["completed_supervised_runs"],
                "expected_supervised_runs": final["expected_supervised_runs"],
                "matrix_complete": final["matrix_complete"],
                "summary": final["aggregate_summary"],
                "summary_path": str((args.output_root / "phase2_summary.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
