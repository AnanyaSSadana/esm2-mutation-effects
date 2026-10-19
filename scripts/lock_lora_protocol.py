#!/usr/bin/env python3
"""Freeze the confirmatory LoRA protocol after validation-only development."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

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


def build_lock(development_summary: dict[str, object], git_commit: str) -> dict[str, object]:
    if development_summary.get("matrix_complete") is not True:
        raise ValueError("Development matrix must be complete before protocol lock")
    if development_summary.get("test_evaluated") is not False:
        raise ValueError("Development summary must be validation-only")
    if development_summary.get("completed_runs") != 12:
        raise ValueError("Development summary must contain all 12 runs")

    protocol = {
        "version": 1,
        "model_name": "esm2_t6_8M_UR50D",
        "configurations": [
            {"name": name, "backbone": backbone, "loss": loss_name}
            for name, backbone, loss_name in CONFIGURATIONS
        ],
        "split_seeds": list(SPLIT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "max_epochs": 100,
        "patience": 5,
        "head_learning_rate": 1e-3,
        "lora_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "maximum_gradient_norm": 3.0,
        "lora_rank": 16,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "list_size": 5,
        "lists_per_batch": 16,
        "list_microbatch_size": 4,
        "mse_positions_per_batch": 16,
        "evaluation_positions_per_batch": 16,
        "checkpoint_selection": "validation overall Spearman",
        "test_policy": "one evaluation per saved best checkpoint after this lock",
        "primary_metric": "test overall Spearman averaged over optimization seeds per split",
    }
    development_digest = canonical_sha256(development_summary)
    return {
        "experiment": "controlled_lora_phase_1_confirmatory_lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "development_summary_sha256": development_digest,
        "development_validation_summary": development_summary["summary"],
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "test_evaluated_at_lock": False,
    }


def main() -> None:
    root = project_root()
    development_path = root / "results" / "lora_development" / "development_summary.json"
    destination = root / "results" / "lora_development" / "protocol_lock.json"
    if destination.exists():
        raise FileExistsError(f"Protocol lock already exists: {destination}")
    if not development_path.exists():
        raise FileNotFoundError(f"Development summary does not exist: {development_path}")
    development_summary = json.loads(development_path.read_text())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = build_lock(development_summary, git_commit)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
