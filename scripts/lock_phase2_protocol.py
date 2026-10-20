#!/usr/bin/env python3
"""Lock Phase 2 after its validation-only subset smoke check."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from esm2_mutation_effects.data import project_root
from esm2_mutation_effects.label_efficiency import canonical_sha256

SPLIT_SEEDS = (42, 43, 44, 45, 46)
SAMPLING_SEEDS = (1, 2, 3)
LABEL_BUDGETS = (20, 40, 200, 401, 1002, 2004, 4008)
FULL_LABEL_BUDGET = 4008
TRAINING_SEED = 1
CONFIGURATIONS = (
    {"name": "frozen_listmle", "backbone": "frozen", "loss": "listmle"},
    {"name": "lora_listmle", "backbone": "lora", "loss": "listmle"},
)


def build_lock(smoke: dict[str, object], git_commit: str) -> dict[str, object]:
    if smoke.get("experiment") != "phase_2_validation_only_20_label_smoke":
        raise ValueError("Unexpected Phase 2 smoke summary")
    if smoke.get("test_evaluated") is not False:
        raise ValueError("Phase 2 smoke must remain validation-only")
    record = smoke.get("record")
    if not isinstance(record, dict):
        raise ValueError("Phase 2 smoke summary is missing its record")
    expected = {
        "configuration": "lora_listmle",
        "split_seed": 42,
        "sampling_seed": 1,
        "n_labeled_train": 20,
        "test_evaluated": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Phase 2 smoke record does not match the required 20-label protocol")
    if not isinstance(record.get("training_subset_sha256"), str):
        raise ValueError("Phase 2 smoke record is missing its subset fingerprint")

    protocol = {
        "version": 1,
        "model_name": "esm2_t6_8M_UR50D",
        "configurations": list(CONFIGURATIONS),
        "split_seeds": list(SPLIT_SEEDS),
        "sampling_seeds": list(SAMPLING_SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "full_label_budget": FULL_LABEL_BUDGET,
        "sampling_unit": "individual labelled mutation rows within training positions",
        "nested_subsets": True,
        "training_seed": TRAINING_SEED,
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
        "evaluation_positions_per_batch": 16,
        "ranking_target_representation": "ordinal_rank_from_float64",
        "checkpoint_selection": "validation overall Spearman",
        "validation_policy": "full fixed validation positions; labels excluded from budget",
        "test_policy": "one evaluation after validation-selected checkpoint per manifest",
    }
    return {
        "experiment": "phase_2_label_efficiency_protocol_lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "smoke_summary_sha256": canonical_sha256(smoke),
        "smoke_record": record,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "test_evaluated_at_lock": False,
    }


def main() -> None:
    root = project_root()
    smoke_path = root / "results" / "phase2_development" / "smoke_summary.json"
    destination = root / "results" / "phase2_label_efficiency" / "protocol_lock.json"
    if destination.exists():
        raise FileExistsError(f"Phase 2 protocol lock already exists: {destination}")
    if not smoke_path.exists():
        raise FileNotFoundError(f"Phase 2 smoke summary does not exist: {smoke_path}")
    smoke = json.loads(smoke_path.read_text())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = build_lock(smoke, git_commit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
