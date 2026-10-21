"""Pre-registered constants and protocol construction for Phase 3B-lite."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .phase3_protocol import canonical_sha256, make_strict_target_manifest, manifest_sha256

# Seed 142 was evaluated exactly once in Phase 3A.  These are the four new,
# pre-registered outer partitions that complete the five-split replication.
NEW_SPLIT_SEEDS = (143, 144, 145, 146)
HISTORICAL_PHASE3A_SPLIT_SEED = 142
SUBSET_SEED = 11
LABEL_BUDGET = 401
OPTIMIZATION_SEED = 1
TARGET_STEPS = 50
MICROBATCH_SIZE = 16
POOLED_UPDATES = 300
REPTILE_EPISODES = 150
REPTILE_INNER_STEPS = 2
REPTILE_OUTER_STEP_SIZE = 0.1
METHODS = ("zero_shot", "standard", "pooled", "reptile")


def make_protocol(
    assay: pd.DataFrame, *, git_commit: str
) -> tuple[dict[str, object], dict[int, pd.DataFrame]]:
    """Build the immutable protocol payload and four strict target manifests."""
    if not git_commit or len(git_commit) < 7:
        raise ValueError("git_commit must be a nonempty Git revision")
    manifests = {
        seed: make_strict_target_manifest(
            assay,
            split_seed=seed,
            subset_seed=SUBSET_SEED,
            label_budget=LABEL_BUDGET,
        )
        for seed in NEW_SPLIT_SEEDS
    }
    manifest_records = [
        {
            "split_seed": seed,
            "subset_seed": SUBSET_SEED,
            "manifest_sha256": manifest_sha256(manifests[seed]),
            "visible_target_labels": int(manifests[seed].target_label_visible.sum()),
            "test_positions": int(
                manifests[seed].loc[manifests[seed].subset == "test", "position"].nunique()
            ),
        }
        for seed in NEW_SPLIT_SEEDS
    ]
    payload: dict[str, object] = {
        "phase": "phase3b_lite_repeated_target_split_replication",
        "git_commit": git_commit,
        "historical_phase3a_split_seed": HISTORICAL_PHASE3A_SPLIT_SEED,
        "new_split_seeds": list(NEW_SPLIT_SEEDS),
        "subset_seed": SUBSET_SEED,
        "target_label_budget": LABEL_BUDGET,
        "optimization_seed": OPTIMIZATION_SEED,
        "microbatch_size": MICROBATCH_SIZE,
        "target_steps": TARGET_STEPS,
        "pooled_updates": POOLED_UPDATES,
        "reptile_episodes": REPTILE_EPISODES,
        "reptile_inner_steps": REPTILE_INNER_STEPS,
        "reptile_outer_step_size": REPTILE_OUTER_STEP_SIZE,
        "methods": list(METHODS),
        "primary_comparison": "pooled_minus_standard_overall_spearman",
        "shared_auxiliary_initializations": True,
        "manifests": manifest_records,
        "test_opened": False,
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return payload, manifests


def validate_protocol(lock: Mapping[str, object], *, git_commit: str) -> None:
    """Reject protocol edits, changed code, or an already-opened test matrix."""
    observed = dict(lock)
    saved_hash = observed.pop("protocol_sha256", None)
    if saved_hash != canonical_sha256(observed):
        raise ValueError("Phase 3B-lite protocol checksum does not match its contents")
    if lock.get("phase") != "phase3b_lite_repeated_target_split_replication":
        raise ValueError("Unexpected Phase 3B-lite protocol phase")
    if lock.get("git_commit") != git_commit:
        raise ValueError("Phase 3B-lite protocol was locked from a different code commit")
    if lock.get("test_opened") is not False:
        raise ValueError("Phase 3B-lite target test matrix has already been opened")
    if tuple(lock.get("new_split_seeds", [])) != NEW_SPLIT_SEEDS:
        raise ValueError("Phase 3B-lite split seeds do not match the pre-registered protocol")
    if lock.get("target_label_budget") != LABEL_BUDGET:
        raise ValueError("Phase 3B-lite label budget does not match the pre-registered protocol")
    if lock.get("microbatch_size") != MICROBATCH_SIZE:
        raise ValueError("Phase 3B-lite microbatch size does not match the pre-registered protocol")
