"""Run the four new pre-registered Phase 3B-lite target partitions once each."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esm2_mutation_effects.adaptation import MaskedMarginalResidualModel
from esm2_mutation_effects.data import (
    default_assay_path,
    default_reference_path,
    load_assay_metadata,
    project_root,
)
from esm2_mutation_effects.diagnostics import position_ranking_metrics
from esm2_mutation_effects.esm_utils import REPRESENTATION_LAYER, choose_device, load_model
from esm2_mutation_effects.lora import inject_lora
from esm2_mutation_effects.phase3_pilot import META_TRAIN_TASKS
from esm2_mutation_effects.phase3_protocol import validate_strict_target_manifest
from esm2_mutation_effects.phase3_training import (
    adapt_fixed_steps,
    prepare_pilot_data,
    train_pooled,
    train_reptile,
)
from esm2_mutation_effects.phase3b_lite import (
    LABEL_BUDGET,
    METHODS,
    MICROBATCH_SIZE,
    NEW_SPLIT_SEEDS,
    OPTIMIZATION_SEED,
    POOLED_UPDATES,
    REPTILE_EPISODES,
    REPTILE_OUTER_STEP_SIZE,
    TARGET_STEPS,
    validate_protocol,
)
from esm2_mutation_effects.training import _predict_rows, seed_everything

EPISODE_COUNTS = {
    task.dms_id: 38 if index < 2 else 37 for index, task in enumerate(META_TRAIN_TASKS)
}


def args_() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "phase3" / "raw")
    parser.add_argument(
        "--zero-shot-dir", type=Path, default=root / "results" / "phase3a_pilot" / "zero_shot"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "phase3b_lite")
    return parser.parse_args()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_model(device: torch.device) -> MaskedMarginalResidualModel:
    backbone, _ = load_model(device)
    inject_lora(backbone, rank=16, alpha=16, dropout=0.05)
    return MaskedMarginalResidualModel(backbone, representation_layer=REPRESENTATION_LAYER).to(
        device
    )


def trainable_state(model: MaskedMarginalResidualModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(
    model: MaskedMarginalResidualModel, state: dict[str, torch.Tensor]
) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("Shared initialization trainable parameters do not match this model")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def predict(
    model: MaskedMarginalResidualModel, data, rows: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return _predict_rows(model, data, rows, device).cpu().numpy()


def evaluate(
    name: str, data, rows: np.ndarray, predictions: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    frame = data.frame.iloc[rows][["mutant", "DMS_score", "position"]].copy()
    frame["predicted_score"] = predictions
    metrics = position_ranking_metrics(
        frame.DMS_score.to_numpy(float),
        frame.predicted_score.to_numpy(float),
        frame.position.to_numpy(),
    )
    return frame, {"method": name, **metrics}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_or_train_shared_initialization(
    method: str,
    tasks: dict,
    device: torch.device,
    path: Path,
    protocol_sha256: str,
) -> dict[str, torch.Tensor]:
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        if payload.get("protocol_sha256") != protocol_sha256 or payload.get("method") != method:
            raise ValueError(f"Shared {method} initialization does not match this protocol")
        return payload["trainable_state"]

    seed_everything(OPTIMIZATION_SEED)
    model = make_model(device)
    if method == "pooled":
        history = train_pooled(
            model,
            tasks,
            {task: count * 2 for task, count in EPISODE_COUNTS.items()},
            device,
            OPTIMIZATION_SEED,
        )
    elif method == "reptile":
        history = train_reptile(
            model,
            tasks,
            EPISODE_COUNTS,
            device,
            OPTIMIZATION_SEED,
            outer_step_size=REPTILE_OUTER_STEP_SIZE,
        )
    else:
        raise ValueError(f"Unsupported shared initialization method: {method}")
    if len(history) != (POOLED_UPDATES if method == "pooled" else REPTILE_EPISODES):
        raise RuntimeError(f"{method} produced an unexpected number of pretraining records")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_sha256": protocol_sha256,
            "method": method,
            "trainable_state": trainable_state(model),
            "pretraining_history": history,
        },
        path,
    )
    return trainable_state(model)


def run_split(
    *,
    split_seed: int,
    manifest: pd.DataFrame,
    target,
    shared_states: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    run_dir: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    """Adapt/evaluate every method once for one previously locked target split."""
    if (run_dir / "summary.json").exists() or (run_dir / "test_predictions.csv").exists():
        raise FileExistsError(f"Target test output already exists for split {split_seed}")
    visible = np.flatnonzero(
        manifest.set_index("mutant").loc[target.frame.mutant, "target_label_visible"].to_numpy()
    )
    test_rows = np.flatnonzero(
        manifest.set_index("mutant").loc[target.frame.mutant, "subset"].to_numpy() == "test"
    )
    if len(visible) != LABEL_BUDGET:
        raise ValueError(f"Split {split_seed} does not expose exactly {LABEL_BUDGET} labels")
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    zero_frame, zero_metrics = evaluate(
        "zero_shot", target, test_rows, target.fixed_zero_shot[test_rows].numpy()
    )
    records.append(zero_metrics)
    predictions.append(zero_frame.assign(method="zero_shot"))
    for method in ("standard", "pooled", "reptile"):
        seed_everything(OPTIMIZATION_SEED)
        model = make_model(device)
        if method in shared_states:
            load_trainable_state(model, shared_states[method])
        losses = adapt_fixed_steps(
            model,
            target,
            visible,
            device,
            steps=TARGET_STEPS,
            seed=OPTIMIZATION_SEED,
        )
        frame, metrics = evaluate(
            method, target, test_rows, predict(model, target, test_rows, device)
        )
        records.append(metrics)
        predictions.append(frame.assign(method=method))
        torch.save(
            {"trainable_state": trainable_state(model), "adaptation_losses": losses},
            run_dir / f"{method}.pt",
        )

    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)
    summary: dict[str, object] = {
        "protocol_sha256": protocol_sha256,
        "split_seed": split_seed,
        "n_visible_target_labels": len(visible),
        "test_evaluated": True,
        "results": records,
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def completed_summary(path: Path, split_seed: int, protocol_sha256: str) -> dict[str, object]:
    summary = json.loads(path.read_text())
    expected = {
        "protocol_sha256": protocol_sha256,
        "split_seed": split_seed,
        "n_visible_target_labels": LABEL_BUDGET,
        "test_evaluated": True,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed split {split_seed} summary does not match the protocol")
    if {row.get("method") for row in summary.get("results", [])} != set(METHODS):
        raise ValueError(f"Completed split {split_seed} does not contain every required method")
    return summary


def main() -> None:
    args = args_()
    if args.microbatch_size != MICROBATCH_SIZE:
        raise ValueError(f"Phase 3B-lite is locked to microbatch size {MICROBATCH_SIZE}")
    root = project_root()
    device = choose_device(args.device)
    lock_path = args.output_dir / "protocol_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("Create the Phase 3B-lite protocol lock before running the matrix")
    lock = json.loads(lock_path.read_text())
    validate_protocol(lock, git_commit=git_commit(root))
    protocol_sha256 = str(lock["protocol_sha256"])
    matrix_path = args.output_dir / "matrix_summary.json"
    if matrix_path.exists():
        raise FileExistsError(
            "Phase 3B-lite matrix already completed; refusing to reopen test outputs"
        )

    required_scores = ["BLAT_ECOLX_Stiffler_2015", *EPISODE_COUNTS]
    missing = [
        task for task in required_scores if not (args.zero_shot_dir / f"{task}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing cached Phase 3 zero-shot scores: {missing}")

    metadata = pd.read_csv(default_reference_path()).set_index("DMS_id", verify_integrity=True)
    _, alphabet = load_model(device)
    target = prepare_pilot_data(
        pd.read_csv(default_assay_path()),
        pd.read_csv(args.zero_shot_dir / "BLAT_ECOLX_Stiffler_2015.csv"),
        str(load_assay_metadata()["target_seq"]),
        alphabet,
    )
    tasks = {
        task.dms_id: prepare_pilot_data(
            pd.read_csv(args.raw_dir / f"{task.dms_id}.csv"),
            pd.read_csv(args.zero_shot_dir / f"{task.dms_id}.csv"),
            str(metadata.loc[task.dms_id, "target_seq"]),
            alphabet,
        )
        for task in META_TRAIN_TASKS
    }
    shared_dir = args.output_dir / "shared_initializations"
    shared_states = {
        method: load_or_train_shared_initialization(
            method,
            tasks,
            device,
            shared_dir / f"{method}.pt",
            protocol_sha256,
        )
        for method in ("pooled", "reptile")
    }

    summaries = []
    for split_seed in NEW_SPLIT_SEEDS:
        run_dir = args.output_dir / "runs" / f"split_{split_seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summaries.append(completed_summary(summary_path, split_seed, protocol_sha256))
            continue
        if (run_dir / "test_predictions.csv").exists():
            raise FileExistsError(
                f"Split {split_seed} has test predictions without a valid summary; refusing rerun"
            )
        manifest = pd.read_csv(args.output_dir / "manifests" / f"split_{split_seed}.csv")
        validate_strict_target_manifest(manifest, label_budget=LABEL_BUDGET)
        summaries.append(
            run_split(
                split_seed=split_seed,
                manifest=manifest,
                target=target,
                shared_states=shared_states,
                device=device,
                run_dir=run_dir,
                protocol_sha256=protocol_sha256,
            )
        )
        print(f"Completed Phase 3B-lite split {split_seed}", flush=True)

    matrix = {
        "phase": "phase3b_lite_repeated_target_split_replication",
        "protocol_lock": str(lock_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "device": str(device),
        "completed_new_splits": list(NEW_SPLIT_SEEDS),
        "matrix_complete": True,
        "test_evaluated": True,
        "runs": summaries,
    }
    atomic_json(matrix_path, matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
