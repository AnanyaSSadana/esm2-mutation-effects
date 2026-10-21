#!/usr/bin/env python3
"""Backfill completed Phase 2 metrics to Weights & Biases without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esm2_mutation_effects.data import project_root


def relative_run_directory(recorded_path: str) -> Path:
    path = Path(recorded_path)
    parts = path.parts
    try:
        index = parts.index("phase2_label_efficiency")
    except ValueError as error:
        raise ValueError(f"Run path has no phase2_label_efficiency component: {path}") from error
    return Path(*parts[index + 1 :])


def history_path(run: dict[str, object], results_root: Path) -> Path:
    return results_root / relative_run_directory(str(run["run_directory"])) / "history.json"


def run_id(run: dict[str, object]) -> str:
    sample = "full" if run["sampling_seed"] is None else str(run["sampling_seed"])
    return (
        f"p2-{run['configuration']}-s{run['split_seed']}-u{sample}-n{run['label_budget']}"
    )


def epoch_metrics(record: dict[str, object]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"epoch": int(record["epoch"])}
    mappings = {
        "train/loss": record.get("train_loss"),
        "validation/loss": record.get("validation_loss"),
        "validation/spearman": record.get("validation_metrics", {}).get("overall_spearman"),
        "system/epoch_seconds": record.get("epoch_seconds"),
        "system/allocated_memory_mib": record.get("allocated_memory_mib"),
        "gradients/maximum_norm": record.get("maximum_gradient_norm"),
        "gradients/maximum_head_norm": record.get("maximum_head_gradient_norm"),
        "gradients/maximum_lora_norm": record.get("maximum_lora_gradient_norm"),
    }
    for name, value in mappings.items():
        if value is not None:
            metrics[name] = float(value)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root() / "results" / "phase2_label_efficiency" / "phase2_summary.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root() / "results" / "phase2_label_efficiency",
        help="Root containing per-run directories and history.json files",
    )
    parser.add_argument("--project", default="esm2-mutation-effects")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="phase2-label-efficiency")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create or update W&B runs; without this flag the script is read-only",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if summary.get("matrix_complete") is not True or len(summary.get("runs", [])) != 190:
        raise ValueError("Refusing to sync an incomplete Phase 2 matrix")
    histories = {
        run_id(run): history_path(run, args.results_root)
        for run in summary["runs"]
    }
    available_histories = sum(path.exists() for path in histories.values())
    plan = {
        "project": args.project,
        "entity": args.entity,
        "group": args.group,
        "runs": len(summary["runs"]),
        "histories_available": available_histories,
        "histories_missing": len(summary["runs"]) - available_histories,
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        print("Dry run only. Add --execute after reviewing this plan and logging in to W&B.")
        return

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install tracking dependencies with: pip install -e '.[tracking]'") from error

    for record in summary["runs"]:
        identifier = run_id(record)
        configuration = {
            "phase": 2,
            "configuration": record["configuration"],
            "backbone": record["backbone"],
            "loss": record["loss"],
            "split_seed": record["split_seed"],
            "sampling_seed": record["sampling_seed"],
            "label_budget": record["label_budget"],
            "n_labeled_train": record["n_labeled_train"],
            "training_subset_sha256": record["training_subset_sha256"],
            "protocol_sha256": summary["protocol_sha256"],
        }
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            job_type=str(record["configuration"]),
            name=identifier,
            id=identifier,
            resume="allow",
            config=configuration,
            reinit=True,
        )
        if run.summary.get("phase2_import_complete"):
            print(f"Skipping already imported W&B run {identifier}")
            run.finish()
            continue

        saved_history = history_path(record, args.results_root)
        if saved_history.exists():
            for epoch in json.loads(saved_history.read_text()):
                metrics = epoch_metrics(epoch)
                run.log(metrics, step=int(metrics["epoch"]))
        test_metrics = record["test_metrics"]
        run.summary.update(
            {
                "selection/best_epoch": record["best_epoch"],
                "selection/best_validation_spearman": record["validation_spearman"],
                **{
                    f"test/{name}": value
                    for name, value in test_metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
                "phase2_import_complete": True,
                "history_imported": saved_history.exists(),
            }
        )
        run.finish()
        print(f"Imported {identifier}")


if __name__ == "__main__":
    main()
