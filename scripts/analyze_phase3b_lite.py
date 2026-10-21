"""Combine Phase 3A with the four new Phase 3B-lite split results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import project_root
from esm2_mutation_effects.phase3b_lite import (
    HISTORICAL_PHASE3A_SPLIT_SEED,
    METHODS,
    NEW_SPLIT_SEEDS,
)

METRICS = (
    "overall_spearman",
    "between_position_spearman",
    "within_position_centered_spearman",
    "within_position_pairwise_accuracy",
    "within_position_spearman_mean",
)


def args_() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase3a-summary",
        type=Path,
        default=root / "results" / "phase3a_pilot" / "run" / "summary.json",
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=root / "results" / "phase3b_lite" / "matrix_summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "phase3b_lite" / "analysis"
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required completed summary is missing: {path}")
    return json.loads(path.read_text())


def rows_for_split(summary: dict[str, object], split_seed: int) -> list[dict[str, object]]:
    if summary.get("test_evaluated") is not True:
        raise ValueError(f"Split {split_seed} does not record a completed test evaluation")
    rows = summary.get("results")
    if not isinstance(rows, list) or {row.get("method") for row in rows} != set(METHODS):
        raise ValueError(f"Split {split_seed} does not contain exactly the required methods")
    return [{"split_seed": split_seed, **row} for row in rows]


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n_splits": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def comparison(name: str, left: pd.Series, right: pd.Series) -> dict[str, object]:
    differences = (left - right).tolist()
    tolerance = 1e-12
    return {
        "comparison": name,
        **describe(differences),
        "wins": sum(value > tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "losses": sum(value < -tolerance for value in differences),
    }


def main() -> None:
    args = args_()
    phase3a = read_summary(args.phase3a_summary)
    matrix = read_summary(args.matrix_summary)
    if matrix.get("matrix_complete") is not True or matrix.get("test_evaluated") is not True:
        raise ValueError("Phase 3B-lite matrix is not complete")
    expected_new = list(NEW_SPLIT_SEEDS)
    if matrix.get("completed_new_splits") != expected_new:
        raise ValueError("Phase 3B-lite matrix does not contain the pre-registered new splits")

    records = rows_for_split(phase3a, HISTORICAL_PHASE3A_SPLIT_SEED)
    for run in matrix.get("runs", []):
        split_seed = int(run.get("split_seed"))
        if split_seed not in NEW_SPLIT_SEEDS:
            raise ValueError(f"Unexpected Phase 3B-lite split seed {split_seed}")
        records.extend(rows_for_split(run, split_seed))
    table = pd.DataFrame.from_records(records).sort_values(["split_seed", "method"])
    if set(table.split_seed) != {HISTORICAL_PHASE3A_SPLIT_SEED, *NEW_SPLIT_SEEDS}:
        raise ValueError("Combined Phase 3 result is missing a required split")

    metric_rows = []
    for method in METHODS:
        for metric in METRICS:
            values = table.loc[table.method == method, metric].dropna().astype(float).tolist()
            metric_rows.append({"method": method, "metric": metric, **describe(values)})
    metric_summary = pd.DataFrame.from_records(metric_rows)
    overall = table.pivot(index="split_seed", columns="method", values="overall_spearman")
    comparisons = [
        comparison("pooled_minus_standard", overall["pooled"], overall["standard"]),
        comparison("reptile_minus_standard", overall["reptile"], overall["standard"]),
        comparison("reptile_minus_pooled", overall["reptile"], overall["pooled"]),
        comparison("pooled_minus_zero_shot", overall["pooled"], overall["zero_shot"]),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "per_split_metrics.csv", index=False)
    metric_summary.to_csv(args.output_dir / "metric_summary.csv", index=False)
    output = {
        "experiment": "phase3b_lite_repeated_target_split_replication",
        "historical_phase3a_split_seed": HISTORICAL_PHASE3A_SPLIT_SEED,
        "new_split_seeds": list(NEW_SPLIT_SEEDS),
        "n_total_splits": 1 + len(NEW_SPLIT_SEEDS),
        "primary_comparison": comparisons[0],
        "secondary_comparisons": comparisons[1:],
        "files": {
            "per_split_metrics": str((args.output_dir / "per_split_metrics.csv").resolve()),
            "metric_summary": str((args.output_dir / "metric_summary.csv").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
