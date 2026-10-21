#!/usr/bin/env python3
"""Validate and analyze the protocol-locked Phase 2 summary."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from esm2_mutation_effects.data import project_root

CONFIGURATIONS = ("frozen_listmle", "lora_listmle")
SPLIT_SEEDS = (42, 43, 44, 45, 46)
SAMPLING_SEEDS = (1, 2, 3)
LABEL_BUDGETS = (20, 40, 200, 401, 1002, 2004, 4008)
FULL_LABEL_BUDGET = 4008
EXPECTED_PROTOCOL_SHA256 = "ecd213c9645890ca90656059867916c78e3a879cec72e9ebf1b46ca2d60cddad"
METRICS = (
    "overall_spearman",
    "between_position_spearman",
    "within_position_centered_spearman",
    "within_position_pairwise_accuracy",
    "within_position_spearman_mean",
)
MODEL_STYLES = {
    "frozen_listmle": {
        "label": "Frozen ESM-2 + ListMLE",
        "color": "#D55E00",
        "marker": "s",
    },
    "lora_listmle": {
        "label": "LoRA ESM-2 + ListMLE",
        "color": "#0072B2",
        "marker": "o",
    },
}


def expected_identities() -> set[tuple[str, int, int | None, int]]:
    identities = set()
    for configuration in CONFIGURATIONS:
        for split_seed in SPLIT_SEEDS:
            for budget in LABEL_BUDGETS:
                samples = (None,) if budget == FULL_LABEL_BUDGET else SAMPLING_SEEDS
                for sampling_seed in samples:
                    identities.add((configuration, split_seed, sampling_seed, budget))
    return identities


def load_and_validate(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text())
    expected_header = {
        "experiment": "phase_2_protocol_locked_label_efficiency",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "expected_supervised_runs": 190,
        "completed_supervised_runs": 190,
        "matrix_complete": True,
        "test_evaluated": True,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_header.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 2 completion metadata mismatch: {mismatches}")

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != 190:
        raise ValueError("Phase 2 summary must contain exactly 190 supervised runs")
    observed = [
        (
            str(run["configuration"]),
            int(run["split_seed"]),
            run["sampling_seed"],
            int(run["label_budget"]),
        )
        for run in runs
    ]
    if len(set(observed)) != len(observed):
        raise ValueError("Phase 2 summary contains duplicate run identities")
    expected = expected_identities()
    if set(observed) != expected:
        raise ValueError(
            "Phase 2 run identities do not match the protocol: "
            f"missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )

    for run in runs:
        if int(run["n_labeled_train"]) != int(run["label_budget"]):
            raise ValueError("A Phase 2 run has an incorrect labeled-training count")
        fingerprint = run.get("training_subset_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("A Phase 2 run has an invalid subset fingerprint")
        metrics = run.get("test_metrics")
        if not isinstance(metrics, dict) or metrics.get("subset") != "test":
            raise ValueError("A Phase 2 run is missing test metrics")
        if metrics.get("overall_spearman") is None:
            raise ValueError("A Phase 2 run is missing overall Spearman")
        for value in metrics.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("A Phase 2 run contains a non-finite metric")

    zero_shot = summary.get("zero_shot_reference")
    if not isinstance(zero_shot, list) or {row["split_seed"] for row in zero_shot} != set(
        SPLIT_SEEDS
    ):
        raise ValueError("Zero-shot reference must contain the five position splits")
    return summary


def supervised_split_means(
    summary: dict[str, object],
) -> dict[tuple[str, int, int, str], float]:
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for run in summary["runs"]:
        for metric in METRICS:
            value = run["test_metrics"].get(metric)
            if value is not None:
                grouped[
                    (
                        str(run["configuration"]),
                        int(run["label_budget"]),
                        int(run["split_seed"]),
                        metric,
                    )
                ].append(float(value))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def zero_shot_by_split(summary: dict[str, object]) -> dict[int, dict[str, object]]:
    return {int(row["split_seed"]): row["test_metrics"] for row in summary["zero_shot_reference"]}


def metric_summary(summary: dict[str, object]) -> pd.DataFrame:
    split_means = supervised_split_means(summary)
    records = []
    for configuration in CONFIGURATIONS:
        for budget in LABEL_BUDGETS:
            for metric in METRICS:
                values = [
                    split_means[(configuration, budget, split_seed, metric)]
                    for split_seed in SPLIT_SEEDS
                    if (configuration, budget, split_seed, metric) in split_means
                ]
                records.append(
                    {
                        "configuration": configuration,
                        "label_budget": budget,
                        "metric": metric,
                        "n_splits": len(values),
                        "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
    zero_shot = zero_shot_by_split(summary)
    for metric in METRICS:
        values = [
            float(zero_shot[split_seed][metric])
            for split_seed in SPLIT_SEEDS
            if zero_shot[split_seed].get(metric) is not None
        ]
        records.append(
            {
                "configuration": "zero_shot_esm2",
                "label_budget": 0,
                "metric": metric,
                "n_splits": len(values),
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return pd.DataFrame.from_records(records)


def _difference_record(prefix: str, values: list[float]) -> dict[str, object]:
    tolerance = 1e-12
    return {
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_sample_std": statistics.stdev(values),
        f"{prefix}_minimum": min(values),
        f"{prefix}_maximum": max(values),
        f"{prefix}_wins": sum(value > tolerance for value in values),
        f"{prefix}_ties": sum(abs(value) <= tolerance for value in values),
        f"{prefix}_losses": sum(value < -tolerance for value in values),
    }


def paired_comparisons(summary: dict[str, object]) -> pd.DataFrame:
    split_means = supervised_split_means(summary)
    zero_shot = zero_shot_by_split(summary)
    records = []
    for budget in LABEL_BUDGETS:
        frozen = {
            split_seed: split_means[("frozen_listmle", budget, split_seed, "overall_spearman")]
            for split_seed in SPLIT_SEEDS
        }
        lora = {
            split_seed: split_means[("lora_listmle", budget, split_seed, "overall_spearman")]
            for split_seed in SPLIT_SEEDS
        }
        zero = {
            split_seed: float(zero_shot[split_seed]["overall_spearman"])
            for split_seed in SPLIT_SEEDS
        }
        lora_zero = [lora[seed] - zero[seed] for seed in SPLIT_SEEDS]
        lora_frozen = [lora[seed] - frozen[seed] for seed in SPLIT_SEEDS]
        frozen_zero = [frozen[seed] - zero[seed] for seed in SPLIT_SEEDS]
        records.append(
            {
                "label_budget": budget,
                "zero_shot_mean": statistics.mean(zero.values()),
                "frozen_mean": statistics.mean(frozen.values()),
                "lora_mean": statistics.mean(lora.values()),
                **_difference_record("lora_minus_zero_shot", lora_zero),
                **_difference_record("lora_minus_frozen", lora_frozen),
                **_difference_record("frozen_minus_zero_shot", frozen_zero),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_results(
    summary: dict[str, object],
    paired: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    aggregate = pd.DataFrame(summary["aggregate_summary"])
    zero_values = [
        float(row["test_metrics"]["overall_spearman"])
        for row in summary["zero_shot_reference"]
    ]
    zero_mean = statistics.mean(zero_values)
    zero_std = statistics.stdev(zero_values)

    fig, (learning_ax, gain_ax) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        constrained_layout=True,
    )
    for configuration, style in MODEL_STYLES.items():
        selected = aggregate.loc[aggregate["configuration"] == configuration].sort_values(
            "label_budget"
        )
        x = selected["label_budget"].to_numpy(dtype=float)
        x_offset = 0.97 if configuration == "frozen_listmle" else 1.03
        mean = selected["mean_test_spearman"].to_numpy(dtype=float)
        std = selected["sample_std_test_spearman"].to_numpy(dtype=float)
        learning_ax.errorbar(
            x * x_offset,
            mean,
            yerr=std,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
            capsize=3,
            label=style["label"],
        )
    learning_ax.axhline(
        zero_mean,
        color="#009E73",
        linestyle="--",
        linewidth=2.2,
        label=f"Zero-shot ESM-2 (split SD {zero_std:.3f})",
    )
    learning_ax.set_xscale("log")
    learning_ax.set_xticks(LABEL_BUDGETS)
    learning_ax.set_xticklabels([f"{budget:,}" for budget in LABEL_BUDGETS], rotation=30)
    learning_ax.set_xlabel("Labeled training mutations")
    learning_ax.set_ylabel("Test Spearman correlation (ρ)")
    learning_ax.set_ylim(0.25, 0.70)
    learning_ax.set_title("Phase 2 label efficiency")
    learning_ax.legend(frameon=False, loc="upper right")

    x = paired["label_budget"].to_numpy(dtype=float)
    comparisons = (
        ("lora_minus_zero_shot", "LoRA − zero-shot", "#0072B2", "o"),
        ("lora_minus_frozen", "LoRA − frozen", "#CC79A7", "D"),
    )
    for prefix, label, color, marker in comparisons:
        gain_ax.errorbar(
            x,
            paired[f"{prefix}_mean"].to_numpy(dtype=float),
            yerr=paired[f"{prefix}_sample_std"].to_numpy(dtype=float),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            capsize=3,
            label=label,
        )
    gain_ax.axhline(0.0, color="#555555", linewidth=1.2)
    gain_ax.axhline(
        0.02,
        color="#009E73",
        linestyle="--",
        linewidth=1.8,
        label="Pre-specified practical gain (+0.02)",
    )
    gain_ax.set_xscale("log")
    gain_ax.set_xticks(LABEL_BUDGETS)
    gain_ax.set_xticklabels([f"{budget:,}" for budget in LABEL_BUDGETS], rotation=30)
    gain_ax.set_xlabel("Labeled training mutations")
    gain_ax.set_ylabel("Paired test Spearman difference")
    gain_ax.set_title("LoRA advantage by label budget")
    gain_ax.text(
        0.02,
        0.98,
        "Mean ± 1 SD across paired split-level differences",
        transform=gain_ax.transAxes,
        va="top",
        color="#4A4A4A",
        fontsize=9,
    )
    gain_ax.legend(frameon=False, loc="lower right")

    for axis in (learning_ax, gain_ax):
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def analysis_record(
    summary: dict[str, object],
    paired: pd.DataFrame,
    source_path: Path,
) -> dict[str, object]:
    threshold_rows = paired.loc[paired["lora_minus_zero_shot_mean"] >= 0.02]
    first_practical_budget = (
        int(threshold_rows.iloc[0]["label_budget"]) if len(threshold_rows) else None
    )
    best_index = paired["lora_mean"].idxmax()
    best = paired.loc[best_index]
    aggregates = pd.DataFrame(summary["aggregate_summary"])
    lora_2004 = aggregates.loc[
        (aggregates["configuration"] == "lora_listmle")
        & (aggregates["label_budget"] == 2004)
    ].iloc[0]
    return {
        "source": str(source_path.resolve()),
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "protocol_sha256": summary["protocol_sha256"],
        "validated_supervised_runs": len(summary["runs"]),
        "matrix_complete": True,
        "zero_shot_mean_test_spearman": float(paired.iloc[0]["zero_shot_mean"]),
        "smallest_budget_with_practical_lora_gain_over_zero_shot": first_practical_budget,
        "best_lora_budget": int(best["label_budget"]),
        "best_lora_mean_test_spearman": float(best["lora_mean"]),
        "best_lora_gain_over_zero_shot": float(best["lora_minus_zero_shot_mean"]),
        "lora_2004_sample_std_across_splits": float(lora_2004["sample_std_test_spearman"]),
        "lora_2004_mean_within_split_sampling_std": float(
            lora_2004["mean_within_split_sampling_std"]
        ),
        "test_results_used_for_protocol_changes": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root() / "results" / "phase2_label_efficiency" / "phase2_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "results" / "phase2_label_efficiency" / "analysis",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if not args.summary.exists():
        raise FileNotFoundError(f"Phase 2 summary not found: {args.summary}")

    summary = load_and_validate(args.summary)
    metric_table = metric_summary(summary)
    paired = paired_comparisons(summary)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["aggregate_summary"]).to_csv(
        output_dir / "aggregate_summary.csv", index=False
    )
    pd.DataFrame(summary["split_level_summary"]).to_csv(
        output_dir / "split_level_summary.csv", index=False
    )
    metric_table.to_csv(output_dir / "metric_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    plot_results(
        summary,
        paired,
        output_dir / "phase2_learning_curves.png",
        output_dir / "phase2_learning_curves.pdf",
        args.dpi,
    )
    record = analysis_record(summary, paired, args.summary)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Wrote Phase 2 analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
