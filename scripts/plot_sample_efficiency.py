#!/usr/bin/env python3
"""Plot sample-efficiency learning curves with variation across position splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esm2_mutation_effects.data import project_root

MODEL_STYLES = {
    "mutation_features": {
        "label": "Mutation-feature ridge",
        "color": "#0072B2",
        "marker": "o",
    },
    "frozen_esm2": {
        "label": "Frozen ESM-2 ridge",
        "color": "#D55E00",
        "marker": "s",
    },
}
REQUIRED_COLUMNS = {
    "train_fraction",
    "model",
    "n_seeds",
    "mean_labeled_train",
    "mean_test_spearman",
    "std_test_spearman",
}


def load_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if summary.duplicated(["train_fraction", "model"]).any():
        raise ValueError("Summary contains duplicate fraction/model rows")
    expected_models = {*MODEL_STYLES, "zero_shot_esm2"}
    if set(summary["model"]) != expected_models:
        raise ValueError(f"Expected exactly these models: {sorted(expected_models)}")
    if (summary["train_fraction"] <= 0).any() or (summary["train_fraction"] > 1).any():
        raise ValueError("Training fractions must be greater than 0 and at most 1")
    if summary[list(REQUIRED_COLUMNS - {"model"})].isna().any().any():
        raise ValueError("Summary contains missing numeric values")

    expected_fractions = set(summary["train_fraction"])
    for model in expected_models:
        model_fractions = set(summary.loc[summary["model"] == model, "train_fraction"])
        if model_fractions != expected_fractions:
            raise ValueError(f"Model {model} does not contain every training fraction")
    return summary.sort_values(["train_fraction", "model"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, png_path: Path, pdf_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    for model, style in MODEL_STYLES.items():
        subset = summary.loc[summary["model"] == model].sort_values("train_fraction")
        x = subset["train_fraction"].to_numpy(dtype=float) * 100
        mean = subset["mean_test_spearman"].to_numpy(dtype=float)
        std = subset["std_test_spearman"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
        )
        ax.fill_between(
            x,
            np.clip(mean - std, -1, 1),
            np.clip(mean + std, -1, 1),
            color=style["color"],
            alpha=0.16,
            linewidth=0,
        )

    zero_shot = summary.loc[summary["model"] == "zero_shot_esm2"]
    zero_mean_values = zero_shot["mean_test_spearman"].to_numpy(dtype=float)
    zero_std_values = zero_shot["std_test_spearman"].to_numpy(dtype=float)
    if not np.allclose(zero_mean_values, zero_mean_values[0]) or not np.allclose(
        zero_std_values, zero_std_values[0]
    ):
        raise ValueError("Zero-shot reference must be constant across training fractions")
    zero_mean = float(zero_mean_values[0])
    zero_std = float(zero_std_values[0])
    x_limits = (0.8, 125.0)
    ax.axhline(
        zero_mean,
        color="#009E73",
        linestyle="--",
        linewidth=2.2,
        label="Zero-shot ESM-2",
    )
    ax.fill_between(
        x_limits,
        [max(-1, zero_mean - zero_std)] * 2,
        [min(1, zero_mean + zero_std)] * 2,
        color="#009E73",
        alpha=0.11,
        linewidth=0,
    )

    fractions = sorted(summary["train_fraction"].unique())
    frozen = summary.loc[summary["model"] == "frozen_esm2"].set_index("train_fraction")
    tick_labels = []
    for fraction in fractions:
        labels = int(round(float(frozen.loc[fraction, "mean_labeled_train"])))
        tick_labels.append(f"{fraction * 100:g}%\n({labels:,})")

    n_seeds = sorted(summary["n_seeds"].astype(int).unique())
    seed_text = str(n_seeds[0]) if len(n_seeds) == 1 else "/".join(map(str, n_seeds))
    ax.set_xscale("log")
    ax.set_xlim(*x_limits)
    ax.set_xticks(np.asarray(fractions) * 100)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-0.15, 0.70)
    ax.set_xlabel("Labeled training fraction (mean fitted variants)")
    ax.set_ylabel("Test Spearman correlation (ρ)")
    ax.set_title("Position-held-out sample efficiency")
    ax.text(
        0.01,
        0.98,
        f"Mean ± 1 SD across {seed_text} position splits",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#4A4A4A",
        fontsize=9.5,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root() / "results" / "sample_efficiency" / "summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results" / "sample_efficiency" / "learning_curve.png",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=project_root() / "results" / "sample_efficiency" / "learning_curve.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(
            f"Summary not found: {args.summary}. Run scripts/run_sample_efficiency.py first."
        )
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    summary = load_summary(args.summary)
    plot_summary(summary, args.output, args.pdf_output, args.dpi)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.pdf_output}")


if __name__ == "__main__":
    main()
