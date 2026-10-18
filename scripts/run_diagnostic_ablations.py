#!/usr/bin/env python3
"""Diagnose position-level and substitution-level signal in current baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esm2_mutation_effects.baselines import (
    FROZEN_ESM2_ALPHAS,
    MUTATION_ALPHAS,
    MUTATION_FEATURE_COLUMNS,
    POSITION_CONTROL_ALPHAS,
    make_categorical_position_model,
    make_frozen_esm2_model,
    make_mutation_feature_model,
    select_alpha,
)
from esm2_mutation_effects.data import load_assay, mutation_table, project_root
from esm2_mutation_effects.diagnostics import position_ranking_metrics
from esm2_mutation_effects.features import CACHED_FEATURE_DEFINITION, split_site_feature_blocks
from esm2_mutation_effects.splits import make_splits, validate_splits

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
SPLIT_ORDER = ("random", "position")
MODEL_ORDER = (
    "position_mean",
    "mutation_features",
    "categorical_position_mutation",
    "zero_shot_esm2",
    "frozen_wild_type_only",
    "frozen_difference_only",
    "frozen_wild_type_difference",
)
METRIC_COLUMNS = (
    "overall_spearman",
    "between_position_spearman",
    "within_position_centered_spearman",
    "within_position_pairwise_accuracy",
    "within_position_spearman_mean",
)


def load_zero_shot_scores(path: Path, assay: pd.DataFrame) -> np.ndarray:
    predictions = pd.read_csv(path)
    required = {"mutant", "predicted_score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing zero-shot columns: {sorted(missing)}")
    if predictions["mutant"].duplicated().any():
        raise ValueError("Zero-shot predictions contain duplicate mutants")
    if set(predictions["mutant"]) != set(assay["mutant"]):
        raise ValueError("Zero-shot predictions must contain every assay mutant exactly once")
    aligned = assay[["mutant"]].merge(
        predictions[["mutant", "predicted_score"]],
        on="mutant",
        validate="one_to_one",
    )
    if aligned["predicted_score"].isna().any():
        raise ValueError("Zero-shot predictions contain missing scores")
    return aligned["predicted_score"].to_numpy(dtype=np.float64)


def load_feature_blocks(path: Path, assay: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        cached_definition = str(cache["feature_definition"].item())
        if cached_definition != CACHED_FEATURE_DEFINITION:
            raise ValueError(
                f"Unexpected cached feature definition {cached_definition!r}; "
                f"expected {CACHED_FEATURE_DEFINITION!r}"
            )
        cached_mutants = pd.DataFrame(
            {"mutant": cache["mutant"].astype(str), "row": range(len(cache["mutant"]))}
        )
        if cached_mutants["mutant"].duplicated().any():
            raise ValueError("Frozen feature cache contains duplicate mutants")
        if set(cached_mutants["mutant"]) != set(assay["mutant"]):
            raise ValueError("Frozen feature cache must contain every assay mutant exactly once")
        aligned = assay[["mutant"]].merge(cached_mutants, on="mutant", validate="one_to_one")
        blocks = split_site_feature_blocks(cache["features"])
        rows = aligned["row"].to_numpy()
        return {name: values[rows] for name, values in blocks.items()}


def position_mean_prediction(
    fit_positions: np.ndarray,
    fit_scores: np.ndarray,
    test_positions: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame({"position": fit_positions, "score": fit_scores})
    means = frame.groupby("position")["score"].mean()
    fallback = float(frame["score"].mean())
    return np.asarray([means.get(position, fallback) for position in test_positions])


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_strategy in SPLIT_ORDER:
        for model in MODEL_ORDER:
            subset = per_run[
                (per_run["split_strategy"] == split_strategy) & (per_run["model"] == model)
            ]
            if subset.empty:
                raise ValueError(f"No results for {split_strategy}/{model}")
            row: dict[str, object] = {
                "split_strategy": split_strategy,
                "model": model,
                "n_seeds": int(subset["seed"].nunique()),
                "mean_n_test": float(subset["n_test"].mean()),
                "mean_n_test_positions": float(subset["n_test_positions"].mean()),
                "mean_n_within_position_pairs": float(
                    subset["n_within_position_pairs"].mean()
                ),
                "mean_n_positions_with_defined_spearman": float(
                    subset["n_positions_with_defined_spearman"].mean()
                ),
            }
            for metric in METRIC_COLUMNS:
                values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy()
                row[f"n_defined_{metric}"] = int(len(values))
                row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else None
                row[f"std_{metric}"] = (
                    float(np.std(values, ddof=1 if len(values) > 1 else 0))
                    if len(values)
                    else None
                )
                row[f"min_{metric}"] = float(np.min(values)) if len(values) else None
                row[f"max_{metric}"] = float(np.max(values)) if len(values) else None
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root() / "data" / "processed" / "esm2_t6_site_features.npz",
    )
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=project_root() / "results" / "esm2_masked_marginal.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "results" / "diagnostic_ablations",
    )
    args = parser.parse_args()

    if not args.seeds:
        raise ValueError("At least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative")
    if not args.features.exists():
        raise FileNotFoundError(f"Frozen feature cache not found: {args.features}")
    if not args.zero_shot.exists():
        raise FileNotFoundError(f"Zero-shot predictions not found: {args.zero_shot}")

    assay = load_assay()
    y = assay["DMS_score"].to_numpy(dtype=np.float64)
    zero_shot = load_zero_shot_scores(args.zero_shot, assay)
    feature_blocks = load_feature_blocks(args.features, assay)
    frozen_features = {
        "frozen_wild_type_only": feature_blocks["wild_type"],
        "frozen_difference_only": feature_blocks["difference"],
        "frozen_wild_type_difference": np.concatenate(
            [feature_blocks["wild_type"], feature_blocks["difference"]], axis=1
        ),
    }
    mutation_features = mutation_table(assay)
    mutation_features["position_fraction"] = (
        mutation_features["position"] / mutation_features["position"].max()
    )
    positions = mutation_features["position"].to_numpy(dtype=np.int64)
    mutation_columns = list(MUTATION_FEATURE_COLUMNS)
    categorical_columns = ["wild_type", "alternate", "position"]

    run_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    def record_result(
        seed: int,
        split_strategy: str,
        model: str,
        test_indices: np.ndarray,
        prediction: np.ndarray,
        selected_alpha: float | None = None,
        validation_spearman: float | None = None,
        n_train: int = 0,
        n_validation: int = 0,
    ) -> None:
        metrics = position_ranking_metrics(
            y[test_indices],
            np.asarray(prediction, dtype=np.float64),
            positions[test_indices],
        )
        run_rows.append(
            {
                "seed": int(seed),
                "split_strategy": split_strategy,
                "model": model,
                "n_train": int(n_train),
                "n_validation": int(n_validation),
                "n_test": int(len(test_indices)),
                "selected_alpha": selected_alpha,
                "validation_spearman": validation_spearman,
                **metrics,
            }
        )
        output = assay.iloc[test_indices][["mutant", "DMS_score"]].copy()
        output.insert(0, "model", model)
        output.insert(0, "split_strategy", split_strategy)
        output.insert(0, "seed", int(seed))
        output["position"] = positions[test_indices]
        output["predicted_score"] = np.asarray(prediction, dtype=np.float64)
        prediction_frames.append(output)

    for seed in args.seeds:
        split_manifest = make_splits(assay, seed=seed)
        validate_splits(split_manifest, set(assay["mutant"]))
        aligned_splits = assay[["mutant"]].merge(
            split_manifest[["mutant", "random", "position"]],
            on="mutant",
            validate="one_to_one",
        )

        for split_strategy in SPLIT_ORDER:
            labels = aligned_splits[split_strategy]
            train = np.flatnonzero((labels == "train").to_numpy())
            validation = np.flatnonzero((labels == "validation").to_numpy())
            test = np.flatnonzero((labels == "test").to_numpy())
            fit = np.concatenate([train, validation])

            position_prediction = position_mean_prediction(
                positions[fit], y[fit], positions[test]
            )
            record_result(
                seed,
                split_strategy,
                "position_mean",
                test,
                position_prediction,
                n_train=len(train),
                n_validation=len(validation),
            )

            selected_mutation, _ = select_alpha(
                make_mutation_feature_model,
                MUTATION_ALPHAS,
                mutation_features.iloc[train][mutation_columns],
                y[train],
                mutation_features.iloc[validation][mutation_columns],
                y[validation],
            )
            mutation_model = make_mutation_feature_model(selected_mutation["alpha"])
            mutation_model.fit(mutation_features.iloc[fit][mutation_columns], y[fit])
            record_result(
                seed,
                split_strategy,
                "mutation_features",
                test,
                mutation_model.predict(mutation_features.iloc[test][mutation_columns]),
                selected_mutation["alpha"],
                selected_mutation["validation_spearman"],
                len(train),
                len(validation),
            )

            selected_position, _ = select_alpha(
                make_categorical_position_model,
                POSITION_CONTROL_ALPHAS,
                mutation_features.iloc[train][categorical_columns],
                y[train],
                mutation_features.iloc[validation][categorical_columns],
                y[validation],
            )
            position_model = make_categorical_position_model(selected_position["alpha"])
            position_model.fit(mutation_features.iloc[fit][categorical_columns], y[fit])
            record_result(
                seed,
                split_strategy,
                "categorical_position_mutation",
                test,
                position_model.predict(mutation_features.iloc[test][categorical_columns]),
                selected_position["alpha"],
                selected_position["validation_spearman"],
                len(train),
                len(validation),
            )

            record_result(
                seed,
                split_strategy,
                "zero_shot_esm2",
                test,
                zero_shot[test],
            )

            for model_name, values in frozen_features.items():
                selected_frozen, _ = select_alpha(
                    make_frozen_esm2_model,
                    FROZEN_ESM2_ALPHAS,
                    values[train],
                    y[train],
                    values[validation],
                    y[validation],
                )
                frozen_model = make_frozen_esm2_model(selected_frozen["alpha"])
                frozen_model.fit(values[fit], y[fit])
                record_result(
                    seed,
                    split_strategy,
                    model_name,
                    test,
                    frozen_model.predict(values[test]),
                    selected_frozen["alpha"],
                    selected_frozen["validation_spearman"],
                    len(train),
                    len(validation),
                )

            print(f"Completed seed {seed}, split {split_strategy}", flush=True)

    per_run = pd.DataFrame(run_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(per_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.output_dir / "per_run.csv"
    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    per_run.to_csv(per_run_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "experiment": "representation_and_position_diagnostic_ablations",
        "seeds": [int(seed) for seed in args.seeds],
        "split_strategies": list(SPLIT_ORDER),
        "models": list(MODEL_ORDER),
        "within_position_policy": {
            "pairwise_accuracy": (
                "pooled within-position pairs; true-score ties excluded; "
                "prediction ties receive half credit"
            ),
            "per_position_spearman_minimum_variants": 3,
            "undefined_correlations": "recorded as null, never replaced with zero",
        },
        "position_mean_policy": (
            "fit on training plus validation labels; unseen positions fall back "
            "to the global fit-set mean"
        ),
        "per_run": json.loads(per_run.to_json(orient="records", double_precision=15)),
        "summary": json.loads(summary.to_json(orient="records", double_precision=15)),
        "files": {
            "per_run_csv": str(per_run_path),
            "predictions_csv": str(predictions_path),
            "summary_csv": str(summary_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    console_payload = {
        "experiment": payload["experiment"],
        "seeds": payload["seeds"],
        "summary": payload["summary"],
        "files": {**payload["files"], "summary_json": str(json_path)},
    }
    print(json.dumps(console_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
