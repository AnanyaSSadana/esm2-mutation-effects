#!/usr/bin/env python3
"""Repeat sample-efficiency curves across position splits and training subsets."""

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
    make_frozen_esm2_model,
    make_mutation_feature_model,
    select_alpha,
)
from esm2_mutation_effects.data import load_assay, mutation_table, project_root
from esm2_mutation_effects.features import (
    CACHED_FEATURE_DEFINITION,
    MODEL_FEATURE_DEFINITION,
    select_nonredundant_site_features,
)
from esm2_mutation_effects.metrics import spearman
from esm2_mutation_effects.sampling import nested_training_subsets
from esm2_mutation_effects.splits import make_splits, validate_splits

DEFAULT_SPLIT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_SAMPLING_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
MODEL_ORDER = ("mutation_features", "zero_shot_esm2", "frozen_esm2")


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


def load_frozen_features(path: Path, assay: pd.DataFrame) -> np.ndarray:
    cache = np.load(path, allow_pickle=False)
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
    features = select_nonredundant_site_features(cache["features"])
    return features[aligned["row"].to_numpy()]


def combined_sampling_seed(split_seed: int, sampling_seed: int) -> int:
    """Derive a stable RNG seed from independent split and sampling seeds."""

    sequence = np.random.SeedSequence([split_seed, sampling_seed])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction in sorted(results["train_fraction"].unique()):
        for model in MODEL_ORDER:
            subset = results[
                (results["train_fraction"] == fraction) & (results["model"] == model)
            ]
            values = subset["test_spearman"].to_numpy(dtype=np.float64)
            if len(values) == 0:
                raise ValueError(f"No results for model {model} at fraction {fraction}")

            split_means = subset.groupby("split_seed")["test_spearman"].mean().to_numpy()
            within_split_stds = []
            if model != "zero_shot_esm2":
                for _, split_rows in subset.groupby("split_seed"):
                    split_values = split_rows["test_spearman"].to_numpy(dtype=np.float64)
                    within_split_stds.append(
                        float(np.std(split_values, ddof=1 if len(split_values) > 1 else 0))
                    )

            sampling_seeds = subset["sampling_seed"].dropna().astype(int).unique()
            n_split_seeds = int(subset["split_seed"].nunique())
            rows.append(
                {
                    "train_fraction": float(fraction),
                    "model": model,
                    "n_seeds": n_split_seeds,
                    "n_split_seeds": n_split_seeds,
                    "n_sampling_seeds": int(len(sampling_seeds)),
                    "n_runs": int(len(values)),
                    "mean_labeled_train": float(subset["n_labeled_train"].mean()),
                    "mean_test_spearman": float(np.mean(values)),
                    "std_test_spearman": float(
                        np.std(values, ddof=1 if len(values) > 1 else 0)
                    ),
                    "min_test_spearman": float(np.min(values)),
                    "max_test_spearman": float(np.max(values)),
                    "between_split_std_of_means": float(
                        np.std(split_means, ddof=1 if len(split_means) > 1 else 0)
                    ),
                    "mean_within_split_sampling_std": (
                        float(np.mean(within_split_stds)) if within_split_stds else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS)
    )
    parser.add_argument(
        "--sampling-seeds", nargs="+", type=int, default=list(DEFAULT_SAMPLING_SEEDS)
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS)
    )
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
        default=project_root() / "results" / "repeated_sample_efficiency",
    )
    args = parser.parse_args()

    if not args.split_seeds or not args.sampling_seeds:
        raise ValueError("At least one split seed and one sampling seed are required")
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError("Split seeds must be unique")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        raise ValueError("Sampling seeds must be unique")
    if any(seed < 0 for seed in [*args.split_seeds, *args.sampling_seeds]):
        raise ValueError("Seeds must be non-negative")
    fractions = sorted(float(fraction) for fraction in args.fractions)
    if not fractions:
        raise ValueError("At least one training fraction is required")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Training fractions must be unique")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("Training fractions must be greater than 0 and at most 1")
    if not args.features.exists():
        raise FileNotFoundError(f"Frozen feature cache not found: {args.features}")
    if not args.zero_shot.exists():
        raise FileNotFoundError(f"Zero-shot predictions not found: {args.zero_shot}")

    assay = load_assay()
    y = assay["DMS_score"].to_numpy(dtype=np.float64)
    zero_shot = load_zero_shot_scores(args.zero_shot, assay)
    frozen = load_frozen_features(args.features, assay)
    mutation_features = mutation_table(assay)
    mutation_features["position_fraction"] = (
        mutation_features["position"] / mutation_features["position"].max()
    )
    mutation_columns = list(MUTATION_FEATURE_COLUMNS)
    results: list[dict[str, object]] = []

    for split_seed in args.split_seeds:
        split_manifest = make_splits(assay, seed=split_seed)
        validate_splits(split_manifest, set(assay["mutant"]))
        labels = assay[["mutant"]].merge(
            split_manifest[["mutant", "position"]].rename(columns={"position": "split"}),
            on="mutant",
            validate="one_to_one",
        )["split"]
        available_train = np.flatnonzero((labels == "train").to_numpy())
        validation = np.flatnonzero((labels == "validation").to_numpy())
        test = np.flatnonzero((labels == "test").to_numpy())
        zero_shot_test_spearman = spearman(y[test], zero_shot[test])

        for fraction in fractions:
            results.append(
                {
                    "split_seed": int(split_seed),
                    "sampling_seed": None,
                    "train_fraction": float(fraction),
                    "model": "zero_shot_esm2",
                    "n_available_train": int(len(available_train)),
                    "n_labeled_train": 0,
                    "n_validation": 0,
                    "n_test": int(len(test)),
                    "selected_alpha": None,
                    "validation_spearman": None,
                    "test_spearman": zero_shot_test_spearman,
                }
            )

        for sampling_seed in args.sampling_seeds:
            derived_seed = combined_sampling_seed(split_seed, sampling_seed)
            training_subsets = nested_training_subsets(
                available_train,
                fractions,
                seed=derived_seed,
            )
            for fraction in fractions:
                # Every sampling seed produces the identical full training set.
                # Evaluate it once per split rather than treating duplicates as runs.
                is_full_training = np.isclose(fraction, 1.0)
                if is_full_training and sampling_seed != args.sampling_seeds[0]:
                    continue
                train = training_subsets[fraction]

                selected_mutation, _ = select_alpha(
                    make_mutation_feature_model,
                    MUTATION_ALPHAS,
                    mutation_features.iloc[train][mutation_columns],
                    y[train],
                    mutation_features.iloc[validation][mutation_columns],
                    y[validation],
                )
                mutation_model = make_mutation_feature_model(selected_mutation["alpha"])
                mutation_model.fit(mutation_features.iloc[train][mutation_columns], y[train])
                mutation_prediction = mutation_model.predict(
                    mutation_features.iloc[test][mutation_columns]
                )
                results.append(
                    {
                        "split_seed": int(split_seed),
                        "sampling_seed": None if is_full_training else int(sampling_seed),
                        "train_fraction": float(fraction),
                        "model": "mutation_features",
                        "n_available_train": int(len(available_train)),
                        "n_labeled_train": int(len(train)),
                        "n_validation": int(len(validation)),
                        "n_test": int(len(test)),
                        "selected_alpha": selected_mutation["alpha"],
                        "validation_spearman": selected_mutation["validation_spearman"],
                        "test_spearman": spearman(y[test], mutation_prediction),
                    }
                )

                selected_frozen, _ = select_alpha(
                    make_frozen_esm2_model,
                    FROZEN_ESM2_ALPHAS,
                    frozen[train],
                    y[train],
                    frozen[validation],
                    y[validation],
                )
                frozen_model = make_frozen_esm2_model(selected_frozen["alpha"])
                frozen_model.fit(frozen[train], y[train])
                frozen_prediction = frozen_model.predict(frozen[test])
                results.append(
                    {
                        "split_seed": int(split_seed),
                        "sampling_seed": None if is_full_training else int(sampling_seed),
                        "train_fraction": float(fraction),
                        "model": "frozen_esm2",
                        "n_available_train": int(len(available_train)),
                        "n_labeled_train": int(len(train)),
                        "n_validation": int(len(validation)),
                        "n_test": int(len(test)),
                        "selected_alpha": selected_frozen["alpha"],
                        "validation_spearman": selected_frozen["validation_spearman"],
                        "test_spearman": spearman(y[test], frozen_prediction),
                    }
                )
                print(
                    f"Completed split {split_seed}, sample {sampling_seed}, "
                    f"fraction {fraction:g}",
                    flush=True,
                )

    per_run = pd.DataFrame(results)
    summary = summarize(per_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.output_dir / "per_run.csv"
    summary_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "experiment": "repeated_position_held_out_sample_efficiency",
        "split_seeds": [int(seed) for seed in args.split_seeds],
        "sampling_seeds": [int(seed) for seed in args.sampling_seeds],
        "training_fractions": fractions,
        "sampling_unit": "individual labeled mutation rows within training positions",
        "nested_training_subsets": True,
        "full_training_policy": (
            "100% is evaluated once per split because all sampling seeds select "
            "the identical training set"
        ),
        "validation_policy": (
            "fixed validation positions select alpha; validation rows are not included "
            "in the final fit"
        ),
        "test_policy": "fixed position-held-out test set per split seed",
        "zero_shot_policy": (
            "one no-label result per split seed; not duplicated across sampling seeds"
        ),
        "mutation_alphas": list(MUTATION_ALPHAS),
        "frozen_esm2_alphas": list(FROZEN_ESM2_ALPHAS),
        "frozen_model_feature_definition": MODEL_FEATURE_DEFINITION,
        "per_run": json.loads(per_run.to_json(orient="records", double_precision=15)),
        "summary": json.loads(summary.to_json(orient="records", double_precision=15)),
        "files": {
            "per_run_csv": str(per_run_path),
            "summary_csv": str(summary_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    console_payload = {
        "experiment": payload["experiment"],
        "split_seeds": payload["split_seeds"],
        "sampling_seeds": payload["sampling_seeds"],
        "training_fractions": payload["training_fractions"],
        "summary": payload["summary"],
        "files": {**payload["files"], "summary_json": str(json_path)},
    }
    print(json.dumps(console_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
