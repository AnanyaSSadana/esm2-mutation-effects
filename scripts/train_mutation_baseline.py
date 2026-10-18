#!/usr/bin/env python3
"""Ridge regression on simple mutation identity and position features."""

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.baselines import (
    MUTATION_ALPHAS,
    MUTATION_FEATURE_COLUMNS,
    make_mutation_feature_model,
    select_alpha,
)
from esm2_mutation_effects.data import load_assay, mutation_table, project_root
from esm2_mutation_effects.metrics import spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-strategy", choices=("random", "position"), default="position")
    parser.add_argument(
        "--splits", type=Path, default=project_root() / "data" / "splits" / "splits.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=project_root() / "results")
    args = parser.parse_args()

    assay = load_assay()
    features = mutation_table(assay)
    features["position_fraction"] = features["position"] / features["position"].max()
    data = features.merge(assay[["mutant", "DMS_score"]], on="mutant", validate="one_to_one")
    splits = pd.read_csv(args.splits)[["mutant", args.split_strategy]].rename(
        columns={args.split_strategy: "split"}
    )
    data = data.merge(splits, on="mutant", validate="one_to_one")

    train = data["split"] == "train"
    validation = data["split"] == "validation"
    test = data["split"] == "test"
    columns = list(MUTATION_FEATURE_COLUMNS)

    selected, candidates = select_alpha(
        make_mutation_feature_model,
        MUTATION_ALPHAS,
        data.loc[train, columns],
        data.loc[train, "DMS_score"],
        data.loc[validation, columns],
        data.loc[validation, "DMS_score"],
    )

    final_model = make_mutation_feature_model(selected["alpha"])
    final_model.fit(
        data.loc[train | validation, columns], data.loc[train | validation, "DMS_score"]
    )
    test_prediction = final_model.predict(data.loc[test, columns])
    test_output = data.loc[test, ["mutant", "DMS_score"]].copy()
    test_output["predicted_score"] = test_prediction

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mutation_features_{args.split_strategy}"
    prediction_path = args.output_dir / f"{stem}.csv"
    metrics_path = args.output_dir / f"{stem}.json"
    test_output.to_csv(prediction_path, index=False)
    metrics = {
        "model": "ridge_mutation_features",
        "split_strategy": args.split_strategy,
        "selected_alpha": selected["alpha"],
        "selection_validation_spearman": selected["validation_spearman"],
        "test_spearman": spearman(test_output["DMS_score"], test_output["predicted_score"]),
        "n_train": int(train.sum()),
        "n_validation": int(validation.sum()),
        "n_test": int(test.sum()),
        "alpha_candidates": candidates,
        "predictions": str(prediction_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
