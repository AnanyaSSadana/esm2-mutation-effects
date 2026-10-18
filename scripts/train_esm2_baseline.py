#!/usr/bin/env python3
"""Ridge regression on cached frozen ESM-2 site representations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.features import (
    CACHED_FEATURE_DEFINITION,
    MODEL_FEATURE_DEFINITION,
    select_nonredundant_site_features,
)
from esm2_mutation_effects.metrics import spearman

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)


def make_model(alpha: float):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10_000),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-strategy", choices=("random", "position"), default="position")
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root() / "data" / "processed" / "esm2_t6_site_features.npz",
    )
    parser.add_argument(
        "--splits", type=Path, default=project_root() / "data" / "splits" / "splits.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=project_root() / "results")
    args = parser.parse_args()

    assay = load_assay()[["mutant", "DMS_score"]]
    cache = np.load(args.features, allow_pickle=False)
    cached_definition = str(cache["feature_definition"].item())
    if cached_definition != CACHED_FEATURE_DEFINITION:
        raise ValueError(
            f"Unexpected cached feature definition {cached_definition!r}; "
            f"expected {CACHED_FEATURE_DEFINITION!r}"
        )
    cached_mutants = pd.DataFrame(
        {"mutant": cache["mutant"].astype(str), "row": range(len(cache["mutant"]))}
    )
    data = assay.merge(cached_mutants, on="mutant", validate="one_to_one")
    splits = pd.read_csv(args.splits)[["mutant", args.split_strategy]].rename(
        columns={args.split_strategy: "split"}
    )
    data = data.merge(splits, on="mutant", validate="one_to_one")
    x = select_nonredundant_site_features(cache["features"])[data["row"].to_numpy()]
    y = data["DMS_score"].to_numpy()
    train = (data["split"] == "train").to_numpy()
    validation = (data["split"] == "validation").to_numpy()
    test = (data["split"] == "test").to_numpy()

    candidates = []
    for alpha in ALPHAS:
        model = make_model(alpha)
        model.fit(x[train], y[train])
        prediction = model.predict(x[validation])
        candidates.append(
            {"alpha": alpha, "validation_spearman": spearman(y[validation], prediction)}
        )
    selected = max(candidates, key=lambda item: item["validation_spearman"])

    final_model = make_model(float(selected["alpha"]))
    final_model.fit(x[train | validation], y[train | validation])
    test_prediction = final_model.predict(x[test])
    test_output = data.loc[test, ["mutant", "DMS_score"]].copy()
    test_output["predicted_score"] = test_prediction

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frozen_esm2_{args.split_strategy}"
    prediction_path = args.output_dir / f"{stem}.csv"
    metrics_path = args.output_dir / f"{stem}.json"
    test_output.to_csv(prediction_path, index=False)
    metrics = {
        "model": "ridge_frozen_esm2_t6_wild_type_delta_site_representations",
        "split_strategy": args.split_strategy,
        "selected_alpha": selected["alpha"],
        "selection_validation_spearman": selected["validation_spearman"],
        "test_spearman": spearman(test_output["DMS_score"], test_output["predicted_score"]),
        "n_train": int(train.sum()),
        "n_validation": int(validation.sum()),
        "n_test": int(test.sum()),
        "alpha_candidates": candidates,
        "cached_feature_definition": cached_definition,
        "model_feature_definition": MODEL_FEATURE_DEFINITION,
        "n_model_features": int(x.shape[1]),
        "feature_dtype": str(x.dtype),
        "ridge_solver": "lsqr",
        "features": str(args.features),
        "predictions": str(prediction_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
