#!/usr/bin/env python3
"""Evaluate any mutant-level prediction file with Spearman's rho."""

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.evaluation import (
    SPLIT_STRATEGIES,
    SUBSETS,
    select_evaluation_subset,
)
from esm2_mutation_effects.metrics import spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--score-column", default="predicted_score")
    parser.add_argument("--split-strategy", choices=SPLIT_STRATEGIES)
    parser.add_argument("--subset", choices=SUBSETS)
    parser.add_argument(
        "--splits", type=Path, default=project_root() / "data" / "splits" / "splits.csv"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    filtering_requested = args.split_strategy is not None or args.subset is not None
    split_manifest = pd.read_csv(args.splits) if filtering_requested else None
    selected = select_evaluation_subset(
        predictions,
        args.score_column,
        split_manifest=split_manifest,
        split_strategy=args.split_strategy,
        subset=args.subset,
    )

    truth = load_assay()[["mutant", "DMS_score"]]
    merged = selected.merge(truth, on="mutant", how="left", validate="one_to_one")
    if merged["DMS_score"].isna().any():
        unknown = merged.loc[merged["DMS_score"].isna(), "mutant"].tolist()[:5]
        raise ValueError(f"Predictions contain unknown mutants, for example: {unknown}")

    result = {
        "metric": "spearman_rho",
        "n": len(merged),
        "score_column": args.score_column,
        "spearman_rho": spearman(
            merged["DMS_score"].to_numpy(), merged[args.score_column].to_numpy()
        ),
    }
    if filtering_requested:
        result["split_strategy"] = args.split_strategy
        result["subset"] = args.subset
    else:
        result["subset"] = "all_predictions"
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
