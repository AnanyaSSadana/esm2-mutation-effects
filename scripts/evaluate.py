#!/usr/bin/env python3
"""Evaluate any mutant-level prediction file with Spearman's rho."""

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import load_assay
from esm2_mutation_effects.metrics import spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--score-column", default="predicted_score")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required = {"mutant", args.score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions["mutant"].duplicated().any():
        raise ValueError("Prediction file contains duplicate mutants")

    truth = load_assay()[["mutant", "DMS_score"]]
    merged = predictions[["mutant", args.score_column]].merge(
        truth, on="mutant", how="left", validate="one_to_one"
    )
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
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
