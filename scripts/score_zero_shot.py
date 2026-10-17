#!/usr/bin/env python3
"""Score substitutions with ESM-2 masked-marginal log odds."""

import argparse
from pathlib import Path

from esm2_mutation_effects.data import load_assay, load_assay_metadata, mutation_table, project_root
from esm2_mutation_effects.esm_utils import choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results" / "esm2_masked_marginal.csv",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import torch

    assay = load_assay()
    mutations = mutation_table(assay)
    target = str(load_assay_metadata()["target_seq"])
    positions = sorted(mutations["position"].unique().tolist())
    if len(target) > 1022:
        raise ValueError("Sequence is longer than ESM-2's residue limit")

    device = choose_device(args.device)
    model, alphabet = load_model(device)
    batch_converter = alphabet.get_batch_converter()
    scores_by_mutant: dict[str, float] = {}

    for start in range(0, len(positions), args.batch_size):
        batch_positions = positions[start : start + args.batch_size]
        records = [(f"position_{position}", target) for position in batch_positions]
        _, _, tokens = batch_converter(records)
        for row_index, position in enumerate(batch_positions):
            # Residues are one-based in mutant strings; token 0 is BOS, so the
            # residue position is also its token index.
            tokens[row_index, position] = alphabet.mask_idx
        tokens = tokens.to(device)
        with torch.inference_mode():
            logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1).cpu()

        for row_index, position in enumerate(batch_positions):
            at_position = mutations.loc[mutations["position"] == position]
            for mutation in at_position.itertuples(index=False):
                alternate = alphabet.get_idx(mutation.alternate)
                wild_type = alphabet.get_idx(mutation.wild_type)
                score = (
                    log_probs[row_index, position, alternate]
                    - log_probs[row_index, position, wild_type]
                )
                scores_by_mutant[mutation.mutant] = float(score)

    output = assay[["mutant", "DMS_score"]].copy()
    output["predicted_score"] = output["mutant"].map(scores_by_mutant)
    if output["predicted_score"].isna().any():
        raise RuntimeError("Some variants were not scored")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} scores to {args.output} using {device}")


if __name__ == "__main__":
    main()
