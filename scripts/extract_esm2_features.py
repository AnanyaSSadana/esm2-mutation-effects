#!/usr/bin/env python3
"""Cache frozen ESM-2 mutation-site representations for supervised learning."""

import argparse
from pathlib import Path

import numpy as np

from esm2_mutation_effects.data import load_assay, load_assay_metadata, mutation_table, project_root
from esm2_mutation_effects.esm_utils import REPRESENTATION_LAYER, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "processed" / "esm2_t6_site_features.npz",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import torch

    assay = load_assay()
    mutations = mutation_table(assay)
    target = str(load_assay_metadata()["target_seq"])
    if len(target) > 1022:
        raise ValueError("Sequence is longer than ESM-2's residue limit")

    device = choose_device(args.device)
    model, alphabet = load_model(device)
    batch_converter = alphabet.get_batch_converter()

    _, _, wild_type_tokens = batch_converter([("wild_type", target)])
    with torch.inference_mode():
        wild_type_representations = (
            model(
                wild_type_tokens.to(device),
                repr_layers=[REPRESENTATION_LAYER],
                return_contacts=False,
            )["representations"][REPRESENTATION_LAYER][0]
            .cpu()
            .numpy()
        )

    feature_batches = []
    for start in range(0, len(assay), args.batch_size):
        stop = min(start + args.batch_size, len(assay))
        batch = assay.iloc[start:stop]
        records = list(zip(batch["mutant"], batch["mutated_sequence"]))
        _, _, tokens = batch_converter(records)
        with torch.inference_mode():
            mutant_representations = (
                model(
                    tokens.to(device),
                    repr_layers=[REPRESENTATION_LAYER],
                    return_contacts=False,
                )["representations"][REPRESENTATION_LAYER]
                .cpu()
                .numpy()
            )

        rows = []
        for batch_index, global_index in enumerate(range(start, stop)):
            # Position is also the representation token index because token 0 is BOS.
            position = int(mutations.iloc[global_index]["position"])
            wt = wild_type_representations[position]
            mutant = mutant_representations[batch_index, position]
            rows.append(np.concatenate([wt, mutant, mutant - wt]))
        feature_batches.append(np.stack(rows))
        print(f"Embedded {stop}/{len(assay)} variants", flush=True)

    features = np.concatenate(feature_batches).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mutant=assay["mutant"].to_numpy(dtype=str),
        features=features,
        representation_layer=np.array(REPRESENTATION_LAYER),
        feature_definition=np.array("concat(wild_type_site,mutant_site,mutant_minus_wild_type)"),
    )
    print(f"Wrote {features.shape} features to {args.output} using {device}")


if __name__ == "__main__":
    main()
