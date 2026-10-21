"""Reusable masked-marginal scoring for Phase 3 pilot assays."""

from __future__ import annotations

import pandas as pd
import torch

from .data import mutation_table


def score_masked_marginals(
    assay: pd.DataFrame,
    target_sequence: str,
    model,
    alphabet,
    device: torch.device,
    *,
    positions_per_batch: int,
) -> pd.DataFrame:
    """Score every assay mutation with one masked forward pass per site."""

    if positions_per_batch < 1:
        raise ValueError("positions_per_batch must be positive")
    target_sequence = str(target_sequence).upper()
    if len(target_sequence) > 1022:
        raise ValueError("Sequence is longer than ESM-2's residue limit")
    if assay["mutant"].duplicated().any():
        raise ValueError("Assay contains duplicate mutant identifiers")
    mutations = mutation_table(assay)
    positions = sorted(mutations["position"].unique().tolist())
    batch_converter = alphabet.get_batch_converter()
    scores_by_mutant: dict[str, float] = {}

    model.eval()
    for start in range(0, len(positions), positions_per_batch):
        batch_positions = positions[start : start + positions_per_batch]
        records = [(f"position_{position}", target_sequence) for position in batch_positions]
        _, _, tokens = batch_converter(records)
        for row_index, position in enumerate(batch_positions):
            tokens[row_index, position] = alphabet.mask_idx
        with torch.inference_mode():
            logits = model(tokens.to(device), repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1).cpu()
        for row_index, position in enumerate(batch_positions):
            at_position = mutations.loc[mutations["position"] == position]
            for mutation in at_position.itertuples(index=False):
                alternate = alphabet.get_idx(mutation.alternate)
                wild_type = alphabet.get_idx(mutation.wild_type)
                scores_by_mutant[mutation.mutant] = float(
                    log_probs[row_index, position, alternate]
                    - log_probs[row_index, position, wild_type]
                )

    result = assay[["mutant", "DMS_score"]].copy()
    result["predicted_score"] = result["mutant"].map(scores_by_mutant)
    if result["predicted_score"].isna().any():
        raise RuntimeError("Some variants were not scored")
    return result
