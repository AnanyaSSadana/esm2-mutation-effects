"""Masked-position inputs and residual mutation-effect scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import AMINO_ACIDS, parse_mutation

AMINO_ACID_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class MaskedPositionInputs:
    tokens: torch.Tensor
    token_positions: torch.Tensor
    residue_positions: tuple[int, ...]


@dataclass(frozen=True)
class EncodedMutations:
    site_indices: torch.Tensor
    wild_type_indices: torch.Tensor
    alternate_indices: torch.Tensor


def build_masked_position_inputs(
    target_sequence: str,
    residue_positions: list[int] | tuple[int, ...],
    alphabet,
) -> MaskedPositionInputs:
    """Build one masked wild-type sequence for every requested one-based position."""

    positions = tuple(int(position) for position in residue_positions)
    if not positions:
        raise ValueError("At least one residue position is required")
    if len(set(positions)) != len(positions):
        raise ValueError("Residue positions must be unique")
    if any(position < 1 or position > len(target_sequence) for position in positions):
        raise ValueError("Residue position is outside the target sequence")

    batch_converter = alphabet.get_batch_converter()
    records = [(f"position_{position}", target_sequence) for position in positions]
    _, _, tokens = batch_converter(records)
    for row_index, position in enumerate(positions):
        # fair-esm prepends BOS at token zero, so a one-based residue position
        # is also the corresponding token index.
        tokens[row_index, position] = alphabet.mask_idx

    return MaskedPositionInputs(
        tokens=tokens,
        token_positions=torch.tensor(positions, dtype=torch.long),
        residue_positions=positions,
    )


def encode_mutations(
    mutants: list[str] | tuple[str, ...],
    residue_positions: list[int] | tuple[int, ...],
) -> EncodedMutations:
    """Map mutation strings to masked-sequence rows and canonical amino-acid indices."""

    site_lookup = {int(position): index for index, position in enumerate(residue_positions)}
    if len(site_lookup) != len(residue_positions):
        raise ValueError("Residue positions must be unique")

    sites: list[int] = []
    wild_types: list[int] = []
    alternates: list[int] = []
    for value in mutants:
        mutation = parse_mutation(value)
        if mutation.position not in site_lookup:
            raise ValueError(f"Mutation {value} has no corresponding masked sequence")
        sites.append(site_lookup[mutation.position])
        wild_types.append(AMINO_ACID_TO_INDEX[mutation.wild_type])
        alternates.append(AMINO_ACID_TO_INDEX[mutation.alternate])

    return EncodedMutations(
        site_indices=torch.tensor(sites, dtype=torch.long),
        wild_type_indices=torch.tensor(wild_types, dtype=torch.long),
        alternate_indices=torch.tensor(alternates, dtype=torch.long),
    )


class MaskedMarginalResidualModel(nn.Module):
    """Add learned masked-site residuals to fixed zero-shot mutation scores."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 320,
        representation_layer: int = 6,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.backbone = backbone
        self.representation_layer = int(representation_layer)
        self.residual_head = nn.Linear(hidden_dim, len(AMINO_ACIDS))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def site_hidden_representations(
        self,
        tokens: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [masked positions, sequence tokens]")
        if token_positions.ndim != 1 or len(token_positions) != len(tokens):
            raise ValueError("token_positions must contain one position per token row")
        if token_positions.device != tokens.device:
            token_positions = token_positions.to(tokens.device)

        output = self.backbone(
            tokens,
            repr_layers=[self.representation_layer],
            return_contacts=False,
        )
        representations = output["representations"][self.representation_layer]
        rows = torch.arange(len(tokens), device=tokens.device)
        return representations[rows, token_positions]

    def site_residuals(
        self,
        tokens: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.site_hidden_representations(tokens, token_positions)
        return self.residual_head(hidden)

    def scores_from_site_residuals(
        self,
        residuals: torch.Tensor,
        site_indices: torch.Tensor,
        wild_type_indices: torch.Tensor,
        alternate_indices: torch.Tensor,
        fixed_zero_shot: torch.Tensor,
    ) -> torch.Tensor:
        n_variants = len(site_indices)
        one_dimensional = (wild_type_indices, alternate_indices, fixed_zero_shot)
        if any(value.ndim != 1 or len(value) != n_variants for value in one_dimensional):
            raise ValueError("Variant indices and fixed scores must be aligned one-dimensional data")
        if site_indices.ndim != 1:
            raise ValueError("site_indices must be one-dimensional")

        device = residuals.device
        sites = site_indices.to(device)
        wild_types = wild_type_indices.to(device)
        alternates = alternate_indices.to(device)
        base_scores = fixed_zero_shot.to(device=device, dtype=residuals.dtype)

        if n_variants and (sites.min() < 0 or sites.max() >= len(residuals)):
            raise ValueError("site_indices refer to a missing masked sequence")
        if n_variants and (
            wild_types.min() < 0
            or alternates.min() < 0
            or wild_types.max() >= len(AMINO_ACIDS)
            or alternates.max() >= len(AMINO_ACIDS)
        ):
            raise ValueError("Amino-acid indices are out of range")

        return base_scores + residuals[sites, alternates] - residuals[sites, wild_types]

    def forward(
        self,
        tokens: torch.Tensor,
        token_positions: torch.Tensor,
        site_indices: torch.Tensor,
        wild_type_indices: torch.Tensor,
        alternate_indices: torch.Tensor,
        fixed_zero_shot: torch.Tensor,
    ) -> torch.Tensor:
        residuals = self.site_residuals(tokens, token_positions)
        return self.scores_from_site_residuals(
            residuals,
            site_indices,
            wild_type_indices,
            alternate_indices,
            fixed_zero_shot,
        )
