"""Tie-safe fixed-update ListMLE, pooled, and Reptile utilities for Phase 3A."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .adaptation import (
    AMINO_ACID_TO_INDEX,
    MaskedMarginalResidualModel,
    build_masked_position_inputs,
)
from .data import mutation_table
from .lora import lora_parameters
from .ranking import listmle_loss
from .training import _predict_rows


@dataclass
class PilotMutationData:
    frame: pd.DataFrame
    masked_tokens: torch.Tensor
    token_positions: torch.Tensor
    site_indices: torch.Tensor
    wild_type_indices: torch.Tensor
    alternate_indices: torch.Tensor
    fixed_zero_shot: torch.Tensor
    score_bins: tuple[np.ndarray, ...]


def prepare_pilot_data(
    assay: pd.DataFrame, zero_shot: pd.DataFrame, target_sequence: str, alphabet
) -> PilotMutationData:
    """Tensorize one assay while retaining score bins for tie-safe ListMLE."""
    merged = assay[["mutant", "DMS_score"]].merge(
        zero_shot[["mutant", "predicted_score"]], on="mutant", how="left", validate="one_to_one"
    )
    if len(merged) != len(assay) or merged["predicted_score"].isna().any():
        raise ValueError("Zero-shot scores do not exactly cover the assay")
    mutations = mutation_table(merged)
    merged["position"] = mutations["position"].to_numpy()
    positions = tuple(sorted(merged["position"].unique().tolist()))
    masked = build_masked_position_inputs(target_sequence, positions, alphabet)
    lookup = {position: index for index, position in enumerate(positions)}
    scores = merged["DMS_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("DMS scores must be finite")
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    bins = tuple(np.flatnonzero(inverse == index) for index in range(len(unique_scores)))
    if len(bins) < 5:
        raise ValueError("At least five distinct score values are required for ListMLE")
    return PilotMutationData(
        frame=merged.reset_index(drop=True),
        masked_tokens=masked.tokens,
        token_positions=masked.token_positions,
        site_indices=torch.tensor([lookup[p] for p in merged.position], dtype=torch.long),
        wild_type_indices=torch.tensor(
            [AMINO_ACID_TO_INDEX[value] for value in mutations.wild_type], dtype=torch.long
        ),
        alternate_indices=torch.tensor(
            [AMINO_ACID_TO_INDEX[value] for value in mutations.alternate], dtype=torch.long
        ),
        fixed_zero_shot=torch.tensor(merged.predicted_score.to_numpy(dtype=np.float32)),
        score_bins=bins,
    )


def sample_tie_free_lists(
    data: PilotMutationData,
    rng: np.random.Generator,
    *,
    list_size: int = 5,
    lists_per_update: int = 16,
    allowed_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Sample one mutation from each of ``list_size`` distinct score bins."""
    if list_size < 2 or lists_per_update < 1:
        raise ValueError("List size and list count must be positive")
    allowed = np.ones(len(data.frame), dtype=bool)
    if allowed_rows is not None:
        allowed[:] = False
        allowed[np.asarray(allowed_rows, dtype=np.int64)] = True
    usable = [values[allowed[values]] for values in data.score_bins]
    usable = [values for values in usable if len(values)]
    if len(usable) < list_size:
        raise ValueError("Visible rows contain too few distinct score values")
    lists = np.empty((lists_per_update, list_size), dtype=np.int64)
    for index in range(lists_per_update):
        selected = rng.choice(len(usable), size=list_size, replace=False)
        lists[index] = [rng.choice(usable[value]) for value in selected]
    return lists


def _optimizer(model: MaskedMarginalResidualModel) -> torch.optim.Optimizer:
    head = list(model.residual_head.parameters())
    adapters = list(lora_parameters(model.backbone))
    if not adapters:
        raise ValueError("Phase 3A requires LoRA parameters")
    return torch.optim.AdamW(
        [{"params": head, "lr": 1e-3}, {"params": adapters, "lr": 1e-4}], weight_decay=0.01
    )


def listmle_update(
    model: MaskedMarginalResidualModel,
    optimizer: torch.optim.Optimizer,
    data: PilotMutationData,
    lists: np.ndarray,
    device: torch.device,
    *,
    microbatch_size: int = 16,
) -> float:
    """Perform exactly one gradient update from one assay's ranking lists."""
    if lists.ndim != 2 or len(lists) < 1 or lists.shape[1] < 2:
        raise ValueError("lists must have shape [positive list count, list size >= 2]")
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(lists), microbatch_size):
        micro = lists[start : start + microbatch_size]
        rows = micro.reshape(-1)
        predictions = _predict_rows(model, data, rows, device).reshape(micro.shape)
        targets = torch.tensor(
            data.frame.iloc[rows].DMS_score.to_numpy(dtype=np.float64), device=device
        ).reshape(micro.shape)
        loss = listmle_loss(predictions, targets)
        (loss * (len(micro) / len(lists))).backward()
        total += float(loss.detach()) * len(micro)
    nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    optimizer.step()
    return total / len(lists)


def train_pooled(
    model: MaskedMarginalResidualModel,
    tasks: dict[str, PilotMutationData],
    task_update_counts: dict[str, int],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    """Ordinary pooled control with exact per-task update exposure."""
    rng = np.random.default_rng(seed)
    schedule = [task for task, count in task_update_counts.items() for _ in range(count)]
    rng.shuffle(schedule)
    optimizer = _optimizer(model)
    history = []
    for step, task in enumerate(schedule, start=1):
        loss = listmle_update(model, optimizer, tasks[task], sample_tie_free_lists(tasks[task], rng), device)
        history.append({"step": step, "task": task, "loss": loss})
    return history


def train_reptile(
    model: MaskedMarginalResidualModel,
    tasks: dict[str, PilotMutationData],
    episode_task_counts: dict[str, int],
    device: torch.device,
    seed: int,
    outer_step_size: float = 0.1,
) -> list[dict[str, object]]:
    """Run exact two-inner-step Reptile episodes without a query-loss gradient."""
    if outer_step_size <= 0:
        raise ValueError("outer_step_size must be positive")
    rng = np.random.default_rng(seed)
    schedule = [task for task, count in episode_task_counts.items() for _ in range(count)]
    rng.shuffle(schedule)
    history = []
    for episode, task in enumerate(schedule, start=1):
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        optimizer = _optimizer(model)
        losses = []
        for _ in range(2):
            lists = sample_tie_free_lists(tasks[task], rng)
            losses.append(listmle_update(model, optimizer, tasks[task], lists, device))
        with torch.no_grad():
            delta_squared = 0.0
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    delta = parameter - before[name]
                    delta_squared += float(delta.float().square().sum())
                    parameter.copy_(before[name] + outer_step_size * delta)
        history.append({"episode": episode, "task": task, "inner_losses": losses, "delta_norm": delta_squared**0.5})
    return history


def adapt_fixed_steps(
    model: MaskedMarginalResidualModel,
    target: PilotMutationData,
    visible_rows: Iterable[int],
    device: torch.device,
    *,
    steps: int = 50,
    seed: int = 1,
) -> list[float]:
    """Adapt using exactly the provided target labels and no validation signal."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rows = np.asarray(list(visible_rows), dtype=np.int64)
    optimizer = _optimizer(model)
    rng = np.random.default_rng(seed)
    return [
        listmle_update(
            model, optimizer, target, sample_tie_free_lists(target, rng, allowed_rows=rows), device
        )
        for _ in range(steps)
    ]
