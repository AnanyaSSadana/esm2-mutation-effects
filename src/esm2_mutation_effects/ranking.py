"""Regression and listwise-ranking objectives for mutation-effect adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    standard_deviation: float

    @classmethod
    def fit(cls, training_targets: torch.Tensor) -> TargetStandardizer:
        if training_targets.ndim != 1 or not len(training_targets):
            raise ValueError("Training targets must be a non-empty one-dimensional tensor")
        if not torch.isfinite(training_targets).all():
            raise ValueError("Training targets must be finite")
        mean = training_targets.mean().item()
        standard_deviation = training_targets.std(unbiased=False).item()
        if standard_deviation <= 0.0:
            raise ValueError("Training targets must have nonzero variance")
        return cls(mean=mean, standard_deviation=standard_deviation)

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean) / self.standard_deviation

    def inverse_transform(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.standard_deviation + self.mean


def mse_loss(predictions: torch.Tensor, standardized_targets: torch.Tensor) -> torch.Tensor:
    if predictions.shape != standardized_targets.shape:
        raise ValueError("Predictions and targets must have the same shape")
    return F.mse_loss(predictions, standardized_targets)


def listmle_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean negative ListMLE log-likelihood for descending target rankings."""

    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("ListMLE inputs must have shape [lists, list size]")
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape")
    if predictions.shape[0] < 1 or predictions.shape[1] < 2:
        raise ValueError("ListMLE requires at least one list containing two items")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("ListMLE inputs must be finite")

    sorted_targets, order = torch.sort(targets, dim=1, descending=True)
    if torch.any(sorted_targets[:, 1:] == sorted_targets[:, :-1]):
        raise ValueError("ListMLE target lists contain ties; define a tie policy before training")

    ordered_predictions = torch.gather(predictions, dim=1, index=order)
    suffix_log_sums = torch.logcumsumexp(ordered_predictions.flip(dims=(1,)), dim=1).flip(
        dims=(1,)
    )
    negative_log_likelihood = suffix_log_sums - ordered_predictions
    return negative_log_likelihood.sum(dim=1).mean()
