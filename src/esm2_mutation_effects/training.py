"""Training and artifact management for masked ESM-2 adaptation."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Sequence

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
from .diagnostics import position_ranking_metrics
from .lora import count_trainable_parameters, lora_parameters
from .ranking import TargetStandardizer, listmle_loss, mse_loss

BACKBONE_MODES = ("frozen", "lora")
LOSS_NAMES = ("mse", "listmle")
RUNTIME_CONFIG_FIELDS = frozenset({"device_type"})
OPTIONAL_CONFIG_DEFAULTS = {
    "n_labeled_train": None,
    "training_subset_sha256": None,
    "ranking_target_representation": "ordinal_rank_from_float64",
}


@dataclass(frozen=True)
class TrainingConfig:
    backbone_mode: str
    loss_name: str
    split_seed: int
    training_seed: int
    device_type: str
    model_name: str = "esm2_t6_8M_UR50D"
    representation_layer: int = 6
    hidden_dim: int = 320
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    head_learning_rate: float = 1e-3
    lora_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 15
    maximum_gradient_norm: float = 3.0
    mse_positions_per_batch: int = 16
    list_size: int = 5
    lists_per_batch: int = 16
    list_microbatch_size: int = 4
    evaluation_positions_per_batch: int = 16
    n_labeled_train: int | None = None
    training_subset_sha256: str | None = None
    ranking_target_representation: str = "ordinal_rank_from_float64"

    def validate(self) -> None:
        if self.backbone_mode not in BACKBONE_MODES:
            raise ValueError(f"Unknown backbone mode: {self.backbone_mode!r}")
        if self.loss_name not in LOSS_NAMES:
            raise ValueError(f"Unknown loss: {self.loss_name!r}")
        positive_integers = {
            "hidden_dim": self.hidden_dim,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "mse_positions_per_batch": self.mse_positions_per_batch,
            "list_size": self.list_size,
            "lists_per_batch": self.lists_per_batch,
            "list_microbatch_size": self.list_microbatch_size,
            "evaluation_positions_per_batch": self.evaluation_positions_per_batch,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.list_size < 2:
            raise ValueError("list_size must be at least 2")
        if self.list_microbatch_size > self.lists_per_batch:
            raise ValueError("list_microbatch_size cannot exceed lists_per_batch")
        positive_values = {
            "head_learning_rate": self.head_learning_rate,
            "lora_learning_rate": self.lora_learning_rate,
            "maximum_gradient_norm": self.maximum_gradient_norm,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.n_labeled_train is not None and self.n_labeled_train < 1:
            raise ValueError("n_labeled_train must be positive when provided")
        if self.training_subset_sha256 is not None:
            if len(self.training_subset_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.training_subset_sha256
            ):
                raise ValueError("training_subset_sha256 must be a lowercase SHA-256 digest")
        if self.ranking_target_representation != "ordinal_rank_from_float64":
            raise ValueError("Unsupported ranking target representation")

    def serializable(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def _experiment_config(config: dict[str, object]) -> dict[str, object]:
    """Return only settings that define the statistical training experiment.

    The compute device is recorded for provenance but does not change the
    split, model, optimization, or checkpoint semantics.  Excluding it here
    lets an interrupted CPU run resume on an available MPS/CUDA device.
    """
    normalized = dict(config)
    for key, value in OPTIONAL_CONFIG_DEFAULTS.items():
        normalized.setdefault(key, value)
    return {key: value for key, value in normalized.items() if key not in RUNTIME_CONFIG_FIELDS}


def training_subset_fingerprint(mutants: Sequence[str]) -> str:
    """Return a canonical fingerprint for a unique set of mutation identifiers."""

    values = [str(mutant) for mutant in mutants]
    if not values:
        raise ValueError("Training subset must contain at least one mutant")
    if len(set(values)) != len(values):
        raise ValueError("Training subset contains duplicate mutants")
    return sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def ordinal_ranking_targets(values: np.ndarray) -> torch.Tensor:
    """Return exact float32 ordinal ranks from unique float64 source targets.

    ListMLE uses targets only to define a ranking. Raw assay scores are retained as float64 for
    sorting because float32 rounding can merge distinct close measurements on MPS. The ordinal
    ranks are exact float32 integers, preserve that source ordering, and introduce no tie policy.
    """

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("Ranking targets must be a non-empty one-dimensional sequence")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking targets must be finite")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    if np.any(ordered_scores[1:] == ordered_scores[:-1]):
        raise ValueError("ListMLE source targets contain exact ties")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return torch.tensor(ranks, dtype=torch.float32)


@dataclass
class PreparedMutationData:
    frame: pd.DataFrame
    masked_tokens: torch.Tensor
    token_positions: torch.Tensor
    site_indices: torch.Tensor
    wild_type_indices: torch.Tensor
    alternate_indices: torch.Tensor
    fixed_zero_shot: torch.Tensor
    targets: torch.Tensor
    ranking_targets: torch.Tensor
    target_standardizer: TargetStandardizer
    training_rows: np.ndarray
    training_subset_sha256: str | None

    def rows(self, subset: str) -> np.ndarray:
        rows = np.flatnonzero(self.frame["split"].to_numpy() == subset)
        if not len(rows):
            raise ValueError(f"No rows belong to subset {subset!r}")
        return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_mutation_data(
    assay: pd.DataFrame,
    zero_shot: pd.DataFrame,
    split_manifest: pd.DataFrame,
    target_sequence: str,
    alphabet,
    training_subset_mutants: Sequence[str] | None = None,
) -> PreparedMutationData:
    """Validate and tensorize one assay without using validation or test labels for scaling."""

    assay_required = {"mutant", "DMS_score"}
    zero_required = {"mutant", "predicted_score"}
    split_required = {"mutant", "split"}
    for name, frame, required in (
        ("assay", assay, assay_required),
        ("zero-shot", zero_shot, zero_required),
        ("split", split_manifest, split_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if frame["mutant"].duplicated().any():
            raise ValueError(f"{name} data contains duplicate mutants")

    data = assay[["mutant", "DMS_score"]].merge(
        zero_shot[["mutant", "predicted_score"]],
        on="mutant",
        how="left",
        validate="one_to_one",
    )
    data = data.merge(
        split_manifest[["mutant", "split"]],
        on="mutant",
        how="left",
        validate="one_to_one",
    )
    if data[["predicted_score", "split"]].isna().any().any():
        raise ValueError("Zero-shot predictions or split assignments are incomplete")
    if len(data) != len(zero_shot) or len(data) != len(split_manifest):
        raise ValueError("Assay, zero-shot predictions, and splits must contain identical mutants")
    if set(data["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split assignments must contain train, validation, and test")
    if not np.isfinite(data["DMS_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("DMS targets must be finite")
    if not np.isfinite(data["predicted_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Zero-shot predictions must be finite")

    mutations = mutation_table(data)
    data["position"] = mutations["position"].to_numpy()
    if (data.groupby("position")["split"].nunique() != 1).any():
        raise ValueError("Position split leaks a residue across subsets")

    positions = tuple(sorted(data["position"].unique().tolist()))
    masked = build_masked_position_inputs(target_sequence, positions, alphabet)
    position_to_site = {position: index for index, position in enumerate(positions)}
    site_indices = torch.tensor(
        [position_to_site[position] for position in data["position"]], dtype=torch.long
    )
    wild_type_indices = torch.tensor(
        [AMINO_ACID_TO_INDEX[value] for value in mutations["wild_type"]], dtype=torch.long
    )
    alternate_indices = torch.tensor(
        [AMINO_ACID_TO_INDEX[value] for value in mutations["alternate"]], dtype=torch.long
    )
    fixed_zero_shot = torch.tensor(
        data["predicted_score"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    raw_targets = data["DMS_score"].to_numpy(dtype=np.float64)
    targets = torch.tensor(raw_targets.astype(np.float32), dtype=torch.float32)
    ranking_targets = ordinal_ranking_targets(raw_targets)
    available_training_rows = np.flatnonzero(data["split"].to_numpy() == "train")
    training_subset_sha256 = None
    if training_subset_mutants is None:
        training_rows = available_training_rows
    else:
        subset = [str(mutant) for mutant in training_subset_mutants]
        training_subset_sha256 = training_subset_fingerprint(subset)
        available_mutants = set(data.iloc[available_training_rows]["mutant"])
        requested_mutants = set(subset)
        invalid = requested_mutants.difference(available_mutants)
        if invalid:
            example = sorted(invalid)[0]
            raise ValueError(
                "Training subset contains a mutant outside the training positions: "
                f"{example}"
            )
        training_rows = np.flatnonzero(data["mutant"].isin(requested_mutants).to_numpy())
    standardizer = TargetStandardizer.fit(targets[training_rows])

    return PreparedMutationData(
        frame=data.reset_index(drop=True),
        masked_tokens=masked.tokens,
        token_positions=masked.token_positions,
        site_indices=site_indices,
        wild_type_indices=wild_type_indices,
        alternate_indices=alternate_indices,
        fixed_zero_shot=fixed_zero_shot,
        targets=targets,
        ranking_targets=ranking_targets,
        target_standardizer=standardizer,
        training_rows=training_rows,
        training_subset_sha256=training_subset_sha256,
    )


def _predict_rows(
    model: MaskedMarginalResidualModel,
    data: PreparedMutationData,
    rows: np.ndarray | torch.Tensor,
    device: torch.device,
    frozen_site_representations: torch.Tensor | None = None,
) -> torch.Tensor:
    row_tensor = torch.as_tensor(rows, dtype=torch.long)
    global_sites = data.site_indices[row_tensor]
    unique_sites, local_sites = torch.unique(global_sites, sorted=True, return_inverse=True)
    if frozen_site_representations is None:
        return model(
            tokens=data.masked_tokens[unique_sites].to(device),
            token_positions=data.token_positions[unique_sites].to(device),
            site_indices=local_sites.to(device),
            wild_type_indices=data.wild_type_indices[row_tensor].to(device),
            alternate_indices=data.alternate_indices[row_tensor].to(device),
            fixed_zero_shot=data.fixed_zero_shot[row_tensor].to(device),
        )

    residuals = model.residual_head(frozen_site_representations[unique_sites.to(device)])
    return model.scores_from_site_residuals(
        residuals=residuals,
        site_indices=local_sites.to(device),
        wild_type_indices=data.wild_type_indices[row_tensor].to(device),
        alternate_indices=data.alternate_indices[row_tensor].to(device),
        fixed_zero_shot=data.fixed_zero_shot[row_tensor].to(device),
    )


def cache_frozen_site_representations(
    model: MaskedMarginalResidualModel,
    data: PreparedMutationData,
    device: torch.device,
    positions_per_batch: int,
) -> torch.Tensor:
    """Compute each fixed masked-site representation once for a frozen backbone."""

    if positions_per_batch < 1:
        raise ValueError("positions_per_batch must be positive")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("Frozen representation caching requires a fully frozen backbone")

    model.backbone.eval()
    site_count = len(data.masked_tokens)
    cached = torch.empty(
        (site_count, model.residual_head.in_features),
        device=device,
        dtype=model.residual_head.weight.dtype,
    )
    with torch.no_grad():
        for start in range(0, site_count, positions_per_batch):
            stop = min(start + positions_per_batch, site_count)
            hidden = model.site_hidden_representations(
                data.masked_tokens[start:stop].to(device),
                data.token_positions[start:stop].to(device),
            )
            cached[start:stop] = hidden
    return cached


def predict_subset(
    model: MaskedMarginalResidualModel,
    data: PreparedMutationData,
    subset: str,
    device: torch.device,
    positions_per_batch: int,
    frozen_site_representations: torch.Tensor | None = None,
) -> pd.DataFrame:
    if positions_per_batch < 1:
        raise ValueError("positions_per_batch must be positive")
    subset_rows = data.rows(subset)
    subset_sites = sorted(set(data.site_indices[subset_rows].tolist()))
    predictions = np.empty(len(subset_rows), dtype=np.float64)
    output_lookup = {int(row): index for index, row in enumerate(subset_rows)}

    model.eval()
    # fair-esm's rotary embedding caches tensors from the first forward pass.
    # inference_mode would mark that cache as inference-only and make a later
    # training forward fail when autograd tries to save it for backward.
    with torch.no_grad():
        for start in range(0, len(subset_sites), positions_per_batch):
            selected_sites = set(subset_sites[start : start + positions_per_batch])
            batch_rows = np.asarray(
                [row for row in subset_rows if int(data.site_indices[row]) in selected_sites],
                dtype=np.int64,
            )
            batch_predictions = _predict_rows(
                model,
                data,
                batch_rows,
                device,
                frozen_site_representations,
            ).cpu().numpy()
            for row, prediction in zip(batch_rows, batch_predictions):
                predictions[output_lookup[int(row)]] = float(prediction)

    output = data.frame.iloc[subset_rows][["mutant", "DMS_score", "position", "split"]].copy()
    output["predicted_score"] = predictions
    return output.reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    return position_ranking_metrics(
        predictions["DMS_score"].to_numpy(dtype=np.float64),
        predictions["predicted_score"].to_numpy(dtype=np.float64),
        predictions["position"].to_numpy(),
    )


def _validation_loss(
    predictions: pd.DataFrame,
    data: PreparedMutationData,
    config: TrainingConfig,
) -> float:
    predicted = torch.tensor(predictions["predicted_score"].to_numpy(), dtype=torch.float32)
    targets = torch.tensor(predictions["DMS_score"].to_numpy(), dtype=torch.float32)
    if config.loss_name == "mse":
        standardized = data.target_standardizer.transform(targets)
        return float(mse_loss(predicted, standardized))

    generator = np.random.default_rng(0)
    order = generator.permutation(len(predictions))
    usable = len(order) - len(order) % config.list_size
    if usable < config.list_size:
        raise ValueError("Validation subset is too small for the configured ListMLE list size")
    order = order[:usable]
    ranking_targets = ordinal_ranking_targets(
        predictions["DMS_score"].to_numpy(dtype=np.float64)
    )
    return float(
        listmle_loss(
            predicted[order].reshape(-1, config.list_size),
            ranking_targets[order].reshape(-1, config.list_size),
        )
    )


def _optimizer(
    model: MaskedMarginalResidualModel,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    parameter_groups = [
        {
            "params": list(model.residual_head.parameters()),
            "lr": config.head_learning_rate,
            "name": "residual_head",
        }
    ]
    if config.backbone_mode == "lora":
        adapters = list(lora_parameters(model.backbone))
        if not adapters:
            raise ValueError("LoRA mode requested but no adapter parameters were found")
        parameter_groups.append(
            {"params": adapters, "lr": config.lora_learning_rate, "name": "lora"}
        )

    optimized = {id(parameter) for group in parameter_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimized != trainable:
        raise ValueError("Optimizer groups do not exactly match the trainable parameters")
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )


def _gradient_norm(parameters) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum())
    return float(np.sqrt(squared_norm))


def _mse_epoch(
    model: MaskedMarginalResidualModel,
    optimizer: torch.optim.Optimizer,
    data: PreparedMutationData,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
    frozen_site_representations: torch.Tensor | None = None,
) -> dict[str, float | None]:
    training_rows = data.training_rows
    training_sites = np.unique(data.site_indices[training_rows].numpy())
    rng = np.random.default_rng(config.training_seed + epoch)
    rng.shuffle(training_sites)
    total_loss = 0.0
    total_rows = 0
    maximum_gradient_norm = 0.0
    maximum_head_gradient_norm = 0.0
    maximum_lora_gradient_norm: float | None = None

    for start in range(0, len(training_sites), config.mse_positions_per_batch):
        selected_sites = set(training_sites[start : start + config.mse_positions_per_batch])
        rows = np.asarray(
            [row for row in training_rows if int(data.site_indices[row]) in selected_sites],
            dtype=np.int64,
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = _predict_rows(
            model,
            data,
            rows,
            device,
            frozen_site_representations,
        )
        targets = data.target_standardizer.transform(data.targets[rows]).to(device)
        loss = mse_loss(predictions, targets)
        loss.backward()
        maximum_head_gradient_norm = max(
            maximum_head_gradient_norm,
            _gradient_norm(model.residual_head.parameters()),
        )
        if config.backbone_mode == "lora":
            lora_gradient_norm = _gradient_norm(lora_parameters(model.backbone))
            maximum_lora_gradient_norm = max(maximum_lora_gradient_norm or 0.0, lora_gradient_norm)
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        total_loss += float(loss.detach()) * len(rows)
        total_rows += len(rows)
    return {
        "loss": total_loss / total_rows,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_head_gradient_norm": maximum_head_gradient_norm,
        "maximum_lora_gradient_norm": maximum_lora_gradient_norm,
    }


def _listmle_epoch(
    model: MaskedMarginalResidualModel,
    optimizer: torch.optim.Optimizer,
    data: PreparedMutationData,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
    frozen_site_representations: torch.Tensor | None = None,
) -> dict[str, float | None]:
    training_rows = data.training_rows
    rng = np.random.default_rng(config.training_seed + epoch)
    shuffled = rng.permutation(training_rows)
    usable = len(shuffled) - len(shuffled) % config.list_size
    lists = shuffled[:usable].reshape(-1, config.list_size)
    total_loss = 0.0
    total_lists = 0
    maximum_gradient_norm = 0.0
    maximum_head_gradient_norm = 0.0
    maximum_lora_gradient_norm: float | None = None

    for batch_start in range(0, len(lists), config.lists_per_batch):
        list_batch = lists[batch_start : batch_start + config.lists_per_batch]
        optimizer.zero_grad(set_to_none=True)
        for micro_start in range(0, len(list_batch), config.list_microbatch_size):
            microbatch = list_batch[micro_start : micro_start + config.list_microbatch_size]
            flat_rows = microbatch.reshape(-1)
            predictions = _predict_rows(
                model,
                data,
                flat_rows,
                device,
                frozen_site_representations,
            ).reshape(-1, config.list_size)
            targets = data.ranking_targets[flat_rows].to(device).reshape(-1, config.list_size)
            loss = listmle_loss(predictions, targets)
            micro_lists = len(microbatch)
            (loss * (micro_lists / len(list_batch))).backward()
            total_loss += float(loss.detach()) * micro_lists
            total_lists += micro_lists
        maximum_head_gradient_norm = max(
            maximum_head_gradient_norm,
            _gradient_norm(model.residual_head.parameters()),
        )
        if config.backbone_mode == "lora":
            lora_gradient_norm = _gradient_norm(lora_parameters(model.backbone))
            maximum_lora_gradient_norm = max(maximum_lora_gradient_norm or 0.0, lora_gradient_norm)
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
    return {
        "loss": total_loss / total_lists,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_head_gradient_norm": maximum_head_gradient_norm,
        "maximum_lora_gradient_norm": maximum_lora_gradient_norm,
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_checkpoint(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _is_better(
    metrics: dict[str, float | int | None],
    validation_loss: float,
    best_spearman: float | None,
    best_loss: float | None,
) -> bool:
    correlation = metrics["overall_spearman"]
    if correlation is None:
        return False
    correlation = float(correlation)
    if best_spearman is None or correlation > best_spearman + 1e-12:
        return True
    return (
        abs(correlation - best_spearman) <= 1e-12
        and (best_loss is None or validation_loss < best_loss - 1e-12)
    )


def _save_predictions(
    output_dir: Path,
    subset: str,
    predictions: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    predictions.to_csv(output_dir / f"{subset}_predictions.csv", index=False)
    _atomic_json(output_dir / f"{subset}_metrics.json", metrics)


def _allocated_memory_mib(device: torch.device) -> float | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    if device.type == "mps" and hasattr(torch, "mps"):
        return float(torch.mps.current_allocated_memory() / (1024**2))
    return None


def _show_progress(record: dict[str, object]) -> None:
    metrics = record["validation_metrics"]
    message = {
        "epoch": record["epoch"],
        "train_loss": record["train_loss"],
        "validation_loss": record["validation_loss"],
        "validation_spearman": metrics["overall_spearman"],
        "improved": record["improved"],
        "epoch_seconds": record["epoch_seconds"],
        "allocated_memory_mib": record["allocated_memory_mib"],
        "maximum_gradient_norm": record["maximum_gradient_norm"],
        "maximum_head_gradient_norm": record["maximum_head_gradient_norm"],
        "maximum_lora_gradient_norm": record["maximum_lora_gradient_norm"],
    }
    print(json.dumps(message, sort_keys=True), flush=True)


def run_training(
    model: MaskedMarginalResidualModel,
    data: PreparedMutationData,
    config: TrainingConfig,
    output_dir: Path,
    device: torch.device,
    *,
    resume: bool = False,
    evaluate_test: bool = False,
    show_progress: bool = False,
) -> dict[str, object]:
    """Train with validation-only checkpoint selection and persist resume-safe artifacts."""

    config.validate()
    if config.n_labeled_train is not None and config.n_labeled_train != len(data.training_rows):
        raise ValueError("Configured label count does not match the prepared training subset")
    if data.training_subset_sha256 is not None:
        if config.training_subset_sha256 != data.training_subset_sha256:
            raise ValueError("Configured training subset fingerprint does not match prepared data")
    elif config.training_subset_sha256 is not None:
        raise ValueError("Configured training subset fingerprint has no prepared subset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    current_config = config.serializable()

    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if _experiment_config(existing_config) != _experiment_config(current_config):
            raise ValueError("Existing run configuration does not match the requested configuration")
        if not resume:
            raise FileExistsError(f"Run already exists at {output_dir}; pass resume=True")
    elif resume:
        raise FileNotFoundError(f"Cannot resume because {config_path} does not exist")
    else:
        _atomic_json(config_path, current_config)

    model = model.to(device)
    optimizer = _optimizer(model, config)
    frozen_site_representations = None
    cache_seconds = None
    if config.backbone_mode == "frozen":
        cache_started = time.perf_counter()
        frozen_site_representations = cache_frozen_site_representations(
            model,
            data,
            device,
            config.evaluation_positions_per_batch,
        )
        cache_seconds = time.perf_counter() - cache_started
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_spearman: float | None = None
    best_loss: float | None = None
    epochs_without_improvement = 0
    stopped_early = False

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume because {last_path} does not exist")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_spearman = checkpoint["best_spearman"]
        best_loss = checkpoint["best_loss"]
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        stopped_early = bool(checkpoint.get("stopped_early", False))
    else:
        seed_everything(config.training_seed)
        initial_predictions = predict_subset(
            model,
            data,
            "validation",
            device,
            config.evaluation_positions_per_batch,
            frozen_site_representations,
        )
        initial_metrics = evaluate_predictions(initial_predictions)
        initial_loss = _validation_loss(initial_predictions, data, config)
        best_spearman = initial_metrics["overall_spearman"]
        best_loss = initial_loss
        initial_record = {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": initial_loss,
            "validation_metrics": initial_metrics,
            "improved": True,
            "epoch_seconds": 0.0,
            "allocated_memory_mib": _allocated_memory_mib(device),
            "maximum_gradient_norm": None,
            "maximum_head_gradient_norm": None,
            "maximum_lora_gradient_norm": None,
        }
        history.append(initial_record)
        if show_progress:
            _show_progress(initial_record)
        _atomic_checkpoint(
            best_path,
            {
                "epoch": 0,
                "model_state": model.state_dict(),
                "validation_loss": initial_loss,
                "validation_metrics": initial_metrics,
                "config": current_config,
            },
        )
        _atomic_checkpoint(
            last_path,
            {
                "epoch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_spearman": best_spearman,
                "best_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_early": False,
                "config": current_config,
            },
        )
        _atomic_json(history_path, history)

    started = time.perf_counter()
    if not stopped_early:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            seed_everything(config.training_seed + epoch)
            model.train()
            if config.backbone_mode == "frozen":
                model.backbone.eval()
            if config.loss_name == "mse":
                training = _mse_epoch(
                    model,
                    optimizer,
                    data,
                    config,
                    device,
                    epoch,
                    frozen_site_representations,
                )
            else:
                training = _listmle_epoch(
                    model,
                    optimizer,
                    data,
                    config,
                    device,
                    epoch,
                    frozen_site_representations,
                )
            train_loss = float(training["loss"])

            validation_predictions = predict_subset(
                model,
                data,
                "validation",
                device,
                config.evaluation_positions_per_batch,
                frozen_site_representations,
            )
            validation_metrics = evaluate_predictions(validation_predictions)
            validation_loss = _validation_loss(validation_predictions, data, config)
            improved = _is_better(
                validation_metrics,
                validation_loss,
                best_spearman,
                best_loss,
            )
            if improved:
                best_epoch = epoch
                best_spearman = float(validation_metrics["overall_spearman"])
                best_loss = validation_loss
                epochs_without_improvement = 0
                _atomic_checkpoint(
                    best_path,
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "validation_loss": validation_loss,
                        "validation_metrics": validation_metrics,
                        "config": current_config,
                    },
                )
            else:
                epochs_without_improvement += 1

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "improved": improved,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "allocated_memory_mib": _allocated_memory_mib(device),
                "maximum_gradient_norm": training["maximum_gradient_norm"],
                "maximum_head_gradient_norm": training["maximum_head_gradient_norm"],
                "maximum_lora_gradient_norm": training["maximum_lora_gradient_norm"],
            }
            history.append(epoch_record)
            if show_progress:
                _show_progress(epoch_record)
            stopped_early = epochs_without_improvement >= config.patience
            _atomic_checkpoint(
                last_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_spearman": best_spearman,
                    "best_loss": best_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                    "stopped_early": stopped_early,
                    "config": current_config,
                },
            )
            _atomic_json(history_path, history)
            if stopped_early:
                break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    validation_predictions = predict_subset(
        model,
        data,
        "validation",
        device,
        config.evaluation_positions_per_batch,
        frozen_site_representations,
    )
    validation_metrics = evaluate_predictions(validation_predictions)
    validation_metrics.update(
        {
            "subset": "validation",
            "selected_epoch": best_epoch,
            "validation_loss": best_loss,
        }
    )
    _save_predictions(output_dir, "validation", validation_predictions, validation_metrics)

    summary: dict[str, object] = {
        "status": "stopped_early" if stopped_early else "completed_max_epochs",
        "best_epoch": best_epoch,
        "best_validation_spearman": best_spearman,
        "best_validation_loss": best_loss,
        "epochs_completed": int(history[-1]["epoch"]),
        "trainable_parameters": count_trainable_parameters(model),
        "frozen_site_representation_cache": frozen_site_representations is not None,
        "frozen_site_cache_seconds": cache_seconds,
        "test_evaluated": bool(evaluate_test),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "output_dir": str(output_dir.resolve()),
        "config": current_config,
        "n_labeled_train": int(len(data.training_rows)),
        "training_subset_sha256": data.training_subset_sha256,
    }

    if evaluate_test:
        test_predictions = predict_subset(
            model,
            data,
            "test",
            device,
            config.evaluation_positions_per_batch,
            frozen_site_representations,
        )
        test_metrics = evaluate_predictions(test_predictions)
        test_metrics.update({"subset": "test", "selected_epoch": best_epoch})
        _save_predictions(output_dir, "test", test_predictions, test_metrics)
        summary["test_metrics"] = test_metrics

    _atomic_json(output_dir / "summary.json", summary)
    return summary
