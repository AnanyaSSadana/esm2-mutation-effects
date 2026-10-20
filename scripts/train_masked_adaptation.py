#!/usr/bin/env python3
"""Train one frozen/LoRA and MSE/ListMLE masked-residual configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.adaptation import MaskedMarginalResidualModel
from esm2_mutation_effects.data import load_assay, load_assay_metadata, project_root
from esm2_mutation_effects.esm_utils import (
    MODEL_NAME,
    REPRESENTATION_LAYER,
    choose_device,
    load_model,
)
from esm2_mutation_effects.lora import inject_lora
from esm2_mutation_effects.splits import make_splits
from esm2_mutation_effects.training import (
    TrainingConfig,
    prepare_mutation_data,
    run_training,
    seed_everything,
    training_subset_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one controlled masked-residual model. Test positions are not evaluated unless "
            "--evaluate-test is supplied explicitly."
        )
    )
    parser.add_argument("--backbone", choices=("frozen", "lora"), required=True)
    parser.add_argument("--loss", choices=("mse", "listmle"), required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-gradient-norm", type=float, default=3.0)
    parser.add_argument("--mse-positions-per-batch", type=int, default=16)
    parser.add_argument("--list-size", type=int, default=5)
    parser.add_argument("--lists-per-batch", type=int, default=16)
    parser.add_argument("--list-microbatch-size", type=int, default=4)
    parser.add_argument("--evaluation-positions-per-batch", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--frozen-site-cache",
        type=Path,
        default=None,
        help=(
            "Persistent cache for frozen masked-site representations. By default, frozen runs "
            "use a device-qualified cache under data/processed/; LoRA runs do not use this cache."
        ),
    )
    parser.add_argument(
        "--no-persistent-frozen-site-cache",
        action="store_true",
        help="Recompute frozen masked-site representations for this invocation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly evaluate held-out test positions after selecting the validation checkpoint",
    )
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=project_root() / "results" / "esm2_masked_marginal.csv",
    )
    parser.add_argument(
        "--training-subset",
        type=Path,
        default=None,
        help=(
            "CSV containing a unique 'mutant' column from this split's training positions. "
            "Its labels are the only labels used for target scaling and training."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/masked_adaptation/<backbone>_<loss>/split_<n>/seed_<n>",
    )
    return parser.parse_args()


def load_training_subset(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Training subset does not exist: {path}")
    subset = pd.read_csv(path)
    if "mutant" not in subset.columns:
        raise ValueError("Training subset must contain a 'mutant' column")
    mutants = subset["mutant"].astype(str).tolist()
    training_subset_fingerprint(mutants)
    return mutants


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    training_subset = load_training_subset(args.training_subset)
    output_dir = args.output_dir or (
        project_root()
        / "results"
        / "masked_adaptation"
        / f"{args.backbone}_{args.loss}"
        / f"split_{args.split_seed}"
        / f"seed_{args.training_seed}"
    )
    config = TrainingConfig(
        backbone_mode=args.backbone,
        loss_name=args.loss,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
        device_type=device.type,
        max_epochs=args.max_epochs,
        patience=args.patience,
        head_learning_rate=args.head_learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        weight_decay=args.weight_decay,
        maximum_gradient_norm=args.maximum_gradient_norm,
        mse_positions_per_batch=args.mse_positions_per_batch,
        list_size=args.list_size,
        lists_per_batch=args.lists_per_batch,
        list_microbatch_size=args.list_microbatch_size,
        evaluation_positions_per_batch=args.evaluation_positions_per_batch,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_labeled_train=len(training_subset) if training_subset is not None else None,
        training_subset_sha256=(
            training_subset_fingerprint(training_subset)
            if training_subset is not None
            else None
        ),
    )
    config.validate()
    if not args.zero_shot.exists():
        raise FileNotFoundError(
            f"Zero-shot predictions do not exist at {args.zero_shot}. "
            "Run scripts/score_zero_shot.py first."
        )

    seed_everything(args.training_seed)
    assay = load_assay()
    metadata = load_assay_metadata()
    zero_shot = pd.read_csv(args.zero_shot)
    split_manifest = make_splits(assay, seed=args.split_seed)[
        ["mutant", "position"]
    ].rename(columns={"position": "split"})

    backbone, alphabet = load_model(device)
    if args.backbone == "lora":
        injection = inject_lora(
            backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        expected_adapter_parameters = 34_560 * args.lora_rank
        if injection.trainable_parameters != expected_adapter_parameters:
            raise RuntimeError(
                "Unexpected ESM-2 LoRA parameter count: "
                f"{injection.trainable_parameters:,}; expected "
                f"{expected_adapter_parameters:,}"
            )
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    model = MaskedMarginalResidualModel(
        backbone,
        hidden_dim=config.hidden_dim,
        representation_layer=REPRESENTATION_LAYER,
    )
    prepared = prepare_mutation_data(
        assay,
        zero_shot,
        split_manifest,
        str(metadata["target_seq"]),
        alphabet,
        training_subset_mutants=training_subset,
    )
    frozen_site_cache_path = None
    if args.backbone == "frozen" and not args.no_persistent_frozen_site_cache:
        frozen_site_cache_path = args.frozen_site_cache or (
            project_root()
            / "data"
            / "processed"
            / f"esm2_t6_masked_site_representations_{device.type}.pt"
        )
    summary = run_training(
        model,
        prepared,
        config,
        output_dir,
        device,
        resume=args.resume,
        evaluate_test=args.evaluate_test,
        show_progress=True,
        frozen_site_cache_path=frozen_site_cache_path,
    )
    summary["model_name"] = MODEL_NAME
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
