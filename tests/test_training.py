import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from esm.data import Alphabet
from esm.model.esm2 import ESM2

from esm2_mutation_effects.adaptation import MaskedMarginalResidualModel
from esm2_mutation_effects.lora import inject_lora
from esm2_mutation_effects.training import (
    TrainingConfig,
    _predict_rows,
    cache_frozen_site_representations,
    ordinal_ranking_targets,
    prepare_mutation_data,
    run_training,
    seed_everything,
)


def synthetic_data():
    assay = pd.DataFrame(
        {
            "mutant": ["A1C", "A1D", "C2A", "C2D", "D3A", "D3C", "E4A", "E4C"],
            "DMS_score": [-1.2, -0.8, -0.4, 0.1, 0.4, 0.8, 1.1, 1.4],
        }
    )
    zero_shot = pd.DataFrame(
        {
            "mutant": assay["mutant"],
            "predicted_score": [-1.0, -0.5, -0.2, 0.2, 0.3, 0.7, 0.9, 1.2],
        }
    )
    split = pd.DataFrame(
        {
            "mutant": assay["mutant"],
            "split": ["train", "train", "train", "train", "validation", "validation", "test", "test"],
        }
    )
    return assay, zero_shot, split


class TrainingTests(unittest.TestCase):
    def _build_smoke_run(self):
        seed_everything(21)
        alphabet = Alphabet.from_architecture("ESM-1b")
        backbone = ESM2(num_layers=1, embed_dim=16, attention_heads=4)
        inject_lora(backbone, rank=2, alpha=2, dropout=0.0)
        model = MaskedMarginalResidualModel(
            backbone,
            hidden_dim=16,
            representation_layer=1,
        )
        assay, zero_shot, split = synthetic_data()
        prepared = prepare_mutation_data(assay, zero_shot, split, "ACDE", alphabet)
        config = TrainingConfig(
            backbone_mode="lora",
            loss_name="listmle",
            split_seed=1,
            training_seed=21,
            device_type="cpu",
            model_name="synthetic_esm2",
            representation_layer=1,
            hidden_dim=16,
            lora_rank=2,
            lora_alpha=2,
            lora_dropout=0.0,
            max_epochs=1,
            patience=2,
            list_size=2,
            lists_per_batch=2,
            list_microbatch_size=1,
            evaluation_positions_per_batch=1,
        )
        return model, prepared, config

    def test_frozen_mse_training_path(self):
        seed_everything(22)
        alphabet = Alphabet.from_architecture("ESM-1b")
        backbone = ESM2(num_layers=1, embed_dim=16, attention_heads=4)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        model = MaskedMarginalResidualModel(
            backbone,
            hidden_dim=16,
            representation_layer=1,
        )
        assay, zero_shot, split = synthetic_data()
        prepared = prepare_mutation_data(assay, zero_shot, split, "ACDE", alphabet)
        config = TrainingConfig(
            backbone_mode="frozen",
            loss_name="mse",
            split_seed=1,
            training_seed=22,
            device_type="cpu",
            model_name="synthetic_esm2",
            representation_layer=1,
            hidden_dim=16,
            max_epochs=1,
            patience=2,
            mse_positions_per_batch=1,
            evaluation_positions_per_batch=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(
                model,
                prepared,
                config,
                Path(directory) / "frozen_mse",
                torch.device("cpu"),
            )
        self.assertEqual(summary["trainable_parameters"], 16 * 20 + 20)
        self.assertTrue(summary["frozen_site_representation_cache"])
        self.assertFalse(summary["test_evaluated"])

    def test_frozen_cache_matches_direct_backbone_predictions(self):
        seed_everything(23)
        alphabet = Alphabet.from_architecture("ESM-1b")
        backbone = ESM2(num_layers=1, embed_dim=16, attention_heads=4).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        model = MaskedMarginalResidualModel(
            backbone,
            hidden_dim=16,
            representation_layer=1,
        ).eval()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.02)
            model.residual_head.bias.normal_(std=0.02)
        assay, zero_shot, split = synthetic_data()
        prepared = prepare_mutation_data(assay, zero_shot, split, "ACDE", alphabet)
        rows = torch.arange(len(prepared.frame))
        with torch.no_grad():
            direct = _predict_rows(model, prepared, rows, torch.device("cpu"))
            cache = cache_frozen_site_representations(
                model,
                prepared,
                torch.device("cpu"),
                positions_per_batch=4,
            )
            cached = _predict_rows(
                model,
                prepared,
                rows,
                torch.device("cpu"),
                cache,
            )
        self.assertTrue(torch.equal(cached, direct))

        trainable = ESM2(num_layers=1, embed_dim=16, attention_heads=4)
        invalid_model = MaskedMarginalResidualModel(
            trainable,
            hidden_dim=16,
            representation_layer=1,
        )
        with self.assertRaisesRegex(ValueError, "fully frozen"):
            cache_frozen_site_representations(
                invalid_model,
                prepared,
                torch.device("cpu"),
                positions_per_batch=4,
            )

    def test_smoke_train_save_and_resume_without_test_evaluation(self):
        model, prepared, config = self._build_smoke_run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            summary = run_training(
                model,
                prepared,
                config,
                output,
                torch.device("cpu"),
            )
            self.assertFalse(summary["test_evaluated"])
            self.assertFalse(summary["frozen_site_representation_cache"])
            self.assertEqual(summary["epochs_completed"], 1)
            for filename in (
                "config.json",
                "last.pt",
                "best.pt",
                "history.json",
                "validation_predictions.csv",
                "validation_metrics.json",
                "summary.json",
            ):
                self.assertTrue((output / filename).exists(), filename)
            self.assertFalse((output / "test_predictions.csv").exists())
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in history], [0, 1])

            resumed = run_training(
                model,
                prepared,
                config,
                output,
                torch.device("cpu"),
                resume=True,
            )
            self.assertEqual(resumed["epochs_completed"], 1)

    def test_test_evaluation_requires_explicit_option(self):
        model, prepared, config = self._build_smoke_run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            summary = run_training(
                model,
                prepared,
                config,
                output,
                torch.device("cpu"),
                evaluate_test=True,
            )
            self.assertTrue(summary["test_evaluated"])
            self.assertTrue((output / "test_predictions.csv").exists())
            self.assertTrue((output / "test_metrics.json").exists())

    def test_existing_run_requires_resume_and_matching_configuration(self):
        model, prepared, config = self._build_smoke_run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            run_training(model, prepared, config, output, torch.device("cpu"))
            with self.assertRaises(FileExistsError):
                run_training(model, prepared, config, output, torch.device("cpu"))

            moved_device = TrainingConfig(**{**config.serializable(), "device_type": "mps"})
            resumed = run_training(
                model,
                prepared,
                moved_device,
                output,
                torch.device("cpu"),
                resume=True,
            )
            self.assertEqual(resumed["epochs_completed"], 1)

            changed = TrainingConfig(**{**config.serializable(), "max_epochs": 2})
            with self.assertRaises(ValueError):
                run_training(
                    model,
                    prepared,
                    changed,
                    output,
                    torch.device("cpu"),
                    resume=True,
                )

    def test_prepare_data_rejects_position_leakage(self):
        assay, zero_shot, split = synthetic_data()
        split.loc[split["mutant"] == "A1D", "split"] = "validation"
        alphabet = Alphabet.from_architecture("ESM-1b")
        with self.assertRaisesRegex(ValueError, "leaks"):
            prepare_mutation_data(assay, zero_shot, split, "ACDE", alphabet)

    def test_training_subset_controls_rows_scaling_and_fingerprint(self):
        assay, zero_shot, split = synthetic_data()
        alphabet = Alphabet.from_architecture("ESM-1b")
        prepared = prepare_mutation_data(
            assay,
            zero_shot,
            split,
            "ACDE",
            alphabet,
            training_subset_mutants=("A1C", "C2A"),
        )
        self.assertEqual(prepared.frame.iloc[prepared.training_rows]["mutant"].tolist(), ["A1C", "C2A"])
        self.assertAlmostEqual(prepared.target_standardizer.mean, -0.8)
        self.assertIsNotNone(prepared.training_subset_sha256)
        with self.assertRaisesRegex(ValueError, "outside the training positions"):
            prepare_mutation_data(
                assay,
                zero_shot,
                split,
                "ACDE",
                alphabet,
                training_subset_mutants=("E4A",),
            )

    def test_ordinal_ranking_preserves_close_float64_targets(self):
        low = np.float64(-1.9791040000000000)
        high = np.nextafter(low, np.float64(0.0))
        self.assertEqual(np.float32(low), np.float32(high))
        ranks = ordinal_ranking_targets(np.asarray([high, low, 0.2]))
        self.assertEqual(ranks.tolist(), [1.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
