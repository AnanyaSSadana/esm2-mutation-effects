import importlib.util
import unittest
from pathlib import Path


def load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = load_script("analyze_phase2.py")
wandb_sync = load_script("sync_phase2_to_wandb.py")


class Phase2AnalysisTests(unittest.TestCase):
    def test_expected_matrix_contains_190_unique_runs(self):
        identities = analysis.expected_identities()
        self.assertEqual(len(identities), 190)
        self.assertIn(("lora_listmle", 42, 1, 20), identities)
        self.assertIn(("frozen_listmle", 46, None, 4008), identities)
        self.assertNotIn(("lora_listmle", 42, 2, 4008), identities)

    def test_wandb_identifiers_and_remote_path_mapping_are_deterministic(self):
        record = {
            "configuration": "lora_listmle",
            "split_seed": 43,
            "sampling_seed": None,
            "label_budget": 4008,
            "run_directory": (
                "/content/drive/MyDrive/esm2-phase2-colab/results/phase2_label_efficiency/"
                "lora_listmle/split_43/sample_1/labels_4008"
            ),
        }
        self.assertEqual(wandb_sync.run_id(record), "p2-lora_listmle-s43-ufull-n4008")
        self.assertEqual(
            wandb_sync.relative_run_directory(record["run_directory"]),
            Path("lora_listmle/split_43/sample_1/labels_4008"),
        )

    def test_epoch_metrics_excludes_missing_values(self):
        metrics = wandb_sync.epoch_metrics(
            {
                "epoch": 3,
                "train_loss": 1.2,
                "validation_loss": 0.8,
                "validation_metrics": {"overall_spearman": 0.4},
                "epoch_seconds": 2.5,
                "allocated_memory_mib": None,
                "maximum_gradient_norm": 1.1,
                "maximum_head_gradient_norm": None,
                "maximum_lora_gradient_norm": 0.3,
            }
        )
        self.assertEqual(metrics["epoch"], 3)
        self.assertEqual(metrics["validation/spearman"], 0.4)
        self.assertNotIn("system/allocated_memory_mib", metrics)
        self.assertNotIn("gradients/maximum_head_norm", metrics)


if __name__ == "__main__":
    unittest.main()
