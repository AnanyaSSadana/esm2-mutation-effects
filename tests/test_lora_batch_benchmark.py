import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_lora_list_batching.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lora_list_batching", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoRAListBatchBenchmarkTests(unittest.TestCase):
    def test_command_uses_identical_subset_and_never_evaluates_test(self):
        command = MODULE.build_command(
            8,
            device="mps",
            split_seed=42,
            sampling_seed=1,
            label_budget=2004,
            training_seed=1,
            output_root=Path("benchmark"),
            training_subset=Path("benchmark/subset_manifest.csv"),
            resume=False,
        )
        self.assertNotIn("--evaluate-test", command)
        self.assertEqual(command[command.index("--lists-per-batch") + 1], "16")
        self.assertEqual(command[command.index("--list-microbatch-size") + 1], "8")
        self.assertEqual(command[command.index("--max-epochs") + 1], "1")
        self.assertEqual(
            command[command.index("--training-subset") + 1],
            "benchmark/subset_manifest.csv",
        )

    def test_default_candidates_include_all_three_sizes(self):
        self.assertEqual(MODULE.DEFAULT_MICROBATCH_SIZES, (4, 8, 16))

    def test_completed_result_rejects_test_artifacts(self):
        metadata = {"n_labeled_train": 2004, "training_subset_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            config = {
                "backbone_mode": "lora",
                "loss_name": "listmle",
                "split_seed": 42,
                "training_seed": 1,
                "max_epochs": 1,
                "patience": 1,
                "lists_per_batch": 16,
                "list_microbatch_size": 4,
                "ranking_target_representation": "ordinal_rank_from_float64",
                **metadata,
            }
            (run_dir / "summary.json").write_text(
                json.dumps({"config": config, "test_evaluated": False})
            )
            (run_dir / "history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "epoch_seconds": 1.0,
                            "allocated_memory_mib": 10.0,
                            "train_loss": 2.0,
                            "validation_metrics": {"overall_spearman": 0.3},
                            "maximum_gradient_norm": 3.0,
                            "maximum_lora_gradient_norm": 1.0,
                        }
                    ]
                )
            )
            (run_dir / "test_metrics.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "forbidden test artifact"):
                MODULE.load_completed_result(
                    run_dir,
                    microbatch_size=4,
                    split_seed=42,
                    training_seed=1,
                    metadata=metadata,
                )


if __name__ == "__main__":
    unittest.main()
