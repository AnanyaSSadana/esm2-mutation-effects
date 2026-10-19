import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_lora_development.py"
SPEC = importlib.util.spec_from_file_location("run_lora_development", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoRADevelopmentTests(unittest.TestCase):
    def test_matrix_commands_are_fixed_and_validation_only(self):
        commands = []
        for training_seed in MODULE.DEVELOPMENT_TRAINING_SEEDS:
            for configuration in MODULE.DEVELOPMENT_CONFIGURATIONS:
                command = MODULE.build_command(
                    configuration,
                    device="mps",
                    training_seed=training_seed,
                    output_root=Path("development"),
                    resume=False,
                )
                commands.append(command)
                self.assertNotIn("--evaluate-test", command)
                self.assertEqual(command[command.index("--split-seed") + 1], "42")
                self.assertEqual(command[command.index("--max-epochs") + 1], "100")
                self.assertEqual(command[command.index("--patience") + 1], "5")
                self.assertEqual(command[command.index("--lists-per-batch") + 1], "16")
                self.assertEqual(command[command.index("--list-microbatch-size") + 1], "4")
                self.assertNotIn("--resume", command)
        self.assertEqual(len(commands), 12)

    def test_resume_flag_is_optional(self):
        command = MODULE.build_command(
            MODULE.DEVELOPMENT_CONFIGURATIONS[0],
            device="cpu",
            training_seed=1,
            output_root=Path("development"),
            resume=True,
        )
        self.assertIn("--resume", command)

    def test_summary_rejects_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory_name:
            output_root = Path(directory_name)
            run_dir = MODULE.run_directory(output_root, "frozen_mse", 1)
            run_dir.mkdir(parents=True)
            config = {
                "backbone_mode": "frozen",
                "loss_name": "mse",
                "split_seed": 42,
                "training_seed": 1,
                "max_epochs": 100,
                "patience": 5,
                "lists_per_batch": 16,
                "list_microbatch_size": 4,
            }
            summary = {
                "config": config,
                "status": "stopped_early",
                "best_epoch": 2,
                "best_validation_spearman": 0.5,
                "best_validation_loss": 1.0,
                "epochs_completed": 17,
                "trainable_parameters": 6420,
                "test_evaluated": False,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "validation_metrics.json").write_text(
                json.dumps({"overall_spearman": 0.5})
            )
            (run_dir / "history.json").write_text(
                json.dumps([{"epoch": 0, "epoch_seconds": 0.0}])
            )
            (run_dir / "test_metrics.json").write_text("{}")

            with self.assertRaisesRegex(RuntimeError, "forbidden test artifacts"):
                MODULE.collect_development_summary(output_root)


if __name__ == "__main__":
    unittest.main()
