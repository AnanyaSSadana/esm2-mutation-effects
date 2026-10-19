import importlib.util
import sys
import unittest
from pathlib import Path


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOCK = load_script("lock_lora_protocol")
RUNNER = load_script("run_lora_confirmatory")


class LoRAConfirmatoryTests(unittest.TestCase):
    def test_lock_requires_complete_validation_only_development(self):
        summary = {
            "matrix_complete": True,
            "test_evaluated": False,
            "completed_runs": 12,
            "summary": [],
        }
        lock = LOCK.build_lock(summary, "abc123")
        self.assertFalse(lock["test_evaluated_at_lock"])
        self.assertEqual(
            lock["protocol_sha256"],
            LOCK.canonical_sha256(lock["protocol"]),
        )

        with self.assertRaisesRegex(ValueError, "complete"):
            LOCK.build_lock({**summary, "matrix_complete": False}, "abc123")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            LOCK.build_lock({**summary, "test_evaluated": True}, "abc123")

    def test_confirmatory_command_requires_test_and_fixed_protocol(self):
        command = RUNNER.build_command(
            RUNNER.CONFIGURATIONS[3],
            device="cpu",
            split_seed=43,
            training_seed=2,
            confirmatory_root=Path("confirmatory"),
            development_root=Path("development"),
            resume=False,
        )
        self.assertIn("--evaluate-test", command)
        self.assertEqual(command[command.index("--patience") + 1], "5")
        self.assertEqual(command[command.index("--list-microbatch-size") + 1], "4")
        self.assertEqual(command[command.index("--split-seed") + 1], "43")
        self.assertEqual(command[command.index("--training-seed") + 1], "2")

    def test_split_42_reuses_development_checkpoint_directory(self):
        directory = RUNNER.run_directory(
            Path("confirmatory"),
            Path("development"),
            "lora_listmle",
            42,
            1,
        )
        self.assertEqual(
            directory,
            Path("development/lora_listmle/split_42/seed_1"),
        )


if __name__ == "__main__":
    unittest.main()
