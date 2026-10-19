import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_lora_list_batching.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lora_list_batching", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoRAListBatchBenchmarkTests(unittest.TestCase):
    def test_command_preserves_effective_batch_and_never_evaluates_test(self):
        command = MODULE.build_command(
            8,
            device="mps",
            split_seed=42,
            training_seed=1,
            output_root=Path("benchmark"),
            resume=False,
        )
        self.assertNotIn("--evaluate-test", command)
        self.assertEqual(command[command.index("--lists-per-batch") + 1], "16")
        self.assertEqual(command[command.index("--list-microbatch-size") + 1], "8")
        self.assertEqual(command[command.index("--max-epochs") + 1], "1")

    def test_resume_flag_is_optional(self):
        command = MODULE.build_command(
            16,
            device="cpu",
            split_seed=42,
            training_seed=1,
            output_root=Path("benchmark"),
            resume=True,
        )
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
