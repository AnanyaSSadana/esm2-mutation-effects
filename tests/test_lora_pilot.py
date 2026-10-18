import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_lora_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_lora_pilot", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoRAPilotTests(unittest.TestCase):
    def test_commands_are_validation_only_and_fixed_to_five_epochs(self):
        for configuration in MODULE.PILOT_CONFIGURATIONS:
            command = MODULE.build_command(
                configuration,
                device="mps",
                split_seed=42,
                training_seed=1,
                output_root=Path("pilot"),
                resume=False,
            )
            self.assertNotIn("--evaluate-test", command)
            self.assertEqual(command[command.index("--max-epochs") + 1], "5")
            self.assertEqual(command[command.index("--patience") + 1], "5")
            self.assertNotIn("--resume", command)

    def test_resume_flag_is_added_only_when_requested(self):
        command = MODULE.build_command(
            MODULE.PILOT_CONFIGURATIONS[0],
            device="cpu",
            split_seed=42,
            training_seed=1,
            output_root=Path("pilot"),
            resume=True,
        )
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
