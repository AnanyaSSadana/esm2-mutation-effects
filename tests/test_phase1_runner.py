import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_phase1.py"
SPEC = importlib.util.spec_from_file_location("run_phase1", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Phase1RunnerTests(unittest.TestCase):
    def test_runner_has_no_test_step_before_lock(self):
        # The program structure invokes development first, and the development runner
        # has no test option; confirmatory evaluation is delegated only after locking.
        source = SCRIPT_PATH.read_text()
        self.assertLess(
            source.index("run_lora_development.py"),
            source.index("lock_lora_protocol.py"),
        )
        self.assertLess(
            source.index("lock_lora_protocol.py"),
            source.index("run_lora_confirmatory.py"),
        )


if __name__ == "__main__":
    unittest.main()
