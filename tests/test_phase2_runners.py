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


smoke = load_script("run_phase2_smoke.py")
lock = load_script("lock_phase2_protocol.py")
runner = load_script("run_phase2_label_efficiency.py")


class Phase2RunnerTests(unittest.TestCase):
    def test_smoke_command_is_validation_only(self):
        command = smoke.build_command(device="cpu", output_root=Path("/tmp/phase2"), resume=False)
        self.assertNotIn("--evaluate-test", command)
        self.assertIn("--training-subset", command)
        self.assertNotIn("--resume", command)

    def test_lock_protocol_matches_confirmation_runner(self):
        smoke_summary = {
            "experiment": "phase_2_validation_only_20_label_smoke",
            "test_evaluated": False,
            "record": {
                "configuration": "lora_listmle",
                "split_seed": 42,
                "sampling_seed": 1,
                "n_labeled_train": 20,
                "training_subset_sha256": "a" * 64,
                "test_evaluated": False,
            },
        }
        built = lock.build_lock(smoke_summary, "abc")
        self.assertEqual(built["protocol"], runner.canonical_protocol())
        self.assertEqual(built["protocol_sha256"], runner.canonical_sha256(built["protocol"]))

    def test_confirmation_run_count_and_full_budget_policy(self):
        self.assertEqual(runner.expected_run_count(), 190)
        command = runner.build_command(
            runner.CONFIGURATIONS[1],
            device="cpu",
            split_seed=42,
            output_root=Path("/tmp/phase2"),
            sampling_seed=1,
            label_budget=20,
            resume=True,
        )
        self.assertIn("--evaluate-test", command)
        self.assertIn("--resume", command)
        self.assertIn("labels_20", " ".join(command))
