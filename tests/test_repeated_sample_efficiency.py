import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_repeated_sample_efficiency.py"
SPEC = importlib.util.spec_from_file_location("run_repeated_sample_efficiency", SCRIPT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


class RepeatedSampleEfficiencyTests(unittest.TestCase):
    def test_combined_seed_is_deterministic_and_uses_both_inputs(self):
        first = experiment.combined_sampling_seed(42, 1)
        self.assertEqual(first, experiment.combined_sampling_seed(42, 1))
        self.assertNotEqual(first, experiment.combined_sampling_seed(43, 1))
        self.assertNotEqual(first, experiment.combined_sampling_seed(42, 2))

    def test_summary_keeps_zero_shot_runs_independent(self):
        rows = []
        for split_seed in (42, 43):
            rows.append(
                {
                    "split_seed": split_seed,
                    "sampling_seed": None,
                    "train_fraction": 0.1,
                    "model": "zero_shot_esm2",
                    "n_labeled_train": 0,
                    "test_spearman": 0.3 + 0.1 * (split_seed - 42),
                }
            )
            for sampling_seed in (1, 2):
                for model, offset in (("mutation_features", 0.0), ("frozen_esm2", 0.2)):
                    rows.append(
                        {
                            "split_seed": split_seed,
                            "sampling_seed": sampling_seed,
                            "train_fraction": 0.1,
                            "model": model,
                            "n_labeled_train": 40,
                            "test_spearman": (
                                offset
                                + 0.1 * (split_seed - 42)
                                + 0.01 * sampling_seed
                            ),
                        }
                    )

        summary = experiment.summarize(pd.DataFrame(rows)).set_index("model")
        self.assertEqual(summary.loc["zero_shot_esm2", "n_runs"], 2)
        self.assertEqual(summary.loc["zero_shot_esm2", "n_sampling_seeds"], 0)
        self.assertEqual(summary.loc["frozen_esm2", "n_runs"], 4)
        self.assertEqual(summary.loc["frozen_esm2", "n_sampling_seeds"], 2)
        self.assertAlmostEqual(summary.loc["zero_shot_esm2", "mean_test_spearman"], 0.35)
        self.assertGreater(
            summary.loc["frozen_esm2", "mean_within_split_sampling_std"],
            0.0,
        )
        self.assertTrue(np.isfinite(summary["between_split_std_of_means"]).all())


if __name__ == "__main__":
    unittest.main()
