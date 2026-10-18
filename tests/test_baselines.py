import unittest

import numpy as np

from esm2_mutation_effects.baselines import make_frozen_esm2_model, select_alpha


class BaselineTests(unittest.TestCase):
    def test_select_alpha_returns_complete_validation_record(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 5))
        y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.01, size=60)

        selected, candidates = select_alpha(
            make_frozen_esm2_model,
            (0.1, 1.0, 10.0),
            x[:40],
            y[:40],
            x[40:],
            y[40:],
        )

        self.assertEqual(len(candidates), 3)
        self.assertIn(selected, candidates)
        self.assertIn(selected["alpha"], {0.1, 1.0, 10.0})
        self.assertTrue(np.isfinite(selected["validation_spearman"]))


if __name__ == "__main__":
    unittest.main()
