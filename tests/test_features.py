import unittest

import numpy as np

from esm2_mutation_effects.features import select_nonredundant_site_features


class FeatureTests(unittest.TestCase):
    def test_removes_exactly_redundant_mutant_block(self):
        wild_type = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mutant = np.array([[1.5, 1.0], [2.0, 6.0]], dtype=np.float32)
        delta = mutant - wild_type
        cached = np.concatenate([wild_type, mutant, delta], axis=1)

        selected = select_nonredundant_site_features(cached)

        np.testing.assert_array_equal(selected, np.concatenate([wild_type, delta], axis=1))
        self.assertEqual(selected.shape, (2, 4))
        self.assertEqual(selected.dtype, np.float64)

    def test_rejects_invalid_feature_matrices(self):
        invalid = (
            np.ones(6),
            np.ones((2, 5)),
            np.array([[1.0, 2.0, np.nan]]),
        )
        for values in invalid:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                select_nonredundant_site_features(values)


if __name__ == "__main__":
    unittest.main()
