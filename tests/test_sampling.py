import unittest

import numpy as np

from esm2_mutation_effects.sampling import nested_training_subsets


class SamplingTests(unittest.TestCase):
    def test_subsets_are_deterministic_nested_and_correctly_sized(self):
        indices = np.arange(100, 300)
        fractions = (0.01, 0.05, 0.10, 0.50, 1.0)

        first = nested_training_subsets(indices, fractions, seed=42)
        second = nested_training_subsets(indices, fractions, seed=42)

        self.assertEqual({key: len(value) for key, value in first.items()}, {
            0.01: 2,
            0.05: 10,
            0.10: 20,
            0.50: 100,
            1.0: 200,
        })
        for fraction in fractions:
            np.testing.assert_array_equal(first[fraction], second[fraction])
        ordered = [set(first[fraction]) for fraction in fractions]
        for smaller, larger in zip(ordered, ordered[1:]):
            self.assertTrue(smaller.issubset(larger))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            nested_training_subsets([], (0.1,), seed=42)
        with self.assertRaises(ValueError):
            nested_training_subsets([1, 1], (0.1,), seed=42)
        with self.assertRaises(ValueError):
            nested_training_subsets([1, 2], (), seed=42)
        with self.assertRaises(ValueError):
            nested_training_subsets([1, 2], (0.0,), seed=42)
        with self.assertRaises(ValueError):
            nested_training_subsets([1, 2], (0.5, 0.5), seed=42)


if __name__ == "__main__":
    unittest.main()
