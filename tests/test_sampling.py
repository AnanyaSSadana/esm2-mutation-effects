import unittest

import numpy as np

from esm2_mutation_effects.sampling import nested_training_subsets, nested_training_subsets_by_size


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

    def test_exact_sizes_are_deterministic_and_nested(self):
        indices = np.arange(100, 200)
        first = nested_training_subsets_by_size(indices, (20, 40, 100), seed=7)
        second = nested_training_subsets_by_size(indices, (20, 40, 100), seed=7)

        self.assertEqual({size: len(rows) for size, rows in first.items()}, {20: 20, 40: 40, 100: 100})
        for size in first:
            np.testing.assert_array_equal(first[size], second[size])
        self.assertTrue(set(first[20]).issubset(first[40]))
        self.assertTrue(set(first[40]).issubset(first[100]))
        with self.assertRaises(ValueError):
            nested_training_subsets_by_size(indices, (20, 20), seed=7)
        with self.assertRaises(ValueError):
            nested_training_subsets_by_size(indices, (101,), seed=7)


if __name__ == "__main__":
    unittest.main()
