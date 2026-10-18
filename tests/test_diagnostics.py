import unittest

import numpy as np

from esm2_mutation_effects.diagnostics import position_ranking_metrics, safe_spearman


class DiagnosticMetricTests(unittest.TestCase):
    def test_perfect_within_position_ranking(self):
        truth = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        prediction = np.array([0.1, 0.9, 2.1, 9.0, 11.5, 12.5])
        positions = np.array([1, 1, 1, 2, 2, 2])

        metrics = position_ranking_metrics(truth, prediction, positions)

        self.assertAlmostEqual(metrics["overall_spearman"], 1.0)
        self.assertAlmostEqual(metrics["between_position_spearman"], 1.0)
        self.assertAlmostEqual(metrics["within_position_pairwise_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["within_position_spearman_mean"], 1.0)
        self.assertEqual(metrics["n_within_position_pairs"], 6)
        self.assertEqual(metrics["n_positions_with_defined_spearman"], 2)

    def test_position_only_predictions_have_undefined_within_correlations(self):
        truth = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        prediction = np.array([1.0, 1.0, 1.0, 11.0, 11.0, 11.0])
        positions = np.array([1, 1, 1, 2, 2, 2])

        metrics = position_ranking_metrics(truth, prediction, positions)

        self.assertAlmostEqual(metrics["between_position_spearman"], 1.0)
        self.assertIsNone(metrics["within_position_centered_spearman"])
        self.assertAlmostEqual(metrics["within_position_pairwise_accuracy"], 0.5)
        self.assertIsNone(metrics["within_position_spearman_mean"])
        self.assertEqual(metrics["n_positions_with_defined_spearman"], 0)

    def test_safe_spearman_marks_constant_predictions_undefined(self):
        self.assertIsNone(safe_spearman(np.arange(4), np.ones(4)))


if __name__ == "__main__":
    unittest.main()
