import unittest

import torch

from esm2_mutation_effects.ranking import TargetStandardizer, listmle_loss, mse_loss


class RankingTests(unittest.TestCase):
    def test_correct_ranking_has_lower_listmle_loss(self):
        targets = torch.tensor([[3.0, 2.0, 1.0]])
        correct = listmle_loss(torch.tensor([[4.0, 1.0, -2.0]]), targets)
        reversed_order = listmle_loss(torch.tensor([[-2.0, 1.0, 4.0]]), targets)
        self.assertLess(float(correct), float(reversed_order))

    def test_listmle_is_differentiable(self):
        predictions = torch.tensor([[0.1, -0.2, 0.3], [1.0, 0.0, -1.0]], requires_grad=True)
        targets = torch.tensor([[2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
        loss = listmle_loss(predictions, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predictions.grad)
        self.assertTrue(torch.all(torch.isfinite(predictions.grad)))

    def test_listmle_rejects_ties_and_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, "ties"):
            listmle_loss(torch.zeros(1, 3), torch.tensor([[1.0, 1.0, 0.0]]))
        with self.assertRaises(ValueError):
            listmle_loss(torch.zeros(3), torch.zeros(3))
        with self.assertRaises(ValueError):
            listmle_loss(torch.zeros(1, 3), torch.zeros(1, 2))

    def test_target_standardization_uses_supplied_training_values(self):
        training = torch.tensor([1.0, 2.0, 3.0])
        standardizer = TargetStandardizer.fit(training)
        transformed = standardizer.transform(training)
        self.assertAlmostEqual(float(transformed.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(transformed.std(unbiased=False)), 1.0, places=6)
        self.assertTrue(torch.allclose(standardizer.inverse_transform(transformed), training))
        with self.assertRaises(ValueError):
            TargetStandardizer.fit(torch.ones(3))

    def test_mse_validates_shape(self):
        self.assertAlmostEqual(
            float(mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 2.0]))),
            0.5,
        )
        with self.assertRaises(ValueError):
            mse_loss(torch.zeros(2), torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
