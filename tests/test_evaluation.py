import unittest

import pandas as pd

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.evaluation import select_evaluation_subset
from esm2_mutation_effects.metrics import spearman


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assay = load_assay()
        cls.predictions = assay[["mutant", "DMS_score"]].rename(
            columns={"DMS_score": "predicted_score"}
        )
        cls.truth = assay.set_index("mutant")["DMS_score"]
        cls.splits = pd.read_csv(project_root() / "data" / "splits" / "splits.csv")

    def test_selects_expected_test_subsets(self):
        expected_sizes = {"position": 494, "random": 500}
        for strategy, expected_size in expected_sizes.items():
            with self.subTest(strategy=strategy):
                selected = select_evaluation_subset(
                    self.predictions,
                    "predicted_score",
                    split_manifest=self.splits,
                    split_strategy=strategy,
                    subset="test",
                )
                self.assertEqual(len(selected), expected_size)
                observed_truth = self.truth.loc[selected["mutant"]].to_numpy()
                self.assertAlmostEqual(
                    spearman(observed_truth, selected["predicted_score"].to_numpy()), 1.0
                )

    def test_requires_split_arguments_together(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            select_evaluation_subset(
                self.predictions,
                "predicted_score",
                split_manifest=self.splits,
                split_strategy="position",
            )

    def test_rejects_duplicate_predictions(self):
        duplicated = pd.concat([self.predictions.iloc[:2], self.predictions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicate mutants"):
            select_evaluation_subset(duplicated, "predicted_score")


if __name__ == "__main__":
    unittest.main()
