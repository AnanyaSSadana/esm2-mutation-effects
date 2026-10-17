import unittest

from esm2_mutation_effects.data import load_assay
from esm2_mutation_effects.splits import make_splits, validate_splits


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assay = load_assay()

    def test_splits_are_deterministic_and_complete(self):
        first = make_splits(self.assay, seed=42)
        second = make_splits(self.assay, seed=42)
        self.assertTrue(first.equals(second))
        validate_splits(first, set(self.assay["mutant"]))

    def test_position_split_has_no_position_leakage(self):
        splits = make_splits(self.assay, seed=42)
        labels_per_position = splits.groupby("residue_position")["position"].nunique()
        self.assertTrue((labels_per_position == 1).all())
        self.assertEqual(set(splits["position"]), {"train", "validation", "test"})

    def test_expected_sizes(self):
        splits = make_splits(self.assay, seed=42)
        self.assertEqual(
            splits["random"].value_counts().to_dict(),
            {"train": 3996, "test": 500, "validation": 500},
        )
        position_counts = splits.groupby("position")["residue_position"].nunique().to_dict()
        self.assertEqual(position_counts, {"train": 211, "test": 26, "validation": 26})


if __name__ == "__main__":
    unittest.main()
