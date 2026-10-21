import unittest

import pandas as pd

from esm2_mutation_effects.phase3b_lite import (
    LABEL_BUDGET,
    NEW_SPLIT_SEEDS,
    make_protocol,
    validate_protocol,
)


class Phase3BLiteTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for position in range(1, 264):
            for alternate in "CD":
                rows.append({"mutant": f"A{position}{alternate}"})
        self.assay = pd.DataFrame(rows)

    def test_protocol_is_deterministic_and_strict(self):
        first_lock, first_manifests = make_protocol(self.assay, git_commit="a" * 40)
        second_lock, second_manifests = make_protocol(self.assay, git_commit="a" * 40)
        self.assertEqual(first_lock, second_lock)
        self.assertEqual(tuple(first_manifests), NEW_SPLIT_SEEDS)
        self.assertEqual(set(first_manifests), set(second_manifests))
        for manifest in first_manifests.values():
            self.assertEqual(int(manifest.target_label_visible.sum()), LABEL_BUDGET)
            self.assertFalse(manifest.loc[manifest.subset == "test", "target_label_visible"].any())
            self.assertTrue((manifest.groupby("position").subset.nunique() == 1).all())
        validate_protocol(first_lock, git_commit="a" * 40)

    def test_protocol_checksum_rejects_edits(self):
        lock, _ = make_protocol(self.assay, git_commit="a" * 40)
        lock["target_label_budget"] = 400
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_protocol(lock, git_commit="a" * 40)


if __name__ == "__main__":
    unittest.main()
