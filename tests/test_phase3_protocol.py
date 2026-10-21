import unittest

import pandas as pd

from esm2_mutation_effects.phase3_protocol import (
    make_strict_target_manifest,
    manifest_sha256,
    validate_strict_target_manifest,
)


class Phase3ProtocolTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for position, wild_type in enumerate("ACDEFGHIKL", start=1):
            for alternate in "MN":
                rows.append({"mutant": f"{wild_type}{position}{alternate}"})
        self.assay = pd.DataFrame(rows)

    def test_manifest_is_deterministic_and_strict(self):
        first = make_strict_target_manifest(self.assay, label_budget=12)
        second = make_strict_target_manifest(self.assay, label_budget=12)
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertEqual(int(first.target_label_visible.sum()), 12)
        self.assertFalse(first.loc[first.subset == "test", "target_label_visible"].any())
        self.assertTrue((first.groupby("position").subset.nunique() == 1).all())

    def test_validator_rejects_test_label(self):
        manifest = make_strict_target_manifest(self.assay, label_budget=12)
        row = manifest.index[manifest.subset == "test"][0]
        manifest.loc[row, "target_label_visible"] = True
        with self.assertRaisesRegex(ValueError, "test"):
            validate_strict_target_manifest(manifest, label_budget=12)


if __name__ == "__main__":
    unittest.main()
