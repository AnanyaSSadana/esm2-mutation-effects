import tempfile
import unittest
from pathlib import Path

from esm2_mutation_effects.data import load_assay
from esm2_mutation_effects.label_efficiency import (
    manifest_metadata,
    nested_mutation_manifests,
    write_manifest,
)
from esm2_mutation_effects.splits import make_splits


class LabelEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assay = load_assay()

    def test_manifests_are_nested_and_limited_to_training_positions(self):
        budgets = (20, 40, 4008)
        first = nested_mutation_manifests(
            self.assay,
            split_seed=42,
            sampling_seed=1,
            budgets=budgets,
        )
        second = nested_mutation_manifests(
            self.assay,
            split_seed=42,
            sampling_seed=1,
            budgets=budgets,
        )
        self.assertEqual({budget: len(value) for budget, value in first.items()}, {
            20: 20,
            40: 40,
            4008: 4008,
        })
        self.assertTrue(set(first[20]["mutant"]).issubset(first[40]["mutant"]))
        self.assertTrue(set(first[40]["mutant"]).issubset(first[4008]["mutant"]))
        self.assertEqual(manifest_metadata(first[40]), manifest_metadata(second[40]))

        split = make_splits(self.assay, seed=42).set_index("mutant")
        self.assertEqual(set(split.loc[first[40]["mutant"], "position"]), {"train"})

    def test_manifest_write_rejects_conflicting_existing_content(self):
        manifests = nested_mutation_manifests(
            self.assay,
            split_seed=42,
            sampling_seed=1,
            budgets=(20, 40),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset_manifest.csv"
            expected = write_manifest(path, manifests[20])
            self.assertEqual(expected, write_manifest(path, manifests[20]))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                write_manifest(path, manifests[40])
