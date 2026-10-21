import unittest

import pandas as pd

from esm2_mutation_effects.phase3_data import validate_pilot_assay
from esm2_mutation_effects.phase3_pilot import PilotDataRequirements


class Phase3DataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "DMS_id": "TEST_ASSAY",
            "target_seq": "ACDEFG",
            "seq_len": 6,
            "includes_multiple_mutants": False,
        }
        self.requirements = PilotDataRequirements(
            minimum_valid_single_substitutions=2,
            minimum_mutated_positions=2,
            maximum_sequence_length=10,
            list_size=2,
        )
        self.frame = pd.DataFrame(
            {
                "mutant": ["A1C", "C2A"],
                "mutated_sequence": ["CCDEFG", "AADEFG"],
                "DMS_score": [-1.0, 0.5],
            }
        )

    def test_validates_single_substitution_table(self):
        summary = validate_pilot_assay(
            self.frame, self.metadata, requirements=self.requirements
        )
        self.assertEqual(summary["valid_single_substitutions"], 2)
        self.assertEqual(summary["unique_mutated_positions"], 2)
        self.assertEqual(summary["dms_score_direction"], "higher_is_better_by_ProteinGym_processed_score_convention")

    def test_rejects_inconsistent_mutated_sequence(self):
        invalid = self.frame.copy()
        invalid.loc[0, "mutated_sequence"] = "CCAEFG"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_pilot_assay(invalid, self.metadata, requirements=self.requirements)

    def test_rejects_multiple_mutant_metadata(self):
        metadata = dict(self.metadata, includes_multiple_mutants=True)
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_pilot_assay(self.frame, metadata, requirements=self.requirements)


if __name__ == "__main__":
    unittest.main()
