import unittest

import pandas as pd

from esm2_mutation_effects.phase3_audit import (
    AuditThresholds,
    audit_substitution_metadata,
    smith_waterman_local_alignment,
)


def metadata_row(
    assay_id: str,
    sequence: str,
    *,
    uniprot: str,
    n_single: int = 600,
    multiple: bool = False,
    direction: int = 1,
    molecule_name: str = "protein",
) -> dict[str, object]:
    return {
        "DMS_id": assay_id,
        "DMS_filename": f"{assay_id}.csv",
        "UniProt_ID": uniprot,
        "target_seq": sequence,
        "seq_len": len(sequence),
        "includes_multiple_mutants": multiple,
        "DMS_number_single_mutants": n_single,
        "raw_DMS_directionality": direction,
        "molecule_name": molecule_name,
        "title": "A DMS study",
        "selection_assay": "growth",
        "selection_type": "Growth",
        "coarse_selection_type": "OrganismalFitness",
    }


class Phase3AuditTests(unittest.TestCase):
    def setUp(self):
        self.tem = "ACDEFGHIKLMNPQRSTVWY" * 4
        self.unrelated = "Y" * 80

    def test_local_alignment_reports_identity_and_reference_coverage(self):
        alignment = smith_waterman_local_alignment(self.tem, self.tem)
        self.assertEqual(alignment.identical_pairs, len(self.tem))
        self.assertEqual(alignment.identity, 1.0)
        self.assertEqual(alignment.reference_coverage, 1.0)

    def test_metadata_audit_separates_target_controls_and_strict_candidates(self):
        frame = pd.DataFrame(
            [
                metadata_row("BLAT_ECOLX_Stiffler_2015", self.tem, uniprot="BLAT_ECOLX"),
                metadata_row("BLAT_ECOLX_Deng_2012", self.tem, uniprot="BLAT_ECOLX"),
                metadata_row(
                    "VIM2_TEST",
                    "M" + self.tem[1:],
                    uniprot="VIM2",
                    molecule_name="Beta-lactamase",
                ),
                metadata_row("OTHER_TEST", self.unrelated, uniprot="OTHER"),
                metadata_row("TOO_SMALL", self.unrelated, uniprot="SMALL", n_single=499),
            ]
        )
        registry, summary = audit_substitution_metadata(frame)
        by_assay = registry.set_index("DMS_id")
        self.assertEqual(by_assay.loc["BLAT_ECOLX_Stiffler_2015", "target_role"], "target")
        self.assertEqual(
            by_assay.loc["BLAT_ECOLX_Deng_2012", "target_role"],
            "same_protein_positive_control",
        )
        self.assertEqual(
            by_assay.loc["VIM2_TEST", "target_role"], "tem_related_excluded_from_strict"
        )
        self.assertEqual(by_assay.loc["OTHER_TEST", "target_role"], "non_target")
        self.assertEqual(by_assay.loc["TOO_SMALL", "metadata_eligibility"], "excluded")
        self.assertEqual(summary["strict_metadata_candidates"], 1)
        self.assertEqual(summary["status"], "metadata_only_not_a_final_cohort")

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditThresholds(minimum_single_substitutions=0).validate()


if __name__ == "__main__":
    unittest.main()
