import unittest

from esm2_mutation_effects.data import (
    default_assay_path,
    load_assay,
    load_assay_metadata,
    parse_mutation,
    sha256,
    validate_assay,
)


class DatasetTests(unittest.TestCase):
    def test_mutation_parser(self):
        mutation = parse_mutation("H24C")
        self.assertEqual(
            (mutation.wild_type, mutation.position, mutation.alternate), ("H", 24, "C")
        )
        for invalid in ("H0", "H24H", "H-24C", "B24C", "H24C:A25G"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_mutation(invalid)

    def test_authoritative_assay_integrity(self):
        frame = load_assay(validate=False)
        summary = validate_assay(frame, load_assay_metadata())
        self.assertEqual(summary["variants"], 4996)
        self.assertEqual(summary["unique_positions"], 263)
        self.assertEqual((summary["position_min"], summary["position_max"]), (24, 286))
        self.assertEqual(summary["sequence_length"], 286)
        self.assertEqual(
            sha256(default_assay_path()),
            "82835aeecc2001c742434457102e02b5aaf38673274aaba00ac85e07bcd94f35",
        )


if __name__ == "__main__":
    unittest.main()
