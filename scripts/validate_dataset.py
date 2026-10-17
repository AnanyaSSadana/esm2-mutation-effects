#!/usr/bin/env python3
"""Validate the downloaded assay against ProteinGym metadata."""

import json

from esm2_mutation_effects.data import (
    default_assay_path,
    load_assay,
    load_assay_metadata,
    sha256,
    validate_assay,
)


def main() -> None:
    frame = load_assay(validate=False)
    summary = validate_assay(frame, load_assay_metadata())
    summary["sha256"] = sha256(default_assay_path())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
