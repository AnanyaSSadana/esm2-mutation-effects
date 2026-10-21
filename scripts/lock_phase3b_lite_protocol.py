"""Create or verify all Phase 3B-lite manifests before opening new test labels."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import load_assay, project_root
from esm2_mutation_effects.phase3_protocol import manifest_sha256
from esm2_mutation_effects.phase3b_lite import make_protocol, validate_protocol


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "phase3b_lite")
    args = parser.parse_args()

    lock, manifests = make_protocol(load_assay(), git_commit=git_commit(root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_dir / "manifests"
    lock_path = args.output_dir / "protocol_lock.json"
    if lock_path.exists():
        saved = json.loads(lock_path.read_text())
        validate_protocol(saved, git_commit=git_commit(root))
        if saved != lock:
            raise ValueError("Existing Phase 3B-lite lock differs from the requested protocol")
    else:
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for seed, manifest in manifests.items():
        path = manifest_dir / f"split_{seed}.csv"
        if path.exists():
            existing_hash = manifest_sha256(pd.read_csv(path))
            if existing_hash != manifest_sha256(manifest):
                raise ValueError(f"Existing manifest differs from pre-registered split {seed}")
        else:
            manifest.to_csv(path, index=False)

    print(
        json.dumps(
            {
                "protocol_lock": str(lock_path.resolve()),
                "protocol_sha256": lock["protocol_sha256"],
                "new_split_seeds": lock["new_split_seeds"],
                "target_label_budget": lock["target_label_budget"],
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
