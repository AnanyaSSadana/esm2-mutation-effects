"""End-to-end, resumable command for the Phase 3B-lite replication matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys

from esm2_mutation_effects.data import project_root


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--microbatch-size", type=int, default=16)
    args = parser.parse_args()
    if args.microbatch_size != 16:
        parser.error("Phase 3B-lite is locked to --microbatch-size 16")
    root = project_root()
    commands = [
        [sys.executable, str(root / "scripts" / "fetch_phase3a_data.py")],
        [
            sys.executable,
            str(root / "scripts" / "score_phase3a_zero_shot.py"),
            "--device",
            args.device,
            "--positions-per-batch",
            str(args.microbatch_size),
        ],
        [sys.executable, str(root / "scripts" / "lock_phase3b_lite_protocol.py")],
        [
            sys.executable,
            str(root / "scripts" / "run_phase3b_lite_matrix.py"),
            "--device",
            args.device,
            "--microbatch-size",
            str(args.microbatch_size),
        ],
        [sys.executable, str(root / "scripts" / "analyze_phase3b_lite.py")],
    ]
    for command in commands:
        run(command)


if __name__ == "__main__":
    main()
