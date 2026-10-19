#!/usr/bin/env python3
"""Resume and complete the locked Phase 1 LoRA experiment end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Complete Phase 1: validation-only development, protocol lock, and the "
            "five-split confirmatory matrix."
        )
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=project_root() / "results" / "phase1_execution_summary.json",
        help="Written only after all Phase 1 stages finish",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=project_root(), check=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    development_summary = root / "results" / "lora_development" / "development_summary.json"
    protocol_lock = root / "results" / "lora_development" / "protocol_lock.json"
    confirmatory_summary = root / "results" / "lora_confirmatory" / "confirmatory_summary.json"

    print("=== Phase 1: validation-only development ===", flush=True)
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "run_lora_development.py"),
            "--device",
            args.device,
        ],
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        development = json.loads(development_summary.read_text())
        if development.get("matrix_complete") is not True:
            raise RuntimeError("Development runner exited without completing its 12-run matrix")
        if development.get("test_evaluated") is not False:
            raise RuntimeError("Development unexpectedly evaluated test positions")

    print("=== Phase 1: protocol lock ===", flush=True)
    if protocol_lock.exists():
        print(f"Existing protocol lock retained: {protocol_lock}", flush=True)
    else:
        run_command(
            [sys.executable, str(root / "scripts" / "lock_lora_protocol.py")],
            dry_run=args.dry_run,
        )

    print("=== Phase 1: confirmatory matrix ===", flush=True)
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "run_lora_confirmatory.py"),
            "--device",
            args.device,
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    confirmatory = json.loads(confirmatory_summary.read_text())
    if confirmatory.get("matrix_complete") is not True:
        raise RuntimeError("Confirmatory runner exited without completing its 60-run matrix")
    if confirmatory.get("test_evaluated") is not True:
        raise RuntimeError("Confirmatory runner did not evaluate the locked test matrix")

    output = {
        "phase": "phase_1_controlled_lora_experiment",
        "status": "complete",
        "development_summary": str(development_summary.resolve()),
        "protocol_lock": str(protocol_lock.resolve()),
        "confirmatory_summary": str(confirmatory_summary.resolve()),
        "test_evaluated": True,
    }
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.log_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.log_path)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
