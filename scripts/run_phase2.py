"""Complete Phase 2: smoke check, protocol lock, and label-efficiency matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from esm2_mutation_effects.data import project_root


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-microbatch-size", type=int, default=4)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=project_root() / "results" / "phase2_execution_summary.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.list_microbatch_size <= 16:
        parser.error("--list-microbatch-size must be between 1 and 16")

    root = project_root()
    smoke = [
        sys.executable,
        str(root / "scripts" / "run_phase2_smoke.py"),
        "--device",
        args.device,
        "--list-microbatch-size",
        str(args.list_microbatch_size),
    ]
    lock = [
        sys.executable,
        str(root / "scripts" / "lock_phase2_protocol.py"),
        "--list-microbatch-size",
        str(args.list_microbatch_size),
    ]
    confirmation = [
        sys.executable,
        str(root / "scripts" / "run_phase2_label_efficiency.py"),
        "--device",
        args.device,
        "--list-microbatch-size",
        str(args.list_microbatch_size),
    ]
    if args.dry_run:
        print(json.dumps({"smoke": smoke, "lock": lock, "confirmation": confirmation}, indent=2))
        return

    print("=== Phase 2: validation-only 20-label smoke check ===", flush=True)
    run_command(smoke)
    lock_path = root / "results" / "phase2_label_efficiency" / "protocol_lock.json"
    if not lock_path.exists():
        print("\n=== Phase 2: protocol lock ===", flush=True)
        run_command(lock)
    else:
        print(f"\n=== Phase 2: reusing protocol lock {lock_path} ===", flush=True)
    print("\n=== Phase 2: confirmatory label-efficiency matrix ===", flush=True)
    run_command(confirmation)

    summary_path = root / "results" / "phase2_label_efficiency" / "phase2_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("matrix_complete") is not True:
        raise RuntimeError("Phase 2 confirmation matrix did not complete")
    payload = {
        "phase": "phase_2_label_efficiency",
        "status": "complete",
        "protocol_lock": str(lock_path.resolve()),
        "summary": str(summary_path.resolve()),
        "test_evaluated": True,
    }
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.log_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.log_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
