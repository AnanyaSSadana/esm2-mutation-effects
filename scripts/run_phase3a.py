"""End-to-end command for the bounded Phase 3A pilot."""
from __future__ import annotations

import argparse
import subprocess
import sys

from esm2_mutation_effects.data import project_root


def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--device", default="auto"); p.add_argument("--microbatch-size", type=int, default=16); a=p.parse_args(); root=project_root()
 for script, extra in (("fetch_phase3a_data.py", []), ("score_phase3a_zero_shot.py", ["--device",a.device,"--positions-per-batch",str(a.microbatch_size)]), ("make_phase3a_target_manifest.py", []), ("run_phase3a_pilot.py", ["--device",a.device,"--microbatch-size",str(a.microbatch_size)])):
  subprocess.run([sys.executable, str(root / "scripts" / script), *extra], cwd=root, check=True)
if __name__ == "__main__": main()
