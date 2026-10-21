"""Download and validate only the five authoritative Phase 3A pilot tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from esm2_mutation_effects.data import default_reference_path, project_root, sha256
from esm2_mutation_effects.phase3_data import validate_pilot_assay
from esm2_mutation_effects.phase3_pilot import (
    PILOT_TASKS,
    PROTEINGYM_LICENSE,
    PROTEINGYM_RECORD_URL,
    PROTEINGYM_SUBSTITUTIONS_MD5,
    PROTEINGYM_SUBSTITUTIONS_URL,
    PROTEINGYM_VERSION,
    PilotDataRequirements,
)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in archive.namelist() if name == filename or name.endswith(f"/{filename}")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {filename!r} in the ProteinGym archive; found {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=default_reference_path())
    parser.add_argument(
        "--raw-dir", type=Path, default=root / "data" / "phase3" / "raw"
    )
    parser.add_argument(
        "--provenance-dir", type=Path, default=root / "data" / "phase3" / "provenance"
    )
    parser.add_argument(
        "--archive-cache", type=Path, default=root / "data" / "phase3" / "cache"
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata does not exist: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    metadata = metadata.set_index("DMS_id", verify_integrity=True)
    task_ids = [task.dms_id for task in PILOT_TASKS]
    missing = sorted(set(task_ids).difference(metadata.index))
    if missing:
        raise ValueError(f"Pilot tasks are missing from metadata: {missing}")

    args.archive_cache.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive_cache / "DMS_ProteinGym_substitutions.zip"
    if args.force_download or not archive_path.exists():
        with tempfile.NamedTemporaryFile(dir=args.archive_cache, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            urllib.request.urlretrieve(PROTEINGYM_SUBSTITUTIONS_URL, temporary_path)
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    observed_md5 = md5(archive_path)
    if observed_md5 != PROTEINGYM_SUBSTITUTIONS_MD5:
        raise ValueError(
            "ProteinGym archive MD5 does not match the authoritative Zenodo record: "
            f"expected {PROTEINGYM_SUBSTITUTIONS_MD5}, observed {observed_md5}"
        )

    requirements = PilotDataRequirements()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for task in PILOT_TASKS:
            row = metadata.loc[task.dms_id]
            filename = str(row["DMS_filename"])
            destination = args.raw_dir / filename
            member = archive_member(archive, filename)
            if not destination.exists():
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            frame = pd.read_csv(destination)
            summary = validate_pilot_assay(
                frame,
                {"DMS_id": task.dms_id, **row.to_dict()},
                requirements=requirements,
            )
            records.append(
                {
                    "role": task.role,
                    "DMS_id": task.dms_id,
                    "DMS_filename": filename,
                    "file_sha256": sha256(destination),
                    "metadata_raw_directionality": float(row["raw_DMS_directionality"]),
                    **summary,
                }
            )

    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    selected_metadata = metadata.loc[task_ids].reset_index()
    selected_metadata.to_csv(args.provenance_dir / "pilot_task_metadata.csv", index=False)
    manifest = {
        "phase": "phase3a_one_day_pilot",
        "status": "raw_data_validated_not_a_training_result",
        "source": {
            "record": PROTEINGYM_RECORD_URL,
            "version": PROTEINGYM_VERSION,
            "license": PROTEINGYM_LICENSE,
            "substitutions_archive_url": PROTEINGYM_SUBSTITUTIONS_URL,
            "substitutions_archive_md5": observed_md5,
            "metadata_path": str(args.metadata.resolve()),
            "metadata_sha256": sha256(args.metadata),
        },
        "requirements": requirements.serializable(),
        "tasks": records,
    }
    destination = args.provenance_dir / "pilot_data_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
