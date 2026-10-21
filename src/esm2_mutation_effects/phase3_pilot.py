"""Locked identifiers and data rules for the bounded Phase 3A pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PROTEINGYM_SUBSTITUTIONS_URL = (
    "https://zenodo.org/records/15293562/files/DMS_ProteinGym_substitutions.zip?download=1"
)
PROTEINGYM_SUBSTITUTIONS_MD5 = "ca1a4d46941ef33cc972245347118c7d"
PROTEINGYM_RECORD_URL = "https://doi.org/10.5281/zenodo.15293562"
PROTEINGYM_VERSION = "1.3"
PROTEINGYM_LICENSE = "MIT"


@dataclass(frozen=True)
class PilotTask:
    dms_id: str
    role: str


META_TRAIN_TASKS = (
    PilotTask("AMIE_PSEAE_Wrenbeck_2017", "meta_train"),
    PilotTask("CALM1_HUMAN_Weile_2017", "meta_train"),
    PilotTask("CCDB_ECOLI_Tripathi_2016", "meta_train"),
    PilotTask("DYR_ECOLI_Nguyen_2023", "meta_train"),
)
DIAGNOSTIC_TASK = PilotTask("CASP3_HUMAN_Roychowdhury_2020", "diagnostic_only")
PILOT_TASKS = META_TRAIN_TASKS + (DIAGNOSTIC_TASK,)
PILOT_TASK_IDS = tuple(task.dms_id for task in PILOT_TASKS)


@dataclass(frozen=True)
class PilotDataRequirements:
    """Minimum raw-table criteria required before a pilot task can be used."""

    minimum_valid_single_substitutions: int = 500
    minimum_mutated_positions: int = 30
    list_size: int = 5
    maximum_sequence_length: int = 1022

    def validate(self) -> None:
        if self.minimum_valid_single_substitutions < 1:
            raise ValueError("minimum_valid_single_substitutions must be positive")
        if self.minimum_mutated_positions < 2:
            raise ValueError("minimum_mutated_positions must be at least two")
        if self.list_size < 2:
            raise ValueError("list_size must be at least two")
        if self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")

    def serializable(self) -> dict[str, int]:
        return asdict(self)
