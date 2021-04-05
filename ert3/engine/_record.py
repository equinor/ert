import json
from typing import Optional
from pathlib import Path

import ert3
from ert3.engine import _utils


def load_record(workspace: Path, record_name: str, record_file: Path) -> None:
    with open(record_file, "r") as f:
        raw_ensrecord = json.load(f)

    ensrecord = ert3.data.EnsembleRecord(
        records=[ert3.data.Record(data=raw_record) for raw_record in raw_ensrecord]
    )
    ert3.storage.add_ensemble_record(
        workspace=workspace,
        record_name=record_name,
        ensemble_record=ensrecord,
    )


def sample_record(
    workspace: Path,
    parameter_group_name: str,
    record_name: str,
    ensemble_size: int,
    experiment_name: Optional[str] = None,
) -> None:
    parameters = _utils.load_parameters(workspace)

    if parameter_group_name not in parameters:
        raise ValueError(f"No parameter group found named: {parameter_group_name}")
    distribution = parameters[parameter_group_name]

    ensrecord = ert3.data.EnsembleRecord(
        records=[distribution.sample() for _ in range(ensemble_size)]
    )
    ert3.storage.add_ensemble_record(
        workspace=workspace,
        record_name=record_name,
        ensemble_record=ensrecord,
        experiment_name=experiment_name,
    )
