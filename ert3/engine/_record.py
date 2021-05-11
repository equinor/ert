import json
from typing import Optional
from pathlib import Path

import ert3


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


def _get_distribution(
    parameter_group_name: str, parameters_config: ert3.config.ParametersConfig
) -> ert3.stats.Distribution:
    for parameter_group in parameters_config:
        if parameter_group.name == parameter_group_name:
            return parameter_group.as_distribution()

    raise ValueError(f"No parameter group found named: {parameter_group_name}")


# pylint: disable=too-many-arguments
def sample_record(
    workspace: Path,
    parameters_config: ert3.config.ParametersConfig,
    parameter_group_name: str,
    record_name: str,
    ensemble_size: int,
    experiment_name: Optional[str] = None,
) -> None:
    distribution = _get_distribution(parameter_group_name, parameters_config)
    ensrecord = ert3.data.EnsembleRecord(
        records=[distribution.sample() for _ in range(ensemble_size)]
    )
    ert3.storage.add_ensemble_record(
        workspace=workspace,
        record_name=record_name,
        ensemble_record=ensrecord,
        experiment_name=experiment_name,
    )
