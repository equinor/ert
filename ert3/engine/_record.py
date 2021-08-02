from typing import Optional
from pathlib import Path

import ert
import ert3


def load_record(
    workspace: Path,
    record_name: str,
    record_file: Path,
    blob_record: bool = False,
) -> None:

    record_coll = ert.data.load_collection_from_file(record_file, blob_record)

    ert.storage.add_ensemble_record(
        workspace=workspace,
        record_name=record_name,
        ensemble_record=record_coll,
    )


# pylint: disable=too-many-arguments
def sample_record(
    workspace: Path,
    parameters_config: ert3.config.ParametersConfig,
    parameter_group_name: str,
    record_name: str,
    ensemble_size: int,
    experiment_name: Optional[str] = None,
) -> None:
    distribution = parameters_config[parameter_group_name].as_distribution()
    ensrecord = ert.data.RecordCollection(
        records=[distribution.sample() for _ in range(ensemble_size)]
    )
    ert.storage.add_ensemble_record(
        workspace=workspace,
        record_name=record_name,
        ensemble_record=ensrecord,
        experiment_name=experiment_name,
    )
