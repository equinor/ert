from typing import TYPE_CHECKING

import ert.data
import ert.storage

if TYPE_CHECKING:
    from ert.ert3.config import ParametersConfig
    from ert.ert3.workspace import Workspace


async def load_record(
    workspace: "Workspace",
    record_name: str,
    transformation: ert.data.RecordTransformation,
) -> None:
    collection = await ert.data.load_collection_from_file(
        transformation=transformation,
    )
    await ert.storage.transmit_record_collection(
        collection, record_name, workspace.name
    )


def sample_record(
    parameters_config: "ParametersConfig",
    parameter_name: str,
    ensemble_size: int,
) -> ert.data.RecordCollection:
    distribution = parameters_config[parameter_name].as_distribution()
    return ert.data.RecordCollection(
        records=tuple(distribution.sample() for _ in range(ensemble_size))
    )
