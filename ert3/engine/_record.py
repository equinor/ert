from pathlib import Path
import asyncio
import ert
import ert3


def load_record(
    workspace: Path,
    record_name: str,
    record_file: Path,
    record_mime: str,
) -> None:

    collection = ert.data.load_collection_from_file(record_file, record_mime)
    future = ert.storage.transmit_record_collection(collection, record_name, workspace)
    asyncio.get_event_loop().run_until_complete(future)


def sample_record(
    parameters_config: ert3.config.ParametersConfig,
    parameter_group_name: str,
    ensemble_size: int,
) -> ert.data.RecordCollection:
    distribution = parameters_config[parameter_group_name].as_distribution()
    return ert.data.RecordCollection(
        records=[distribution.sample() for _ in range(ensemble_size)]
    )
