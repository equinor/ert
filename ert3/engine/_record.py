import ert
import ert3


async def load_record(
    workspace: ert3.workspace.Workspace,
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
    parameters_config: ert3.config.ParametersConfig,
    parameter_group_name: str,
    ensemble_size: int,
) -> ert.data.RecordCollection:
    distribution = parameters_config[parameter_group_name].as_distribution()
    return ert.data.RecordCollection(
        records=tuple(distribution.sample() for _ in range(ensemble_size))
    )
