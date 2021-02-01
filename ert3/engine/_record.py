import ert3
from ert3.engine import _utils

import json
import yaml


def load_record(workspace, record_name, record_file):
    record = json.load(record_file)
    record_file.close()

    ert3.storage.add_variables(
        workspace,
        record_name,
        record,
    )


def sample_record(workspace, parameter_group_name, record_name, ensemble_size):
    parameters = _utils.load_parameters(workspace)

    if parameter_group_name not in parameters:
        raise ValueError(f"No parameter group found named: {parameter_group_name}")
    distribution = parameters[parameter_group_name]

    ert3.storage.add_variables(
        workspace,
        record_name,
        [distribution.sample() for _ in range(ensemble_size)],
    )
