import ert3

import yaml
import pathlib
import os


_STORAGE_FILE = "storage.yaml"
_VARIABLES = "__variables__"


def _generate_storage_location(workspace):
    workspace = pathlib.Path(workspace)
    return workspace / ert3._WORKSPACE_DATA_ROOT / _STORAGE_FILE


def _assert_storage_initialized(storage_location):
    if not os.path.isfile(storage_location):
        raise ValueError("Storage is not initialized")


def init(workspace):
    storage_location = _generate_storage_location(workspace)

    if os.path.exists(storage_location):
        raise ValueError(f"Storage already initialized for workspace {workspace}")

    if not os.path.exists(storage_location.parent):
        os.makedirs(storage_location.parent)

    with open(storage_location, "w") as f:
        yaml.dump({}, f)

    init_experiment(workspace, _VARIABLES)


def init_experiment(workspace, experiment_name):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name in storage:
        raise KeyError(f"Cannot initialize existing experiment: {experiment_name}")

    storage[experiment_name] = {}

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_experiment_names(workspace):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    experiment_names = set(storage.keys())
    experiment_names.remove(_VARIABLES)
    return experiment_names


def _add_data(workspace, experiment_name, data_type, data, required_types=()):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot add {data_type} data to non-existing experiment: {experiment_name}"
        )

    if data_type in storage[experiment_name]:
        msg = f"{data_type} data is already stored for experiment"
        raise KeyError(msg.capitalize())

    for req in required_types:
        if req not in storage[experiment_name]:
            raise KeyError(
                f"Cannot add {data_type} data to experiment without {req} data"
            )

    storage[experiment_name][data_type] = data

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def add_input_data(workspace, experiment_name, input_data):
    _add_data(workspace, experiment_name, "input", input_data)


def add_output_data(workspace, experiment_name, output_data):
    _add_data(
        workspace,
        experiment_name,
        "output",
        output_data,
        required_types=["input"],
    )


def _get_data(workspace, experiment_name, data_type):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get {data_type} data, no experiment named: {experiment_name}"
        )

    if data_type not in storage[experiment_name]:
        raise KeyError(f"No {data_type} data for experiment: {experiment_name}")

    return storage[experiment_name][data_type]


def get_input_data(workspace, experiment_name):
    return _get_data(workspace, experiment_name, "input")


def get_output_data(workspace, experiment_name):
    return _get_data(workspace, experiment_name, "output")


def add_variables(workspace, var_name, data):
    _add_data(workspace, _VARIABLES, var_name, data)


def get_variables(workspace, var_name):
    return _get_data(workspace, _VARIABLES, var_name)
