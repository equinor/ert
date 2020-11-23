import ert3

import yaml
import pathlib
import os


_STORAGE_FILE = "storage.yaml"


def _generate_storage_location(workspace_root):
    workspace_root = pathlib.Path(workspace_root)
    return workspace_root / ert3._WORKSPACE_DATA_ROOT / _STORAGE_FILE


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


def init_experiment(experiment):
    storage_location = _generate_storage_location(experiment.workspace_root)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name in storage:
        raise KeyError(f"Cannot initialize existing experiment: {experiment_name}")

    storage[experiment_name] = {}

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_experiment_names(workspace):
    storage_location = _generate_storage_location(workspace.root)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    return storage.keys()


def _add_data(experiment, data_type, data, required_types=()):
    storage_location = _generate_storage_location(experiment.workspace_root)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment.name not in storage:
        raise KeyError(
            f"Cannot add {data_type} data to non-existing experiment: {experiment.name}"
        )

    if data_type in storage[experiment.name]:
        msg = f"{data_type} data is already stored for experiment"
        raise KeyError(msg.capitalize())

    for req in required_types:
        if req not in storage[experiment_name]:
            raise KeyError(f"Cannot add {data_type} data to experiment without {req} data")

    storage[experiment.name][data_type] = data

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def add_input_data(experiment, input_data):
    _add_data(experiment, "input", input_data)


def add_output_data(experiment, output_data):
    _add_data(
        experiment,
        "output",
        output_data,
        required_types=["input"],
    )


def _get_data(experiment, data_type):
    storage_location = _generate_storage_location(experiment.workspace_root)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment.name not in storage:
        raise KeyError(
            f"Cannot get {data_type} data, no experiment named: {experiment.name}"
        )

    if data_type not in storage[experiment.name]:
        raise KeyError(f"No {data_type} data for experiment: {experiment.name}")

    return storage[experiment.name][data_type]


def get_input_data(experiment):
    return _get_data(experiment, "input")


def get_output_data(experiment):
    return _get_data(experiment, "output")


def experiment_have_run(experiment):
    try:
        output = get_output_data(experiment)
    except ValueError as err:
        if "No output data for experiment" in err.msg:
            return False
        else:
            raise err

    return True

