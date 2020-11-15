import ert3

import yaml
import pathlib
import os


_STORAGE_FILE = "storage.yaml"


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

    return storage.keys()


def add_input_data(workspace, experiment_name, input_data):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot add input data to non-existing experiment: {experiment_name}"
        )

    if "input" in storage[experiment_name]:
        raise KeyError(
            f"Input data is already stored for experiment: {experiment_name}"
        )

    storage[experiment_name]["input"] = input_data
    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_input_data(workspace, experiment_name):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(f"Cannot get input data, no experiment named: {experiment_name}")

    if "input" not in storage[experiment_name]:
        raise KeyError(f"No input data for experiment: {experiment_name}")

    return storage[experiment_name]["input"]


def add_output_data(workspace, experiment_name, output_data):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot add output data to non-existing experiment: {experiment_name}"
        )

    if "output" in storage[experiment_name]:
        raise KeyError(
            f"Output data is already stored for experiment: {experiment_name}"
        )

    if "input" not in storage[experiment_name]:
        raise KeyError(f"Cannot add output data to experiment without input data")

    storage[experiment_name]["output"] = output_data
    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_output_data(workspace, experiment_name):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get output data, no experiment named: {experiment_name}"
        )

    if "output" not in storage[experiment_name]:
        raise KeyError(f"No output data for experiment: {experiment_name}")

    return storage[experiment_name]["output"]
