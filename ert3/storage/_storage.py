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


class Storage:
    def __init__(self, workspace, experiment_name):
        storage_location = _generate_storage_location(workspace)
        _assert_storage_initialized(storage_location)
        self._workspace = workspace
        self._experiment_name = experiment_name
        self._storage_location = storage_location
        self._input_data = None
        self._output_data = None

        with open(self._storage_location) as f:
            self._storage = yaml.safe_load(f)

        if self._experiment_name not in self._storage:
            raise KeyError(
                f"No experiment named: {self._experiment_name} in {self._storage}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def input_data(self):
        self._input_data = self._get_data("input")
        return self._input_data

    @input_data.setter
    def input_data(self, data):
        self._input_data = self._add_data("input", data)

    @property
    def output_data(self):
        self._output_data = self._get_data("output")
        return self._output_data

    @output_data.setter
    def output_data(self, data):
        self._add_data(
            "output",
            data,
            required_types=["input"],
        )

    def _get_data(self, data_type):
        storage = self._storage

        if data_type not in storage[self._experiment_name]:
            raise KeyError(f"No {data_type} data for experiment: {self._experiment_name}")

        return storage[self._experiment_name][data_type]

    def _add_data(self, data_type, data, required_types=()):
        storage = self._storage

        if self._experiment_name not in storage:
            raise KeyError(
                f"Cannot add {data_type} data to non-existing experiment: {self._experiment_name}"
            )

        if data_type in storage[self._experiment_name]:
            msg = f"{data_type} data is already stored for experiment"
            raise KeyError(msg.capitalize())

        for req in required_types:
            if req not in storage[self._experiment_name]:
                raise KeyError(f"Cannot add {data_type} data to experiment without {req} data")

        storage[self._experiment_name][data_type] = data

        with open(self._storage_location, "w") as f:
            yaml.dump(storage, f)

    @staticmethod
    def init_storage(workspace):
        storage_location = _generate_storage_location(workspace)

        if os.path.exists(storage_location):
            raise ValueError(f"Storage already initialized for workspace {workspace}")

        if not os.path.exists(storage_location.parent):
            os.makedirs(storage_location.parent)

        with open(storage_location, "w") as f:
            yaml.dump({}, f)
