import ert3

import yaml
import pathlib
import os
import requests
import io

from typing import List

_STORAGE_FILE = "storage.yaml"
_STORAGE_URL = "http://localhost:8000"


_DATA = "__data__"
_PARAMETERS = "__parameters__"

_ENSEMBLE_RECORDS = "__ensemble_records__"

_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)


def _generate_storage_location(workspace):
    workspace = pathlib.Path(workspace)
    return workspace / ert3._WORKSPACE_DATA_ROOT / _STORAGE_FILE


def _assert_storage_initialized(storage_location):
    if not os.path.isfile(storage_location):
        raise ValueError("Storage is not initialized")


def init(*, workspace):
    storage_location = _generate_storage_location(workspace)

    if os.path.exists(storage_location):
        raise ValueError(f"Storage already initialized for workspace {workspace}")

    if not os.path.exists(storage_location.parent):
        os.makedirs(storage_location.parent)

    with open(storage_location, "w") as f:
        yaml.dump({}, f)

    for special_key in _SPECIAL_KEYS:
        init_experiment(workspace=workspace, experiment_name=special_key, parameters=[])


def init_experiment(*, workspace, experiment_name, parameters):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name in storage:
        raise KeyError(f"Cannot initialize existing experiment: {experiment_name}")

    response = requests.post(url=f"{_STORAGE_URL}/ensembles")

    storage[experiment_name] = {
        _PARAMETERS: list(parameters),
        _DATA: {},
        "id": response.json()["id"],
    }

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_experiment_names(*, workspace):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    experiment_names = set(storage.keys())
    for special_key in _SPECIAL_KEYS:
        experiment_names.remove(special_key)
    return experiment_names


def _add_data(workspace, experiment_name, record_name, data, record_class="Normal"):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    ensemble_id = storage[experiment_name]["id"]

    response = requests.post(
        url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records/{record_name}/file",
        params={"record_class": record_class},
        files={"file": (record_name, io.StringIO(data), "something")},
    )
    assert response.status_code == 200


def _get_data(workspace: str, experiment_name: str, record_name: str):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    ensemble_id = storage[experiment_name]["id"]

    response = requests.get(
        url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records/{record_name}"
    )

    if response.status_code == 404:
        raise KeyError(f"No {record_name} data for experiment: {experiment_name}")

    return response.content


def add_ensemble_record(
    *, workspace, record_name, ensemble_record, experiment_name=None, record_class=None
):
    if experiment_name is None:
        experiment_name = _ENSEMBLE_RECORDS
    _add_data(
        workspace, experiment_name, record_name, ensemble_record.json(), record_class
    )


def get_ensemble_record(*, workspace, record_name, experiment_name=None):
    if experiment_name is None:
        experiment_name = _ENSEMBLE_RECORDS
    return ert3.data.EnsembleRecord.parse_raw(
        _get_data(workspace, experiment_name, record_name)
    )


def get_ensemble_record_names(
    *, workspace: str, experiment_name: str = None
) -> List[str]:
    if experiment_name is None:
        experiment_name = _ENSEMBLE_RECORDS
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get record names of non-existing experiment: {experiment_name}"
        )
    ensemble_id = storage[experiment_name]["id"]
    response = requests.get(url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records")
    return response.json()


def get_experiment_parameters(*, workspace, experiment_name):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )
    ensemble_id = storage[experiment_name]["id"]
    response = requests.get(url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/input_records")

    return {
        name: ert3.data.EnsembleRecord.parse_raw(rec["data"])
        for name, rec in response.json().items()
    }


def get_ensemble_records(*, workspace, experiment_name):
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )
    ensemble_id = storage[experiment_name]["id"]
    response = requests.get(
        url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/output_records"
    )

    return {
        name: ert3.data.EnsembleRecord.parse_raw(rec["data"])
        for name, rec in response.json().items()
    }
