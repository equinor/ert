import ert3

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union
import yaml
import logging

logger = logging.getLogger()


_STORAGE_FILE = "storage.yaml"
try:
    _STORAGE_URL = os.environ["ERT_STORAGE_URL"]
except KeyError:
    logger.warning(
        "ERT_STORAGE_URL is not set, assuming ert-storage running "
        "on localhost: http://127.0.0.1:8000"
    )
    _STORAGE_URL = "http://127.0.0.1:8000"

_DATA = "__data__"
_PARAMETERS = "__parameters__"

_ENSEMBLE_RECORDS = "__ensemble_records__"

_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)


def _generate_storage_location(workspace: Union[str, Path]) -> Path:
    workspace = Path(workspace)
    return workspace / ert3._WORKSPACE_DATA_ROOT / _STORAGE_FILE


def _assert_storage_initialized(storage_location: Path) -> None:
    if not os.path.isfile(storage_location):
        raise ValueError("Storage is not initialized")


def init(*, workspace: Union[str, Path]) -> None:
    storage_location = _generate_storage_location(workspace)

    if os.path.exists(storage_location):
        raise ValueError(f"Storage already initialized for workspace {workspace}")

    if not os.path.exists(storage_location.parent):
        os.makedirs(storage_location.parent)

    with open(storage_location, "w") as f:
        yaml.dump({}, f)

    for special_key in _SPECIAL_KEYS:
        init_experiment(workspace=workspace, experiment_name=special_key, parameters=[])


def init_experiment(
    *, workspace: Union[str, Path], experiment_name: str, parameters: Iterable[str]
) -> None:
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name in storage:
        raise KeyError(f"Cannot initialize existing experiment: {experiment_name}")

    storage[experiment_name] = {_PARAMETERS: list(parameters), _DATA: {}}

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def get_experiment_names(*, workspace: Union[str, Path]) -> Iterable[str]:
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    experiment_names = set(storage.keys())
    for special_key in _SPECIAL_KEYS:
        experiment_names.remove(special_key)
    return experiment_names


def _add_data(
    workspace: Union[str, Path], experiment_name: str, data_type: str, data: Any
) -> None:
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot add {data_type} data to non-existing experiment: {experiment_name}"
        )

    experiment_data = storage[experiment_name][_DATA]

    if data_type in experiment_data:
        msg = f"{data_type} data is already stored for experiment"
        raise KeyError(msg.capitalize())

    experiment_data[data_type] = data

    with open(storage_location, "w") as f:
        yaml.dump(storage, f)


def _get_data(workspace: Union[str, Path], experiment_name: str, data_type: str) -> Any:
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get {data_type} data, no experiment named: {experiment_name}"
        )

    if data_type not in storage[experiment_name][_DATA]:
        raise KeyError(f"No {data_type} data for experiment: {experiment_name}")

    return storage[experiment_name][_DATA][data_type]


def add_ensemble_record(
    *,
    workspace: Union[str, Path],
    record_name: str,
    ensemble_record: ert3.data.EnsembleRecord,
    experiment_name: Optional[str] = None,
) -> None:
    if experiment_name is None:
        experiment_name = _ENSEMBLE_RECORDS
    _add_data(workspace, experiment_name, record_name, ensemble_record.json())


def get_ensemble_record(
    *,
    workspace: Union[str, Path],
    record_name: str,
    experiment_name: Optional[str] = None,
) -> ert3.data.EnsembleRecord:
    if experiment_name is None:
        experiment_name = _ENSEMBLE_RECORDS
    return ert3.data.EnsembleRecord.parse_raw(
        _get_data(workspace, experiment_name, record_name)
    )


def get_ensemble_record_names(
    *, workspace: Union[str, Path], experiment_name: Optional[str] = None
) -> Iterable[str]:
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

    return list(str(key) for key in storage[experiment_name][_DATA].keys())


def get_experiment_parameters(
    *, workspace: Union[str, Path], experiment_name: str
) -> Iterable[str]:
    storage_location = _generate_storage_location(workspace)
    _assert_storage_initialized(storage_location)

    with open(storage_location) as f:
        storage = yaml.safe_load(f)

    if experiment_name not in storage:
        raise KeyError(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    return list(str(pname) for pname in storage[experiment_name][_PARAMETERS])
