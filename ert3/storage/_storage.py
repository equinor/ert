import codecs
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast, Set
import io

import cloudpickle
import requests
import ert3


_STORAGE_URL = "http://localhost:8000"
_DATA = "__data__"
_PARAMETERS = "__parameters__"
_ENSEMBLE_RECORDS = "__ensemble_records__"
_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)


def _get_experiment_by_name(experiment_name: str) -> Dict[str, Any]:
    response = requests.get(url=f"{_STORAGE_URL}/experiments")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    experiments = {exp["name"]: exp for exp in response.json()}
    return experiments.get(experiment_name, None)


def init(*, workspace: Path) -> None:
    response = requests.get(url=f"{_STORAGE_URL}/experiments")
    experiment_names = {exp["name"]: exp["ensembles"] for exp in response.json()}

    for special_key in _SPECIAL_KEYS:
        if f"{workspace}.{special_key}" in experiment_names:
            raise ValueError("Storage already initialized")
        _init_experiment(
            workspace=workspace,
            experiment_name=f"{workspace}.{special_key}",
            parameters=[],
            ensemble_size=-1,
        )


def init_experiment(
    *,
    workspace: Path,
    experiment_name: str,
    parameters: Iterable[str],
    ensemble_size: int,
) -> None:
    if ensemble_size <= 0:
        raise ValueError("Ensemble cannot have a size <= 0")

    _init_experiment(
        workspace=workspace,
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
    )


def _init_experiment(
    *,
    workspace: Path,
    experiment_name: str,
    parameters: Iterable[str],
    ensemble_size: int,
) -> None:
    if not experiment_name:
        raise ValueError("Cannot initialize experiment without a name")

    if _get_experiment_by_name(experiment_name) is not None:
        raise KeyError(f"Cannot initialize existing experiment: {experiment_name}")

    exp_response = requests.post(
        url=f"{_STORAGE_URL}/experiments", json={"name": experiment_name}
    )
    exp_id = exp_response.json()["id"]
    response = requests.post(
        url=f"{_STORAGE_URL}/experiments/{exp_id}/ensembles",
        json={
            "parameters": list(parameters),
            "size": ensemble_size,
        },
    )
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)


def get_experiment_names(*, workspace: Path) -> Set[str]:
    response = response = requests.get(url=f"{_STORAGE_URL}/experiments")
    experiment_names = {exp["name"] for exp in response.json()}
    for special_key in _SPECIAL_KEYS:
        key = f"{workspace}.{special_key}"
        if key in experiment_names:
            experiment_names.remove(key)
    return experiment_names


def _add_data(
    workspace: Path, experiment_name: str, record_name: str, data: Any
) -> None:

    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise KeyError(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensembles"][0]  # currently just one ens per exp
    response = requests.post(
        url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records/{record_name}/file",
        files={"file": (record_name, io.StringIO(data), "something")},
    )
    if response.status_code == 409:
        raise KeyError("Record already exists")

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)


def _get_data(workspace: Path, experiment_name: str, record_name: str) -> Any:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise KeyError(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    ensemble_id = experiment["ensembles"][0]  # currently just one ens per exp
    response = requests.get(
        url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records/{record_name}"
    )

    if response.status_code == 404:
        raise KeyError(f"No {record_name} data for experiment: {experiment_name}")

    return response.content


def add_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    ensemble_record: ert3.data.EnsembleRecord,
    experiment_name: Optional[str] = None,
) -> None:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"

    # If a Record is serialized to JSON without associating it with a type,
    # information is lost. For instance, serializing an int mapping
    # {0: 1, 1: 3} to JSON, keys are made into strings in JSON. If it is not
    # known that the Record is a int, str map, information is lost.
    #
    # TODO: https://github.com/equinor/ert/issues/1550 should implement better
    # usage of ert-storage, specifically using numerical endpoints when
    # applicable.
    pickle_str = codecs.encode(cloudpickle.dumps(ensemble_record), "base64").decode()
    data = json.dumps({"data": pickle_str})
    _add_data(workspace, experiment_name, record_name, data)


def get_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    experiment_name: Optional[str] = None,
) -> ert3.data.EnsembleRecord:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    data = json.loads(_get_data(workspace, experiment_name, record_name))["data"]
    return cast(
        ert3.data.EnsembleRecord,
        cloudpickle.loads(codecs.decode(data.encode(), "base64")),
    )


def get_ensemble_record_names(
    *, workspace: Path, experiment_name: Optional[str] = None
) -> Iterable[str]:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise KeyError(
            f"Cannot get record names of non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensembles"][0]  # currently just one ens per exp
    response = requests.get(url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/records")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    return list(response.json().keys())


def get_experiment_parameters(
    *, workspace: Path, experiment_name: str
) -> Iterable[str]:

    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise KeyError(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensembles"][0]  # currently just one ens per exp
    response = requests.get(url=f"{_STORAGE_URL}/ensembles/{ensemble_id}/parameters")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    return list(response.json())


def delete_experiment(*, workspace: Path, experiment_name: str) -> None:

    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Experiment does not exist: {experiment_name}"
        )
    response = requests.delete(url=f"{_STORAGE_URL}/experiments/{experiment['id']}")

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
