import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    List,
    Union,
)
import io
import logging
import pandas as pd
from pydantic import BaseModel
import requests
import ert3

from ert_shared.storage.connection import get_info

logger = logging.getLogger(__name__)

_STORAGE_TOKEN = None
_STORAGE_URL = None
_ENSEMBLE_RECORDS = "__ensemble_records__"
_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)

# Character used as separator for parameter record names. This is used as a
# workaround for webviz-ert, which expects each parameter record to have exactly
# one value per realisation.
_PARAMETER_RECORD_SEPARATOR = "."


class _NumericalMetaData(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    ensemble_size: int
    record_type: ert3.data.RecordType


def _assert_server_info() -> None:
    global _STORAGE_URL, _STORAGE_TOKEN  # pylint: disable=global-statement

    if _STORAGE_URL is None:
        info = get_info()
        _STORAGE_URL = info["baseurl"]
        _STORAGE_TOKEN = info["auth"][1]


def _get_from_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> requests.Response:

    _assert_server_info()
    if not headers:
        headers = {}
    headers["Token"] = _STORAGE_TOKEN

    resp = requests.get(url=f"{_STORAGE_URL}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to fetch from %s. Response: %s", path, resp.text)

    return resp


def _delete_on_server(
    path: str, headers: Optional[Dict[Any, Any]] = None, status_code: int = 200
) -> requests.Response:

    _assert_server_info()
    if not headers:
        headers = {}
    headers["Token"] = _STORAGE_TOKEN
    resp = requests.delete(
        url=f"{_STORAGE_URL}/{path}",
        headers=headers,
    )
    if resp.status_code != status_code:
        logger.error("Failed to delete %s. Response: %s", path, resp.text)

    return resp


def _post_to_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> requests.Response:

    _assert_server_info()
    if not headers:
        headers = {}
    headers["Token"] = _STORAGE_TOKEN
    resp = requests.post(url=f"{_STORAGE_URL}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to post to %s. Response: %s", path, resp.text)

    return resp


def _put_to_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> requests.Response:

    _assert_server_info()
    if not headers:
        headers = {}
    headers["Token"] = _STORAGE_TOKEN
    resp = requests.put(url=f"{_STORAGE_URL}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to put to %s. Response: %s", path, resp.text)

    return resp


def _get_experiment_by_name(experiment_name: str) -> Dict[str, Any]:
    response = _get_from_server(path="experiments")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    experiments = {exp["name"]: exp for exp in response.json()}
    return experiments.get(experiment_name, None)


def init(*, workspace: Path) -> None:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"]: exp["ensemble_ids"] for exp in response.json()}

    for special_key in _SPECIAL_KEYS:
        if f"{workspace}.{special_key}" in experiment_names:
            raise ValueError("Storage already initialized")
        _init_experiment(
            workspace=workspace,
            experiment_name=f"{workspace}.{special_key}",
            parameters={},
            ensemble_size=-1,
            responses=[],
        )


def init_experiment(
    *,
    workspace: Path,
    experiment_name: str,
    parameters: Mapping[str, Iterable[str]],
    ensemble_size: int,
    responses: Iterable[str],
) -> None:
    if ensemble_size <= 0:
        raise ValueError("Ensemble cannot have a size <= 0")

    _init_experiment(
        workspace=workspace,
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
        responses=responses,
    )


def _init_experiment(
    *,
    workspace: Path,
    experiment_name: str,
    parameters: Mapping[str, Iterable[str]],
    ensemble_size: int,
    responses: Iterable[str],
) -> None:
    if not experiment_name:
        raise ValueError("Cannot initialize experiment without a name")

    if _get_experiment_by_name(experiment_name) is not None:
        raise ert3.exceptions.ElementExistsError(
            f"Cannot initialize existing experiment: {experiment_name}"
        )

    if len(set(parameters.keys()).intersection(responses)) > 0:
        raise ert3.exceptions.StorageError(
            "Experiment parameters and responses cannot have a name in common"
        )

    exp_response = _post_to_server(path="experiments", json={"name": experiment_name})
    exp_id = exp_response.json()["id"]
    response = _post_to_server(
        f"experiments/{exp_id}/ensembles",
        json={
            "parameter_names": [
                f"{record}.{param}"
                for record, params in parameters.items()
                for param in params
            ],
            "response_names": list(responses),
            "size": ensemble_size,
            "userdata": {"name": experiment_name},
        },
    )
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)


def get_experiment_names(*, workspace: Path) -> Set[str]:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"] for exp in response.json()}
    for special_key in _SPECIAL_KEYS:
        key = f"{workspace}.{special_key}"
        if key in experiment_names:
            experiment_names.remove(key)
    return experiment_names


def _get_record_type(ensemble_record: ert3.data.EnsembleRecord) -> ert3.data.RecordType:
    record_type = ensemble_record.records[0].record_type
    for record in ensemble_record.records:
        if record.record_type != record_type:
            raise ValueError("Inconsistent record type")

    return record_type


def _add_numerical_data(
    workspace: Path,
    experiment_name: str,
    record_name: str,
    record_data: Union[pd.DataFrame, pd.Series],
    record_type: ert3.data.RecordType,
) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    metadata = _NumericalMetaData(
        ensemble_size=len(record_data),
        record_type=record_type,
    )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    record_url = f"ensembles/{ensemble_id}/records/{record_name}"

    response = _post_to_server(
        f"{record_url}/matrix",
        data=record_data.to_csv().encode(),
        headers={"content-type": "text/csv"},
    )

    if response.status_code == 409:
        raise ert3.exceptions.ElementExistsError("Record already exists")

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)

    meta_response = _put_to_server(f"{record_url}/userdata", json=metadata.dict())

    if meta_response.status_code != 200:
        raise ert3.exceptions.StorageError(meta_response.text)


def _response2records(
    response_content: bytes, record_type: ert3.data.RecordType
) -> ert3.data.EnsembleRecord:
    dataframe = pd.read_csv(
        io.BytesIO(response_content), index_col=0, float_precision="round_trip"
    )

    records: List[ert3.data.Record]
    if record_type == ert3.data.RecordType.LIST_FLOAT:
        records = [
            ert3.data.Record(data=row.to_list()) for _, row in dataframe.iterrows()
        ]
    elif record_type == ert3.data.RecordType.MAPPING_INT_FLOAT:
        records = [
            ert3.data.Record(data={int(k): v for k, v in row.to_dict().items()})
            for _, row in dataframe.iterrows()  # pylint: disable=no-member
        ]
    elif record_type == ert3.data.RecordType.MAPPING_STR_FLOAT:
        records = [
            ert3.data.Record(data=row.to_dict())
            for _, row in dataframe.iterrows()  # pylint: disable=no-member
        ]
    else:
        raise ValueError(
            f"Unexpected record type when loading numerical record: {record_type}"
        )
    return ert3.data.EnsembleRecord(records=records)


def _combine_records(
    ensemble_records: List[ert3.data.EnsembleRecord],
) -> ert3.data.EnsembleRecord:
    # Combine records into the first ensemble record
    combined_records: List[ert3.data.Record] = []
    for record_idx, _ in enumerate(ensemble_records[0].records):
        record0 = ensemble_records[0].records[record_idx]

        if isinstance(record0.data, list):
            ldata = [
                val
                for data in (
                    ensemble_record.records[record_idx].data
                    for ensemble_record in ensemble_records
                )
                if isinstance(data, list)
                for val in data
            ]
            combined_records.append(ert3.data.Record(data=ldata))
        elif isinstance(record0.data, dict):
            ddata = {
                key: val
                for data in (
                    ensemble_record.records[record_idx].data
                    for ensemble_record in ensemble_records
                )
                if isinstance(data, dict)
                for key, val in data.items()
            }
            combined_records.append(ert3.data.Record(data=ddata))
    return ert3.data.EnsembleRecord(records=combined_records)


def _get_numerical_metadata(ensemble_id: str, record_name: str) -> _NumericalMetaData:
    response = _get_from_server(
        f"ensembles/{ensemble_id}/records/{record_name}/userdata"
    )

    if response.status_code == 404:
        raise ert3.exceptions.ElementMissingError(
            f"No metadata for {record_name} in ensemble: {ensemble_id}"
        )

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)

    return _NumericalMetaData(**json.loads(response.content))


def _get_numerical_data(
    workspace: Path, experiment_name: str, record_name: str
) -> ert3.data.EnsembleRecord:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    metadata = _get_numerical_metadata(ensemble_id, record_name)

    response = _get_from_server(
        f"ensembles/{ensemble_id}/records/{record_name}",
        headers={"accept": "text/csv"},
    )

    if response.status_code == 404:
        raise ert3.exceptions.ElementMissingError(
            f"No {record_name} data for experiment: {experiment_name}"
        )

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)

    return _response2records(
        response.content,
        metadata.record_type,
    )


def _get_experiment_parameters(
    workspace: Path, experiment_name: str
) -> Mapping[str, Iterable[str]]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/parameters")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    parameters: MutableMapping[str, List[str]] = {}
    for name in response.json():
        key, val = name.split(".")
        if key in parameters:
            parameters[key].append(val)
        else:
            parameters[key] = [val]
    return parameters


def add_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    ensemble_record: ert3.data.EnsembleRecord,
    experiment_name: Optional[str] = None,
) -> None:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    dataframe = pd.DataFrame([r.data for r in ensemble_record.records])
    record_type = _get_record_type(ensemble_record)

    parameters = _get_experiment_parameters(workspace, experiment_name)
    if record_name in parameters:
        # Split by columns
        for column_label in dataframe:
            _add_numerical_data(
                workspace,
                experiment_name,
                f"{record_name}.{column_label}",
                dataframe[column_label],
                record_type,
            )
    else:
        _add_numerical_data(
            workspace,
            experiment_name,
            record_name,
            dataframe,
            record_type,
        )


def get_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    experiment_name: Optional[str] = None,
) -> ert3.data.EnsembleRecord:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    param_names = _get_experiment_parameters(workspace, experiment_name)
    if record_name in param_names:
        ensemble_records = [
            _get_numerical_data(
                workspace,
                experiment_name,
                record_name + _PARAMETER_RECORD_SEPARATOR + param_name,
            )
            for param_name in param_names[record_name]
        ]
        return _combine_records(ensemble_records)
    else:
        return _get_numerical_data(workspace, experiment_name, record_name)


def get_ensemble_record_names(
    *, workspace: Path, experiment_name: Optional[str] = None, _flatten: bool = True
) -> Iterable[str]:
    # _flatten is a parameter used only for testing separated parameter records
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot get record names of non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(path=f"ensembles/{ensemble_id}/records")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)

    # Flatten any parameter records that were split
    if _flatten:
        return {x.split(_PARAMETER_RECORD_SEPARATOR)[0] for x in response.json().keys()}
    return list(response.json().keys())


def get_experiment_parameters(
    *, workspace: Path, experiment_name: str
) -> Iterable[str]:
    return list(_get_experiment_parameters(workspace, experiment_name))


def get_experiment_responses(*, workspace: Path, experiment_name: str) -> Iterable[str]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Cannot get responses from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/responses")
    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
    return list(response.json())


def delete_experiment(*, workspace: Path, experiment_name: str) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert3.exceptions.NonExistantExperiment(
            f"Experiment does not exist: {experiment_name}"
        )
    response = _delete_on_server(path=f"experiments/{experiment['id']}")

    if response.status_code != 200:
        raise ert3.exceptions.StorageError(response.text)
