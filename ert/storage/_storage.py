import io
import json
import logging
from collections import defaultdict
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

import httpx
import pandas as pd
import requests
from pydantic import BaseModel

import ert
from ert_shared.storage.connection import get_info

logger = logging.getLogger(__name__)
read_csv = partial(pd.read_csv, index_col=0, float_precision="round_trip")
DictStrAny = Dict[str, Any]

_ENSEMBLE_RECORDS = "__ensemble_records__"
_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)

# Character used as separator for parameter record names. This is used as a
# workaround for webviz-ert, which expects each parameter record to have exactly
# one value per realisation.
_PARAMETER_RECORD_SEPARATOR = "."
_OCTET_STREAM = "application/octet-stream"
_CSV = "text/csv"


class _NumericalMetaData(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    ensemble_size: int
    record_type: ert.data.RecordType


class StorageInfo:
    _url: Optional[str] = None
    _token: Optional[str] = None

    @classmethod
    def _set_info(cls) -> None:
        info = get_info()
        cls._url = info["baseurl"]
        cls._token = info["auth"][1]

    @classmethod
    def url(cls) -> str:
        if StorageInfo._url is None:
            cls._set_info()
        return str(StorageInfo._url)

    @classmethod
    def token(cls) -> str:
        if StorageInfo._token is None:
            cls._set_info()
        return str(StorageInfo._token)


class StorageRecordTransmitter(ert.data.RecordTransmitter):
    def __init__(self, name: str, storage_url: str, iens: Optional[int] = None):
        super().__init__(ert.data.RecordTransmitterType.ert_storage)
        self._name: str = name
        self._uri = f"{storage_url}/{name}"
        self._real_id: Optional[int] = iens
        if self._real_id is not None:
            self._uri = f"{self._uri}?realization_index={self._real_id}"

    async def _transmit_numerical_record(self, record: ert.data.NumericalRecord) -> str:
        url = f"{self._uri}/matrix"
        if self._real_id is not None:
            url = f"{url}?realization_index={self._real_id}"
        await add_record(url, record)
        return self._uri

    async def _transmit_blob_record(self, record: ert.data.BlobRecord) -> str:
        url = f"{self._uri}/file"
        if self._real_id is not None:
            url = f"{url}?realization_index={self._real_id}"
        await add_record(url, record)
        return self._uri

    async def _load_numerical_record(self) -> ert.data.NumericalRecord:
        assert self._record_type
        record = await load_record(self._uri, self._record_type)
        return ert.data.NumericalRecord(data=record.data)

    async def _load_blob_record(self) -> ert.data.BlobRecord:
        assert self._record_type
        record = await load_record(self._uri, self._record_type)
        return ert.data.BlobRecord(data=record.data)


async def _get_from_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> httpx.Response:
    async with httpx.AsyncClient() as session:
        resp = await session.get(url=url, headers=headers, timeout=None, **kwargs)

    if resp.status_code != HTTPStatus.OK:
        logger.error("Failed to fetch from %s. Response: %s", url, resp.text)
        raise ert.exceptions.StorageError(resp.text)

    return resp


async def _post_to_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> httpx.Response:
    async with httpx.AsyncClient() as session:
        resp = await session.post(url=url, headers=headers, **kwargs)

    if resp.status_code != HTTPStatus.OK:
        logger.error("Failed to post to %s. Response: %s", url, resp.text)
        if resp.status_code == HTTPStatus.CONFLICT:
            raise ert.exceptions.ElementExistsError(resp.text)
        raise ert.exceptions.StorageError(resp.text)

    return resp


def _set_content_header(
    header: str, record_type: ert.data.RecordType, headers: Optional[DictStrAny] = None
) -> DictStrAny:
    content_type = _OCTET_STREAM if record_type == ert.data.RecordType.BYTES else _CSV
    if headers is None:
        return {header: content_type}
    headers_ = headers.copy()
    headers_[header] = content_type
    return headers_


async def add_record(url: str, record: ert.data.Record) -> None:
    headers = {
        "Token": StorageInfo.token(),
    }

    assert record.record_type
    if record.record_type != ert.data.RecordType.BYTES:
        headers = _set_content_header(
            header="content-type", record_type=record.record_type, headers=headers
        )
        data = pd.DataFrame([record.data]).to_csv().encode()
        await _post_to_server_async(url=url, headers=headers, data=data)
    else:
        assert isinstance(record.data, bytes)
        data = {"file": io.BytesIO(record.data)}
        await _post_to_server_async(url=url, headers=headers, files=data)


def _interpret_series(row: pd.Series, record_type: ert.data.RecordType) -> Any:
    if record_type not in {item.value for item in ert.data.RecordType}:
        raise ValueError(
            f"Unexpected record type when loading numerical record: {record_type}"
        )

    if record_type == ert.data.RecordType.MAPPING_INT_FLOAT:
        return {int(k): v for k, v in row.to_dict().items()}
    return (
        row.to_list()
        if record_type == ert.data.RecordType.LIST_FLOAT
        else row.to_dict()
    )


def _response_to_record_collection(
    content: bytes, metadata: _NumericalMetaData
) -> ert.data.RecordCollection:
    record_type = metadata.record_type
    records: Iterable[ert.data.Record]
    if record_type == ert.data.RecordType.BYTES:
        records = (
            ert.data.BlobRecord(data=content) for _ in range(metadata.ensemble_size)
        )
    else:
        records = (
            ert.data.NumericalRecord(
                data=_interpret_series(row=row, record_type=metadata.record_type)
            )
            for _, row in read_csv(io.BytesIO(content)).iterrows()
        )
    return ert.data.RecordCollection(records=tuple(records))


async def load_record(url: str, record_type: ert.data.RecordType) -> ert.data.Record:
    headers = {
        "Token": StorageInfo.token(),
    }
    headers = _set_content_header(
        header="accept", headers=headers, record_type=record_type
    )
    response = await _get_from_server_async(url=url, headers=headers)
    content = response.content
    if record_type != ert.data.RecordType.BYTES:
        dataframe: pd.DataFrame = read_csv(io.BytesIO(content))
        for _, row in dataframe.iterrows():  # pylint: disable=no-member
            return ert.data.NumericalRecord(
                data=_interpret_series(row=row, record_type=record_type)
            )
    return ert.data.BlobRecord(data=content)


def _get_from_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> requests.Response:

    if not headers:
        headers = {}
    headers["Token"] = StorageInfo.token()

    resp = requests.get(url=f"{StorageInfo.url()}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to fetch from %s. Response: %s", path, resp.text)

    return resp


def get_records_url(workspace: Path, experiment_name: Optional[str] = None) -> str:
    storage_url = StorageInfo.url()
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    return f"{storage_url}/ensembles/{ensemble_id}/records"


def _delete_on_server(
    path: str, headers: Optional[Dict[Any, Any]] = None, status_code: int = 200
) -> requests.Response:

    if not headers:
        headers = {}
    headers["Token"] = StorageInfo.token()
    resp = requests.delete(
        url=f"{StorageInfo.url()}/{path}",
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

    if not headers:
        headers = {}
    headers["Token"] = StorageInfo.token()
    resp = requests.post(url=f"{StorageInfo.url()}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to post to %s. Response: %s", path, resp.text)

    return resp


def _put_to_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> requests.Response:

    if not headers:
        headers = {}
    headers["Token"] = StorageInfo.token()
    resp = requests.put(url=f"{StorageInfo.url()}/{path}", headers=headers, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to put to %s. Response: %s", path, resp.text)

    return resp


def _get_experiment_by_name(experiment_name: str) -> Dict[str, Any]:
    response = _get_from_server(path="experiments")
    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
    experiments = {exp["name"]: exp for exp in response.json()}
    return experiments.get(experiment_name, None)


def init(*, workspace: Path) -> None:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"]: exp["ensemble_ids"] for exp in response.json()}

    for special_key in _SPECIAL_KEYS:
        if f"{workspace}.{special_key}" in experiment_names:
            raise ValueError("Storage already initialized")
        _init_experiment(
            experiment_name=f"{workspace}.{special_key}",
            parameters={},
            ensemble_size=-1,
            responses=[],
        )


def init_experiment(
    *,
    experiment_name: str,
    parameters: Mapping[str, Iterable[str]],
    ensemble_size: int,
    responses: Iterable[str],
) -> None:
    if ensemble_size <= 0:
        raise ValueError("Ensemble cannot have a size <= 0")

    _init_experiment(
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
        responses=responses,
    )


def _is_numeric_parameter(params: Iterable[str]) -> bool:
    return len(list(params)) > 0


def _init_experiment(
    *,
    experiment_name: str,
    parameters: Mapping[str, Iterable[str]],
    ensemble_size: int,
    responses: Iterable[str],
) -> None:
    if not experiment_name:
        raise ValueError("Cannot initialize experiment without a name")

    if _get_experiment_by_name(experiment_name) is not None:
        raise ert.exceptions.ElementExistsError(
            f"Cannot initialize existing experiment: {experiment_name}"
        )

    if len(set(parameters.keys()).intersection(responses)) > 0:
        raise ert.exceptions.StorageError(
            "Experiment parameters and responses cannot have a name in common"
        )

    exp_response = _post_to_server(path="experiments", json={"name": experiment_name})
    exp_id = exp_response.json()["id"]

    parameter_names = []
    for record, params in parameters.items():
        if _is_numeric_parameter(params):
            for param in params:
                parameter_names.append(f"{record}.{param}")
        else:
            parameter_names.append(record)

    response = _post_to_server(
        f"experiments/{exp_id}/ensembles",
        json={
            "parameter_names": parameter_names,
            "response_names": list(responses),
            "size": ensemble_size,
            "userdata": {"name": experiment_name},
        },
    )
    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)


def get_experiment_names(*, workspace: Path) -> Set[str]:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"] for exp in response.json()}
    for special_key in _SPECIAL_KEYS:
        key = f"{workspace}.{special_key}"
        if key in experiment_names:
            experiment_names.remove(key)
    return experiment_names


def _add_numerical_data(
    experiment_name: str,
    record_name: str,
    record_data: Union[pd.DataFrame, pd.Series],
    record_type: Optional[ert.data.RecordType],
) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
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
        headers={"content-type": _CSV},
    )

    if response.status_code == 409:
        raise ert.exceptions.ElementExistsError("Record already exists")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    meta_response = _put_to_server(f"{record_url}/userdata", json=metadata.dict())

    if meta_response.status_code != 200:
        raise ert.exceptions.StorageError(meta_response.text)


def _combine_records(
    ensemble_records: List[ert.data.RecordCollection],
) -> ert.data.RecordCollection:

    if len(ensemble_records) == 1:
        # Nothing to combine only one record collection here
        return ensemble_records[0]

    # Combine records into the first ensemble record
    combined_records: List[ert.data.Record] = []
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
            combined_records.append(ert.data.NumericalRecord(data=ldata))
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
            combined_records.append(ert.data.NumericalRecord(data=ddata))
    return ert.data.RecordCollection(records=combined_records)


def _get_numerical_metadata(ensemble_id: str, record_name: str) -> _NumericalMetaData:
    response = _get_from_server(
        f"ensembles/{ensemble_id}/records/{record_name}/userdata"
    )

    if response.status_code == 404:
        raise ert.exceptions.ElementMissingError(
            f"No metadata for {record_name} in ensemble: {ensemble_id}"
        )

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    return _NumericalMetaData(**json.loads(response.content))


def _get_data(
    experiment_name: str,
    record_name: str,
) -> ert.data.RecordCollection:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    metadata = _get_numerical_metadata(ensemble_id, record_name)

    response = _get_from_server(
        path=f"ensembles/{ensemble_id}/records/{record_name}",
        headers=_set_content_header(header="accept", record_type=metadata.record_type),
    )

    if response.status_code == 404:
        raise ert.exceptions.ElementMissingError(
            f"No {record_name} data for experiment: {experiment_name}"
        )

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    return _response_to_record_collection(
        content=response.content,
        metadata=metadata,
    )


def _is_numeric_parameter_response(name: str) -> bool:
    return "." in name


def _get_experiment_parameters(experiment_name: str) -> Mapping[str, Iterable[str]]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/parameters")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    parameters = defaultdict(list)
    for name in response.json():
        if _is_numeric_parameter_response(name):
            key, val = name.split(".")
            parameters[key].append(val)
        else:
            parameters[name] = []

    return parameters


def add_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    ensemble_record: ert.data.RecordCollection,
    experiment_name: Optional[str] = None,
) -> None:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    dataframe = pd.DataFrame([r.data for r in ensemble_record.records])

    if ensemble_record.record_type == ert.data.RecordType.BYTES:
        _add_blob_data(experiment_name, record_name, ensemble_record)
    else:
        parameters = _get_experiment_parameters(experiment_name)
        if record_name in parameters:
            # Split by columns
            for column_label in dataframe:
                _add_numerical_data(
                    experiment_name,
                    f"{record_name}.{column_label}",
                    dataframe[column_label],
                    ensemble_record.record_type,
                )
        else:
            _add_numerical_data(
                experiment_name,
                record_name,
                dataframe,
                ensemble_record.record_type,
            )


def _add_blob_data(
    experiment_name: str,
    record_name: str,
    ensemble_record: ert.data.RecordCollection,
) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot add {record_name} data to "
            f"non-existing experiment: {experiment_name}"
        )

    metadata = _NumericalMetaData(
        ensemble_size=ensemble_record.ensemble_size,
        record_type=ert.data.RecordType.BYTES,
    )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    record_url = f"ensembles/{ensemble_id}/records/{record_name}"

    assert ensemble_record
    assert ensemble_record.ensemble_size != 0
    # If the RecordCollection has more than one record we assume
    # all records are the same and store only one record.
    # We store the original size in the metadata
    record = ensemble_record.records[0]

    assert isinstance(record.data, bytes)
    response = _post_to_server(
        f"{record_url}/file",
        files={
            "file": (
                record_name,
                io.BytesIO(record.data),
                _OCTET_STREAM,
            )
        },
    )

    if response.status_code == 409:
        raise ert.exceptions.ElementExistsError("Record already exists")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    meta_response = _put_to_server(f"{record_url}/userdata", json=metadata.dict())

    if meta_response.status_code != 200:
        raise ert.exceptions.StorageError(meta_response.text)


def get_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    experiment_name: Optional[str] = None,
) -> ert.data.RecordCollection:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )
    param_names = _get_experiment_parameters(experiment_name)
    if record_name not in param_names or not param_names[record_name]:
        return _get_data(
            experiment_name=experiment_name,
            record_name=record_name,
        )

    ensemble_records = [
        _get_data(
            experiment_name=experiment_name,
            record_name=record_name + _PARAMETER_RECORD_SEPARATOR + param_name,
        )
        for param_name in param_names[record_name]
    ]
    return _combine_records(ensemble_records)


def get_ensemble_record_names(
    *, workspace: Path, experiment_name: Optional[str] = None, _flatten: bool = True
) -> Iterable[str]:
    # _flatten is a parameter used only for testing separated parameter records
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get record names of non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(path=f"ensembles/{ensemble_id}/records")
    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    # Flatten any parameter records that were split
    if _flatten:
        return {x.split(_PARAMETER_RECORD_SEPARATOR)[0] for x in response.json().keys()}
    return list(response.json().keys())


def get_experiment_parameters(*, experiment_name: str) -> Iterable[str]:
    return list(_get_experiment_parameters(experiment_name))


def get_experiment_responses(*, experiment_name: str) -> Iterable[str]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get responses from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/responses")
    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
    return list(response.json())


def delete_experiment(*, experiment_name: str) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Experiment does not exist: {experiment_name}"
        )
    response = _delete_on_server(path=f"experiments/{experiment['id']}")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
