import io
import logging
import asyncio
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

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

    @property
    def uri(self) -> str:
        if not self.is_transmitted():
            raise RuntimeError(f"Record {self._name} not transmitted")
        return self._uri

    @property
    def record_type(self) -> Optional[ert.data.RecordType]:
        assert self.is_transmitted()
        return self._record_type

    def set_transmitted(self, uri: str, record_type: ert.data.RecordType) -> None:
        self._set_transmitted_state(uri, record_type)

    async def _transmit_numerical_record(self, record: ert.data.NumericalRecord) -> str:
        url = f"{self._uri}/matrix"
        if self._real_id is not None:
            url = f"{url}?realization_index={self._real_id}"
            self._uri = f"{self._uri}?realization_index={self._real_id}"
        await add_record(url, record)
        return self._uri

    async def _transmit_blob_record(self, record: ert.data.BlobRecord) -> str:
        url = f"{self._uri}/file"
        if self._real_id is not None:
            url = f"{url}?realization_index={self._real_id}"
            self._uri = f"{self._uri}?realization_index={self._real_id}"
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


async def get_record_storage_transmitters(
    records_url: str,
    record_name: str,
    record_source: Optional[str] = None,
    ensemble_size: Optional[int] = None,
) -> Dict[int, Dict[str, StorageRecordTransmitter]]:
    if record_source is None:
        record_source = record_name
    uri = f"{records_url}/{record_source}"
    metadata = await get_record_metadata(uri)
    record_type = metadata["record_type"]
    uris = metadata["uris"]
    # We giving an ensemble size we expect the number of uris in the record metadata
    # to match the size of the ensemble or be equal to 1, in the case of an
    # uniform record and, has the same data stored only once for all the realizations
    if ensemble_size is not None and 1 < len(uris) != ensemble_size:
        raise ert.exceptions.ErtError(
            f"Ensemble size {ensemble_size} does not match stored record ensemble "
            + f"size {len(uris)}"
        )

    transmitters = []
    for record_uri in uris:
        transmitter = StorageRecordTransmitter(record_source, records_url)
        # Record data has already been stored, now just setting the transmitter uri and
        # record type
        transmitter.set_transmitted(record_uri, record_type)
        transmitters.append(transmitter)

    if ensemble_size is not None and len(transmitters) == 1:
        return {iens: {record_name: transmitters[0]} for iens in range(ensemble_size)}
    return {
        iens: {record_name: transmitter}
        for iens, transmitter in enumerate(transmitters)
    }


def _get(url: str, headers: Dict[str, Any]) -> requests.Response:
    return requests.get(url, headers=headers)


async def _get_from_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> requests.Response:

    loop = asyncio.get_event_loop()

    # Using sync code because one of the httpx dependencies (anyio) throws an
    # AttributeError: module 'anyio._backends._asyncio' has no attribute 'current_time'
    # Refactor and try to use aiohttp or httpx once the issue above is fixed
    future = loop.run_in_executor(
        None,
        partial(_get, url, headers),
    )
    resp = await future

    if resp.status_code != HTTPStatus.OK:
        logger.error("Failed to fetch from %s. Response: %s", url, resp.text)
        if resp.status_code == HTTPStatus.NOT_FOUND:
            raise ert.exceptions.ElementMissingError(resp.text)
        raise ert.exceptions.StorageError(resp.text)
    return resp


def _post(url: str, headers: Dict[str, Any], **kwargs: Any) -> requests.Response:
    return requests.post(url=url, headers=headers, **kwargs)


async def _post_to_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> requests.Response:

    loop = asyncio.get_event_loop()
    # Using sync code because one of the httpx dependencies (anyio) throws an
    # AttributeError: module 'anyio._backends._asyncio' has no attribute 'current_time'
    # Refactor and try to use aiohttp or httpx once the issue above is fixed
    future = loop.run_in_executor(
        None,
        partial(_post, url, headers, **kwargs),
    )
    resp = await future

    if resp.status_code != HTTPStatus.OK:
        logger.error("Failed to post to %s. Response: %s", url, resp.text)
        if resp.status_code == HTTPStatus.CONFLICT:
            raise ert.exceptions.ElementExistsError(resp.text)
        raise ert.exceptions.StorageError(resp.text)
    return resp


def _put(url: str, headers: Dict[str, Any], **kwargs: Any) -> requests.Response:
    return requests.put(url=url, headers=headers, **kwargs)


async def _put_to_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> requests.Response:
    loop = asyncio.get_event_loop()

    # Using sync code because one of the httpx dependencies (anyio) throws an
    # AttributeError: module 'anyio._backends._asyncio' has no attribute 'current_time'
    # Refactor and try to use aiohttp or httpx once the issue above is fixed
    future = loop.run_in_executor(
        None,
        partial(_put, url, headers, **kwargs),
    )
    resp = await future

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


async def get_record_metadata(record_url: str) -> Dict[Any, Any]:
    headers = {
        "Token": StorageInfo.token(),
    }
    # TODO once storage returns proper record metadata information add proper support
    # for metadata
    url = f"{record_url}/userdata?realization_index=0"
    resp = await _get_from_server_async(url, headers)
    ret: Dict[Any, Any] = resp.json()
    return ret


async def add_record_metadata(
    record_urls: str, record_name: str, metadata: Dict[Any, Any]
) -> None:
    headers = {
        "Token": StorageInfo.token(),
    }
    url = f"{record_urls}/{record_name}/userdata?realization_index=0"
    await _put_to_server_async(url, headers, json=metadata)


async def transmit_record_collection(
    record_coll: ert.data.RecordCollection,
    record_name: str,
    workspace: Path,
    experiment_name: Optional[str] = None,
) -> Dict[int, Dict[str, StorageRecordTransmitter]]:
    assert record_coll.ensemble_size is not None
    record: ert.data.Record
    metadata: Dict[Any, Any] = {
        "record_type": record_coll.record_type,
        "uris": [],
    }

    records_url = await get_records_url_async(workspace, experiment_name)
    if experiment_name is not None:
        ensemble_id = await _get_ensemble_id_async(workspace, experiment_name)
        ensemble_size = await _get_ensemble_size(ensemble_id=ensemble_id)
    else:
        ensemble_size = record_coll.ensemble_size

    # Handle special case when we have a uniform record collection (collection of size
    # one)
    if record_coll.ensemble_size not in (1, ensemble_size):
        raise ert.exceptions.ErtError(
            f"Experiment ensemble size {ensemble_size} does not match"
            f" data size {record_coll.ensemble_size}"
        )

    if record_coll.ensemble_size == 1:
        record = record_coll.records[0]
        transmitter = ert.storage.StorageRecordTransmitter(
            name=record_name, storage_url=records_url, iens=0
        )
        await transmitter.transmit_record(record)
        metadata["uris"].append(transmitter.uri)
        await add_record_metadata(records_url, record_name, metadata)
        return {iens: {record_name: transmitter} for iens in range(ensemble_size)}

    transmitters: Dict[int, Dict[str, StorageRecordTransmitter]] = {}
    transmitter_list = []
    for iens, record in enumerate(record_coll.records):
        transmitter = StorageRecordTransmitter(record_name, records_url, iens=iens)
        await transmitter.transmit_record(record)
        transmitter_list.append(transmitter)
        transmitters[iens] = {record_name: transmitter}

    for transmitter in transmitter_list:
        metadata["uris"].append(transmitter.uri)
    await add_record_metadata(records_url, record_name, metadata)

    return transmitters


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
            f"Experiment {experiment_name} does not exist"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    return f"{storage_url}/ensembles/{ensemble_id}/records"


async def get_records_url_async(
    workspace: Path, experiment_name: Optional[str] = None
) -> str:
    storage_url = StorageInfo.url()
    ensemble_id = await _get_ensemble_id_async(workspace, experiment_name)
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


async def _get_ensemble_id_async(
    workspace: Path, experiment_name: Optional[str] = None
) -> str:
    storage_url = StorageInfo.url()
    url = f"{storage_url}/experiments"
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"

    headers = {"Token": StorageInfo.token()}
    response = await _get_from_server_async(url, headers)
    experiments = {exp["name"]: exp for exp in response.json()}
    experiment = experiments.get(experiment_name, None)
    if experiment is not None:
        return str(experiment["ensemble_ids"][0])
    raise ert.exceptions.NonExistantExperiment(
        f"Experiment {experiment_name} does not exist"
    )


async def _get_ensemble_size(ensemble_id: str) -> int:
    storage_url = StorageInfo.url()
    url = f"{storage_url}/ensembles/{ensemble_id}"
    headers = {"Token": StorageInfo.token()}
    response = await _get_from_server_async(url, headers)
    response_json = response.json()
    return int(response_json["size"])


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
    parameters: Iterable[str],
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


def _init_experiment(
    *,
    experiment_name: str,
    parameters: Iterable[str],
    ensemble_size: int,
    responses: Iterable[str],
) -> None:
    if not experiment_name:
        raise ValueError("Cannot initialize experiment without a name")

    if _get_experiment_by_name(experiment_name) is not None:
        raise ert.exceptions.ElementExistsError(
            f"Cannot initialize existing experiment: {experiment_name}"
        )

    if len(set(parameters).intersection(responses)) > 0:
        raise ert.exceptions.StorageError(
            "Experiment parameters and responses cannot have a name in common"
        )

    exp_response = _post_to_server(path="experiments", json={"name": experiment_name})
    exp_id = exp_response.json()["id"]

    response = _post_to_server(
        f"experiments/{exp_id}/ensembles",
        json={
            "parameter_names": list(parameters),
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


def _get_experiment_parameters(experiment_name: str) -> Iterable[str]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/parameters")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    return list(response.json())


def get_ensemble_record(
    *,
    workspace: Path,
    record_name: str,
    experiment_name: Optional[str] = None,
    source: Optional[str] = None,
    ensemble_size: Optional[int] = None,
) -> ert.data.RecordCollection:
    records_url = ert.storage.get_records_url(
        workspace=workspace, experiment_name=experiment_name
    )

    transmitters = asyncio.get_event_loop().run_until_complete(
        ert.storage.get_record_storage_transmitters(
            records_url=records_url,
            record_name=record_name,
            record_source=source,
            ensemble_size=ensemble_size,
        )
    )
    records = []
    for transmitter_map in transmitters.values():
        for transmitter in transmitter_map.values():
            records.append(
                asyncio.get_event_loop().run_until_complete(transmitter.load())
            )

    return ert.data.RecordCollection(records=records)


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
    return _get_experiment_parameters(experiment_name)


def get_experiment_responses(*, experiment_name: str) -> Iterable[str]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get responses from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
    return list(response.json()["response_names"])


def delete_experiment(*, experiment_name: str) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Experiment does not exist: {experiment_name}"
        )
    response = _delete_on_server(path=f"experiments/{experiment['id']}")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
