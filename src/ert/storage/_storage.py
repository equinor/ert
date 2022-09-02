import io
import json
import logging
from functools import partial
from http import HTTPStatus
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Set, Tuple, Union

import httpx
import pandas as pd

import ert.data
import ert.exceptions
from ert.async_utils import get_event_loop
from ert.services import Storage

logger = logging.getLogger(__name__)
read_csv = partial(pd.read_csv, index_col=0, float_precision="round_trip")

_ENSEMBLE_RECORDS = "__ensemble_records__"
_SPECIAL_KEYS = (_ENSEMBLE_RECORDS,)

# Character used as separator for parameter record names. This is used as a
# workaround for webviz-ert, which expects each parameter record to have exactly
# one value per realisation.
_PARAMETER_RECORD_SEPARATOR = "."
_OCTET_STREAM = "application/octet-stream"
_CSV = "text/csv"


class StorageRecordTransmitter(ert.data.RecordTransmitter):
    def __init__(self, name: str, storage_url: str, iens: Optional[int] = None):
        super().__init__(ert.data.RecordTransmitterType.ert_storage)
        self._name: str = name
        self._uri = f"{storage_url}/{name}"
        self._real_id: Optional[int] = iens

    async def _get_recordtree_transmitters(
        self,
        trans_records: Dict[str, str],
        record_type: ert.data.RecordType,
        path: Optional[str] = None,
    ) -> Dict[str, ert.data.RecordTransmitter]:
        _storage_url = self._uri[: self._uri.rfind("/")]
        transmitters: Dict[str, ert.data.RecordTransmitter] = {}
        for record_path, record_uri in trans_records.items():
            if path is None or path in record_path:
                record_name = record_path.split("/")[-1]
                transmitter = StorageRecordTransmitter(record_name, _storage_url)
                transmitter.set_transmitted(record_uri, record_type)
                transmitters[record_path] = transmitter
        return transmitters

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

    async def _transmit_recordtree(
        self, record: Union[ert.data.NumericalRecordTree, ert.data.BlobRecordTree]
    ) -> str:
        data: Dict[str, str] = {}
        storage_url = self._uri[: self._uri.rfind("/")]
        for record_path in record.flat_record_dict:
            record_name = record_path.split("/")[-1]
            transmitter = StorageRecordTransmitter(
                record_name, storage_url, iens=self._real_id
            )
            await transmitter.transmit_record(record.flat_record_dict[record_path])
            data[record_path] = transmitter._uri
        await self._transmit_blob_record(
            ert.data.BlobRecord(data=json.dumps(data).encode("utf-8"))
        )
        # Since metadata is stored only for record with real_id == 0, within async
        # processing we make sure that only the same realization write the metadata
        if self._real_id == 0:
            await add_record_metadata(
                storage_url, self._name, {"record_type": record.record_type}
            )
        return self._uri

    async def _load_numerical_record(self) -> ert.data.NumericalRecord:
        assert self._record_type
        record = await load_record(self._uri, self._record_type)
        if not isinstance(record, ert.data.NumericalRecord):
            raise TypeError(f"unexpected blobrecord for numerical {self._uri}")
        return record

    async def _load_blob_record(self) -> ert.data.BlobRecord:
        assert self._record_type
        record = await load_record(self._uri, self._record_type)
        if not isinstance(record, ert.data.BlobRecord):
            raise TypeError(f"unexpected numerical record for blob {self._uri}")
        return record


async def _get_record_storage_transmitters(
    records_url: str,
    record_name: str,
    record_source: Optional[str] = None,
    ensemble_size: Optional[int] = None,
) -> Tuple[List[StorageRecordTransmitter], ert.data.RecordCollectionType]:
    if record_source is None:
        record_source = record_name
    uri = f"{records_url}/{record_source}"
    metadata = await get_record_metadata(uri)
    record_type = metadata["record_type"]
    collection_type: ert.data.RecordCollectionType = metadata["collection_type"]
    uris = metadata["uris"]
    # We expect the number of uris in the record metadata to match the size of
    # the ensemble or be equal to 1, in the case of an uniform record
    if ensemble_size is not None:
        if collection_type == ert.data.RecordCollectionType.UNIFORM and len(uris) != 1:
            raise ert.exceptions.ErtError(
                "Ensemble is uniform but stores multiple records"
            )
        if (
            collection_type != ert.data.RecordCollectionType.UNIFORM
            and len(uris) != ensemble_size
        ):
            raise ert.exceptions.ErtError(
                f"Ensemble size {ensemble_size} does not match stored record ensemble "
                + f"for {record_name} of size {len(uris)}"
            )

    transmitters = []
    for record_uri in uris:
        transmitter = StorageRecordTransmitter(record_source, records_url)
        # Record data has already been stored, now just setting the transmitter uri and
        # record type
        transmitter.set_transmitted(record_uri, record_type)
        transmitters.append(transmitter)

    return transmitters, collection_type


async def get_record_storage_transmitters(
    records_url: str,
    record_name: str,
    record_source: Optional[str] = None,
    ensemble_size: Optional[int] = None,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    transmitters, collection_type = await _get_record_storage_transmitters(
        records_url, record_name, record_source, ensemble_size
    )
    if (
        ensemble_size is not None
        and collection_type == ert.data.RecordCollectionType.UNIFORM
    ):
        return {iens: {record_name: transmitters[0]} for iens in range(ensemble_size)}
    return {
        iens: {record_name: transmitter}
        for iens, transmitter in enumerate(transmitters)
    }


def _get(url: str, headers: Dict[str, Any]) -> httpx.Response:
    with Storage.session() as session:
        return session.get(url, headers=headers, timeout=60)


async def _get_from_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> httpx.Response:

    loop = get_event_loop()

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


def _post(url: str, headers: Dict[str, Any], **kwargs: Any) -> httpx.Response:
    with Storage.session() as session:
        return session.post(url=url, headers=headers, timeout=60, **kwargs)


async def _post_to_server_async(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> httpx.Response:
    if headers is None:
        headers = {}

    loop = get_event_loop()
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


def _put(url: str, headers: Dict[str, Any], **kwargs: Any) -> httpx.Response:
    with Storage.session() as session:
        return session.put(url=url, headers=headers, timeout=60, **kwargs)


async def _put_to_server_async(
    url: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> httpx.Response:
    loop = get_event_loop()

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
    header: str,
    record_type: ert.data.RecordType,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    content_type = _OCTET_STREAM if record_type == ert.data.RecordType.BYTES else _CSV
    if headers is None:
        return {header: content_type}
    headers_ = headers.copy()
    headers_[header] = content_type
    return headers_


async def add_record(url: str, record: ert.data.Record) -> None:
    assert record.record_type
    if record.record_type != ert.data.RecordType.BYTES:
        headers = _set_content_header(
            header="content-type", record_type=record.record_type
        )
        data = pd.DataFrame([record.data]).to_csv().encode()
        await _post_to_server_async(url=url, headers=headers, data=data)
    else:
        assert isinstance(record.data, bytes)
        data = {"file": io.BytesIO(record.data)}
        await _post_to_server_async(url=url, files=data)


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


async def load_record(url: str, record_type: ert.data.RecordType) -> ert.data.Record:
    if record_type in (
        ert.data.RecordType.BLOB_TREE,
        ert.data.RecordType.NUMERICAL_TREE,
    ):
        headers = _set_content_header(
            header="accept", record_type=ert.data.RecordType.BYTES
        )
    else:
        headers = _set_content_header(header="accept", record_type=record_type)
    response = await _get_from_server_async(url=url, headers=headers)
    content = response.content
    if record_type in (
        ert.data.RecordType.LIST_FLOAT,
        ert.data.RecordType.MAPPING_INT_FLOAT,
        ert.data.RecordType.MAPPING_STR_FLOAT,
        ert.data.RecordType.SCALAR_FLOAT,
    ):
        dataframe: pd.DataFrame = read_csv(io.BytesIO(content))
        if record_type == ert.data.RecordType.SCALAR_FLOAT:
            return ert.data.NumericalRecord(data=float(dataframe.iloc[0, 0]))
        for _, row in dataframe.iterrows():
            return ert.data.NumericalRecord(
                data=_interpret_series(row=row, record_type=record_type)
            )
    return ert.data.BlobRecord(data=content)


async def get_record_metadata(record_url: str) -> Dict[Any, Any]:
    # TODO once storage returns proper record metadata information add proper support
    # for metadata
    url = f"{record_url}/userdata?realization_index=0"
    resp = await _get_from_server_async(url, {})
    ret: Dict[Any, Any] = resp.json()
    return ret


async def add_record_metadata(
    record_urls: str, record_name: str, metadata: Dict[Any, Any]
) -> None:
    url = f"{record_urls}/{record_name}/userdata?realization_index=0"
    await _put_to_server_async(url, {}, json=metadata)


async def transmit_awaitable_record_collection(
    record_awaitable: Awaitable[ert.data.RecordCollection],
    record_name: str,
    workspace_name: str,
    experiment_name: Optional[str] = None,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    record_coll = await record_awaitable
    return await transmit_record_collection(
        record_coll, record_name, workspace_name, experiment_name
    )


async def transmit_record_collection(
    record_coll: ert.data.RecordCollection,
    record_name: str,
    workspace_name: str,
    experiment_name: Optional[str] = None,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    record: ert.data.Record
    metadata: Dict[Any, Any] = {
        "record_type": record_coll.record_type,
        "collection_type": record_coll.collection_type,
        "uris": [],
    }

    records_url = await get_records_url_async(workspace_name, experiment_name)
    if experiment_name is not None:
        ensemble_id = await _get_ensemble_id_async(workspace_name, experiment_name)
        ensemble_size = await _get_ensemble_size(ensemble_id=ensemble_id)
    else:
        ensemble_size = len(record_coll)

    if len(record_coll) != ensemble_size:
        raise ert.exceptions.ErtError(
            f"Experiment ensemble size {ensemble_size} does not match"
            f" data size {len(record_coll)}"
        )

    # Handle special case of a uniform record collection
    if record_coll.collection_type == ert.data.RecordCollectionType.UNIFORM:
        record = record_coll.records[0]
        transmitter = StorageRecordTransmitter(
            name=record_name, storage_url=records_url, iens=0
        )
        await transmitter.transmit_record(record)
        metadata["uris"].append(transmitter.uri)
        await add_record_metadata(records_url, record_name, metadata)
        return {iens: {record_name: transmitter} for iens in range(ensemble_size)}

    transmitters: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {}
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
) -> httpx.Response:
    if not headers:
        headers = {}

    with Storage.session() as session:
        resp = session.get(path, headers=headers, timeout=60, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to fetch from %s. Response: %s", path, resp.text)

    return resp


def get_records_url(workspace_name: str, experiment_name: Optional[str] = None) -> str:
    if experiment_name is None:
        experiment_name = f"{workspace_name}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistentExperiment(
            f"Experiment {experiment_name} does not exist"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    return f"/ensembles/{ensemble_id}/records"


async def get_records_url_async(
    workspace_name: str, experiment_name: Optional[str] = None
) -> str:
    ensemble_id = await _get_ensemble_id_async(workspace_name, experiment_name)
    return f"/ensembles/{ensemble_id}/records"


def _delete_on_server(
    path: str, headers: Optional[Dict[Any, Any]] = None, status_code: int = 200
) -> httpx.Response:
    if not headers:
        headers = {}
    with Storage.session() as session:
        resp = session.delete(
            path,
            headers=headers,
            timeout=60,
        )
    if resp.status_code != status_code:
        logger.error("Failed to delete %s. Response: %s", path, resp.text)

    return resp


def _post_to_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> httpx.Response:
    if not headers:
        headers = {}
    with Storage.session() as session:
        resp = session.post(path, headers=headers, timeout=60, **kwargs)
    if resp.status_code != status_code:
        logger.error("Failed to post to %s. Response: %s", path, resp.text)

    return resp


def _put_to_server(
    path: str,
    headers: Optional[Dict[Any, Any]] = None,
    status_code: int = 200,
    **kwargs: Any,
) -> httpx.Response:
    if not headers:
        headers = {}
    with Storage.session() as session:
        resp = session.put(path, headers=headers, timeout=60, **kwargs)
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
    workspace_name: str, experiment_name: Optional[str] = None
) -> str:
    url = "/experiments"
    if experiment_name is None:
        experiment_name = f"{workspace_name}.{_ENSEMBLE_RECORDS}"

    response = await _get_from_server_async(url, {})
    experiments = {exp["name"]: exp for exp in response.json()}
    experiment = experiments.get(experiment_name, None)
    if experiment is not None:
        return str(experiment["ensemble_ids"][0])
    raise ert.exceptions.NonExistentExperiment(
        f"Experiment {experiment_name} does not exist"
    )


async def _get_ensemble_size(ensemble_id: str) -> int:
    url = f"/ensembles/{ensemble_id}"
    response = await _get_from_server_async(url, {})
    response_json = response.json()
    return int(response_json["size"])


def init(*, workspace_name: str) -> None:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"]: exp["ensemble_ids"] for exp in response.json()}

    for special_key in _SPECIAL_KEYS:
        if f"{workspace_name}.{special_key}" in experiment_names:
            raise RuntimeError(
                f"Workspace {workspace_name} already registered in storage"
            )
        _init_experiment(
            experiment_name=f"{workspace_name}.{special_key}",
            parameters={},
            ensemble_size=-1,
            responses=[],
        )


def assert_storage_initialized(workspace_name: str) -> None:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"]: exp["ensemble_ids"] for exp in response.json()}

    for special_key in _SPECIAL_KEYS:
        if f"{workspace_name}.{special_key}" not in experiment_names:
            raise ert.exceptions.StorageError(
                "Storage is not initialized properly. "
                + "The workspace needs to be reinitialized"
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


def get_experiment_names(*, workspace_name: str) -> Set[str]:
    response = _get_from_server(path="experiments")
    experiment_names = {exp["name"] for exp in response.json()}
    for special_key in _SPECIAL_KEYS:
        key = f"{workspace_name}.{special_key}"
        if key in experiment_names:
            experiment_names.remove(key)
    return experiment_names


def _get_experiment_parameters(experiment_name: str) -> Iterable[str]:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistentExperiment(
            f"Cannot get parameters from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/parameters")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    return list(response.json())


async def _get_record_collection(
    records_url: str,
    record_name: str,
    ensemble_size: int,
    record_source: Optional[str] = None,
) -> Tuple[List[StorageRecordTransmitter], ert.data.RecordCollectionType]:
    return await _get_record_storage_transmitters(
        records_url, record_name, record_source, ensemble_size
    )


def get_ensemble_record(
    *,
    workspace_name: str,
    record_name: str,
    ensemble_size: int,
    experiment_name: Optional[str] = None,
    source: Optional[str] = None,
) -> ert.data.RecordCollection:
    records_url = get_records_url(
        workspace_name=workspace_name, experiment_name=experiment_name
    )

    transmitters, collection_type = get_event_loop().run_until_complete(
        _get_record_collection(
            records_url=records_url,
            record_name=record_name,
            record_source=source,
            ensemble_size=ensemble_size,
        )
    )
    records = tuple(
        get_event_loop().run_until_complete(transmitter.load())
        for transmitter in transmitters
    )
    return ert.data.RecordCollection(
        records=records, length=ensemble_size, collection_type=collection_type
    )


def get_ensemble_record_names(
    *, workspace_name: str, experiment_name: Optional[str] = None, _flatten: bool = True
) -> Iterable[str]:
    # _flatten is a parameter used only for testing separated parameter records
    if experiment_name is None:
        experiment_name = f"{workspace_name}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistentExperiment(
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
        raise ert.exceptions.NonExistentExperiment(
            f"Cannot get responses from non-existing experiment: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    response = _get_from_server(f"ensembles/{ensemble_id}/responses")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    # The ensemble responses are sent in the following form:
    #     {
    #         "polynomial_output": {"id": id, "name": name, "userdata": {}}
    #     }
    # therefore we extract only the keys

    return list(response.json().keys())


def delete_experiment(*, experiment_name: str) -> None:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistentExperiment(
            f"Experiment does not exist: {experiment_name}"
        )
    response = _delete_on_server(path=f"experiments/{experiment['id']}")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)
