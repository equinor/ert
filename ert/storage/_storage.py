from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
    List,
    Union,
)
import io
import logging
import json
from pathlib import Path

from http import HTTPStatus
import httpx
import aiofiles
import pandas as pd
from pydantic import BaseModel
import requests
import ert

from ert_shared.storage.connection import get_info

logger = logging.getLogger(__name__)

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
    _TYPE: ert.data.RecordTransmitterType = ert.data.RecordTransmitterType.ert_storage

    def __init__(self, name: str, storage_url: str, iens: Optional[int] = None):
        super().__init__()
        self._name: str = name
        self._storage_url = storage_url
        self._uri: str = ""
        self._record_type: Optional[ert.data.RecordType] = None
        self._real_id: Optional[int] = iens

    def _set_transmitted(self, uri: str, record_type: ert.data.RecordType) -> None:
        super()._set_transmitted_state()
        self._uri = uri
        self._record_type = record_type

    @property
    def transmitter_type(self) -> ert.data.RecordTransmitterType:
        return self._TYPE

    async def _transmit(self, record: ert.data.Record) -> None:
        record_type = record.record_type
        url_base = f"{self._storage_url}/{self._name}"

        if record_type != ert.data.RecordType.BYTES:
            url = f"{url_base}/matrix"
        else:
            url = f"{url_base}/file"
        if self._real_id is not None:
            url = f"{url}?realization_index={self._real_id}"
            url_base = f"{url_base}?realization_index={self._real_id}"

        await add_record(url, record)

        self._set_transmitted(url_base, record_type=record_type)

    async def transmit_data(
        self,
        data: ert.data.record_data,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        return await self._transmit(ert.data.make_record(data))

    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        record: ert.data.Record
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                record = ert.data.NumericalRecord(data=json.loads(contents))
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                record = ert.data.BlobRecord(data=contents)
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )
        return await self._transmit(record)

    async def load(self) -> ert.data.Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        return await load_record(self._uri, self._record_type)

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        record: ert.data.Record = await self.load()
        if isinstance(record, ert.data.NumericalRecord):
            contents = json.dumps(record.data)
            async with aiofiles.open(location, mode="w") as f:
                await f.write(contents)
        else:
            async with aiofiles.open(location, mode="wb") as f:  # type: ignore
                await f.write(record.data)  # type: ignore


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
        raise ert.exceptions.StorageError(resp.text)

    return resp


async def add_record(url: str, record: Any) -> None:
    headers = {
        "Token": StorageInfo.token(),
    }

    if record.record_type != ert.data.RecordType.BYTES:
        headers["content-type"] = "text/csv"
        data = pd.DataFrame([record.data]).to_csv().encode()
        await _post_to_server_async(url=url, headers=headers, data=data)
    else:
        data = {"file": io.BytesIO(record.data)}
        await _post_to_server_async(url=url, headers=headers, files=data)


async def load_record(url: str, record_type: Any) -> ert.data.Record:
    headers = {
        "Token": StorageInfo.token(),
    }

    if record_type != ert.data.RecordType.BYTES:
        headers["accept"] = "text/csv"
    else:
        headers["accept"] = "application/octet-stream"

    resp = await _get_from_server_async(url=url, headers=headers)
    content = resp.content
    if record_type != ert.data.RecordType.BYTES:
        dataframe = pd.read_csv(
            io.BytesIO(content), index_col=0, float_precision="round_trip"
        )
        for _, row in dataframe.iterrows():  # pylint: disable=no-member
            if record_type == ert.data.RecordType.LIST_FLOAT:
                data = row.to_list()
            elif record_type == ert.data.RecordType.MAPPING_STR_FLOAT:
                data = row.to_dict()
            elif record_type == ert.data.RecordType.MAPPING_INT_FLOAT:
                data = {int(k): v for k, v in row.to_dict().items()}
            else:
                raise ValueError(
                    f"Unexpected record type when loading numerical record: {record_type}"
                )
            return ert.data.NumericalRecord(data=data)
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


def get_records_url(workspace: Path) -> str:
    storage_url = StorageInfo.url()
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
    record_type: ert.data.RecordType,
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
        headers={"content-type": "text/csv"},
    )

    if response.status_code == 409:
        raise ert.exceptions.ElementExistsError("Record already exists")

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    meta_response = _put_to_server(f"{record_url}/userdata", json=metadata.dict())

    if meta_response.status_code != 200:
        raise ert.exceptions.StorageError(meta_response.text)


def _response2records(
    response_content: bytes, metadata: _NumericalMetaData
) -> ert.data.EnsembleRecord:

    record_type = metadata.record_type

    # the local variable dataframe is not read by pylint as pandas.Dataframe,
    # but due to chunking a pandas.TextFileReader
    # https://stackoverflow.com/questions/41844485/why-the-object-which-i-read-a-csv-file-using-pandas-from-is-textfilereader-obj

    records: List[ert.data.Record]
    if record_type == ert.data.RecordType.LIST_FLOAT:
        dataframe = pd.read_csv(
            io.BytesIO(response_content), index_col=0, float_precision="round_trip"
        )
        records = [
            ert.data.NumericalRecord(data=row.to_list())
            for _, row in dataframe.iterrows()  # pylint: disable=no-member
        ]
    elif record_type == ert.data.RecordType.MAPPING_INT_FLOAT:
        dataframe = pd.read_csv(
            io.BytesIO(response_content), index_col=0, float_precision="round_trip"
        )
        records = [
            ert.data.NumericalRecord(data={int(k): v for k, v in row.to_dict().items()})
            for _, row in dataframe.iterrows()  # pylint: disable=no-member
        ]
    elif record_type == ert.data.RecordType.MAPPING_STR_FLOAT:
        dataframe = pd.read_csv(
            io.BytesIO(response_content), index_col=0, float_precision="round_trip"
        )
        records = [
            ert.data.NumericalRecord(data=row.to_dict())
            for _, row in dataframe.iterrows()  # pylint: disable=no-member
        ]
    elif record_type == ert.data.RecordType.BYTES:
        assert metadata
        records = [
            ert.data.BlobRecord(data=response_content)
            for _ in range(metadata.ensemble_size)
        ]
    else:
        raise ValueError(
            f"Unexpected record type when loading numerical record: {record_type}"
        )
    return ert.data.EnsembleRecord(records=records)


def _combine_records(
    ensemble_records: List[ert.data.EnsembleRecord],
) -> ert.data.EnsembleRecord:
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
    return ert.data.EnsembleRecord(records=combined_records)


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
) -> ert.data.EnsembleRecord:
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    ensemble_id = experiment["ensemble_ids"][0]  # currently just one ens per exp
    metadata = _get_numerical_metadata(ensemble_id, record_name)

    if metadata.record_type == ert.data.RecordType.BYTES:
        headers = {"accept": "application/octet-stream"}
    else:
        headers = {"accept": "text/csv"}

    response = _get_from_server(
        path=f"ensembles/{ensemble_id}/records/{record_name}",
        headers=headers,
    )

    if response.status_code == 404:
        raise ert.exceptions.ElementMissingError(
            f"No {record_name} data for experiment: {experiment_name}"
        )

    if response.status_code != 200:
        raise ert.exceptions.StorageError(response.text)

    return _response2records(
        response_content=response.content,
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
    ensemble_record: ert.data.EnsembleRecord,
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
    ensemble_record: ert.data.EnsembleRecord,
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
    # If the EnsembleRecord has more than one record we assume
    # all records are the same and store only one record.
    # We store the original size in the metadata
    record = ensemble_record.records[0]

    assert isinstance(record.data, bytes)
    response = _post_to_server(
        f"{record_url}/file",
        files={
            "file": (record_name, io.BytesIO(record.data), "application/octet-stream")
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
) -> ert.data.EnsembleRecord:
    if experiment_name is None:
        experiment_name = f"{workspace}.{_ENSEMBLE_RECORDS}"
    experiment = _get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ert.exceptions.NonExistantExperiment(
            f"Cannot get {record_name} data, no experiment named: {experiment_name}"
        )

    param_names = _get_experiment_parameters(experiment_name)
    if record_name in param_names and param_names[record_name]:
        ensemble_records = [
            _get_data(
                experiment_name=experiment_name,
                record_name=record_name + _PARAMETER_RECORD_SEPARATOR + param_name,
            )
            for param_name in param_names[record_name]
        ]
        return _combine_records(ensemble_records)
    else:
        return _get_data(
            experiment_name=experiment_name,
            record_name=record_name,
        )


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
