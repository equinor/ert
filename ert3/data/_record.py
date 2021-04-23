import json
import shutil
import typing
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    Dict,
)

import aiofiles

# Type hinting for wrap must be turned off until (1) is resolved.
# (1) https://github.com/Tinche/aiofiles/issues/8
from aiofiles.os import wrap  # type: ignore
from pydantic import (
    BaseModel,
    StrictBytes,
    StrictFloat,
    StrictInt,
    StrictStr,
    validator,
    root_validator,
)

_copy = wrap(shutil.copy)

strict_number = Union[StrictInt, StrictFloat]
record_data = Union[
    List[strict_number],
    Dict[StrictStr, strict_number],
    Dict[StrictInt, strict_number],
    List[StrictBytes],
]


def parse_json_key_as_int(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {int(k): v for k, v in obj.items()}
    return obj


class _DataElement(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


def _build_record_index(
    data: record_data,
) -> Tuple[Union[StrictInt, StrictStr], ...]:
    if isinstance(data, MutableMapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class RecordType(str, Enum):
    LIST_FLOAT = "LIST_FLOAT"
    MAPPING_INT_FLOAT = "MAPPING_INT_FLOAT"
    MAPPING_STR_FLOAT = "MAPPING_STR_FLOAT"
    LIST_BYTES = "LIST_BYTES"


class Record(_DataElement):
    data: record_data
    index: Optional[Tuple[Union[StrictStr, StrictInt], ...]] = None

    @validator("index", pre=True)
    def index_validator(
        cls,
        index: Optional[Tuple[Union[StrictStr, StrictInt], ...]],
        values: Dict[str, Any],
    ) -> Optional[Tuple[Union[StrictStr, StrictInt], ...]]:
        if index is None and "data" in values:
            index = _build_record_index(values["data"])
        return index

    @property
    def record_type(self) -> RecordType:
        if isinstance(self.data, list):
            if not self.data:
                return RecordType.LIST_FLOAT
            if isinstance(self.data[0], (int, float)):
                return RecordType.LIST_FLOAT
            if isinstance(self.data[0], bytes):
                return RecordType.LIST_BYTES
        elif isinstance(self.data, Mapping):
            if not self.data:
                return RecordType.MAPPING_STR_FLOAT
            if isinstance(list(self.data.keys())[0], (int, float)):
                return RecordType.MAPPING_INT_FLOAT
            if isinstance(list(self.data.keys())[0], str):
                return RecordType.MAPPING_STR_FLOAT
        raise TypeError(
            f"Not able to deduce record type from data was: {type(self.data)}"
        )

    @root_validator(skip_on_failure=True)
    def ensure_consistent_index(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        assert (
            "data" in record and "index" in record
        ), "both data and index must be defined for a record"
        norm_record_index = _build_record_index(record["data"])
        assert (
            norm_record_index == record["index"]
        ), f"inconsistent index {norm_record_index} vs {record['index']}"
        return record


class EnsembleRecord(_DataElement):
    records: Tuple[Record, ...]
    ensemble_size: Optional[int] = None

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if ensemble_size == None and "records" in values:
            ensemble_size = len(values["records"])
        return ensemble_size

    @root_validator(skip_on_failure=True)
    def ensure_consistent_ensemble_size(
        cls, ensemble_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert "records" in ensemble_record and "ensemble_size" in ensemble_record
        assert len(ensemble_record["records"]) == ensemble_record["ensemble_size"]
        return ensemble_record


class MultiEnsembleRecord(_DataElement):
    ensemble_records: Mapping[str, EnsembleRecord]
    record_names: Optional[Tuple[str, ...]] = None
    ensemble_size: Optional[int] = None

    @validator("record_names", pre=True, always=True)
    def record_names_validator(
        cls, record_names: Optional[Tuple[str, ...]], values: Dict[str, Any]
    ) -> Optional[Tuple[str, ...]]:
        if record_names == None and "ensemble_records" in values:
            ensemble_records = values["ensemble_records"]
            record_names = tuple(ensemble_records.keys())
        return record_names

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if (
            ensemble_size == None
            and "ensemble_records" in values
            and "record_names" in values
        ):
            record_names = values["record_names"]
            ensemble_records = values["ensemble_records"]
            first_record = ensemble_records[record_names[0]]
            try:
                ensemble_size = first_record.ensemble_size
            except AttributeError:
                ensemble_size = len(first_record["records"])
        return ensemble_size

    @root_validator(skip_on_failure=True)
    def ensure_consistent_ensemble_size(
        cls, multi_ensemble_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        ensemble_size = multi_ensemble_record["ensemble_size"]
        for ensemble_record in multi_ensemble_record["ensemble_records"].values():
            if ensemble_size != ensemble_record.ensemble_size:
                raise AssertionError("Inconsistent ensemble record size")
        return multi_ensemble_record

    @root_validator(skip_on_failure=True)
    def ensure_consistent_record_names(
        cls, multi_ensemble_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert "record_names" in multi_ensemble_record
        record_names = tuple(multi_ensemble_record["ensemble_records"].keys())
        assert multi_ensemble_record["record_names"] == record_names
        return multi_ensemble_record

    def __len__(self) -> int:
        assert self.record_names is not None
        return len(self.record_names)


class RecordTransmitterState(Enum):
    transmitted = auto()
    not_transmitted = auto()


class RecordTransmitterType(Enum):
    in_memory = auto()
    ert_storage = auto()
    shared_disk = auto()


class RecordTransmitter:
    def __init__(self) -> None:
        self._state = RecordTransmitterState.not_transmitted

    def _set_transmitted_state(self) -> None:
        self._state = RecordTransmitterState.transmitted

    def is_transmitted(self) -> bool:
        return self._state == RecordTransmitterState.transmitted

    @property
    @abstractmethod
    def transmitter_type(self) -> RecordTransmitterType:
        pass

    @abstractmethod
    async def dump(self, location: Path) -> None:
        pass

    @abstractmethod
    async def load(self) -> Record:
        pass

    @abstractmethod
    async def transmit_data(
        self,
        data: record_data,
    ) -> None:
        pass

    @abstractmethod
    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        pass


class SharedDiskRecordTransmitter(RecordTransmitter):
    _TYPE: RecordTransmitterType = RecordTransmitterType.shared_disk

    def __init__(self, name: str, storage_path: Path):
        super().__init__()
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._concrete_key = f"{name}_{uuid.uuid4()}"
        self._uri: typing.Optional[str] = None
        self._record_type: typing.Optional[RecordType] = None

    def _set_transmitted(self, uri: Path, record_type: RecordType) -> None:
        super()._set_transmitted_state()
        self._uri = str(uri)
        self._record_type = record_type

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._TYPE

    async def _transmit(self, record: Record) -> None:
        storage_uri = self._storage_path / self._concrete_key
        if record.record_type != RecordType.LIST_BYTES:
            contents = json.dumps(record.data)
            async with aiofiles.open(storage_uri, mode="w") as f:
                await f.write(contents)
        else:
            async with aiofiles.open(storage_uri, mode="wb") as f:  # type: ignore
                await f.write(record.data[0])  # type: ignore
        self._set_transmitted(storage_uri, record_type=record.record_type)

    async def transmit_data(
        self,
        data: record_data,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        return await self._transmit(record)

    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                record = Record(data=json.loads(contents))
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                record = Record(data=[contents])
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )
        return await self._transmit(record)

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type != RecordType.LIST_BYTES:
            async with aiofiles.open(str(self._uri)) as f:
                contents = await f.read()
                if self._record_type == RecordType.MAPPING_INT_FLOAT:
                    data = json.loads(contents, object_hook=parse_json_key_as_int)
                else:
                    data = json.loads(contents)
                return Record(data=data)
        else:
            async with aiofiles.open(str(self._uri), mode="rb") as f:  # type: ignore
                data = await f.read()
                return Record(data=[data])

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        await _copy(self._uri, str(location))


class InMemoryRecordTransmitter(RecordTransmitter):
    _TYPE: RecordTransmitterType = RecordTransmitterType.in_memory
    # TODO: these fields should be Record, but that does not work until
    # https://github.com/cloudpipe/cloudpickle/issues/403 has been released.
    _data: Optional[Any] = None
    _index: Optional[Any] = None

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def _set_transmitted(self, record: Record) -> None:
        super()._set_transmitted_state()
        self._data = record.data
        self._index = record.index

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._TYPE

    @abstractmethod
    async def transmit_data(
        self,
        data: record_data,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        self._set_transmitted(record)

    @abstractmethod
    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                record = Record(data=json.loads(contents))
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                record = Record(data=[contents])
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )
        self._set_transmitted(record)

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        return Record(data=self._data, index=self._index)

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        if self._data is None:
            raise ValueError("cannot dump Record with no data")
        record = Record(data=self._data, index=self._index)
        if record.record_type != RecordType.LIST_BYTES:
            async with aiofiles.open(str(location), mode="w") as f:
                await f.write(json.dumps(self._data))
        else:
            async with aiofiles.open(str(location), mode="wb") as f:  # type: ignore
                await f.write(self._data[0])
