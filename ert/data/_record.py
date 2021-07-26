import json
import shutil
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
numerical_record_data = Union[
    List[strict_number],
    Dict[StrictStr, strict_number],
    Dict[StrictInt, strict_number],
]
blob_record_data = StrictBytes
record_data = Union[numerical_record_data, blob_record_data]


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


RecordIndex = Tuple[Union[StrictInt, StrictStr], ...]


def _build_record_index(
    data: numerical_record_data,
) -> RecordIndex:
    if isinstance(data, MutableMapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class RecordType(str, Enum):
    LIST_FLOAT = "LIST_FLOAT"
    MAPPING_INT_FLOAT = "MAPPING_INT_FLOAT"
    MAPPING_STR_FLOAT = "MAPPING_STR_FLOAT"
    BYTES = "BYTES"


class Record(_DataElement):
    data: record_data
    record_type: Optional[RecordType] = None

    @validator("record_type", pre=True)
    def record_type_validator(
        cls,
        record_type: Optional[RecordType],
        values: Dict[str, Any],
    ) -> Optional[RecordType]:
        if record_type is None and "data" in values:
            data = values["data"]
            if isinstance(data, list):
                if not data or isinstance(data[0], (int, float)):
                    return RecordType.LIST_FLOAT
            elif isinstance(data, bytes):
                return RecordType.BYTES
            elif isinstance(data, Mapping):
                if not data:
                    return RecordType.MAPPING_STR_FLOAT
                if isinstance(list(data.keys())[0], (int, float)):
                    return RecordType.MAPPING_INT_FLOAT
                if isinstance(list(data.keys())[0], str):
                    return RecordType.MAPPING_STR_FLOAT
        return record_type

    def get_instance(self) -> "Record":
        if self.record_type is not None:
            if self.record_type == RecordType.BYTES:
                return BlobRecord(data=self.data)
        return NumericalRecord(data=self.data)


class NumericalRecord(Record):
    data: numerical_record_data
    index: Optional[RecordIndex] = None

    @validator("index", pre=True)
    def index_validator(
        cls,
        index: Optional[RecordIndex],
        values: Dict[str, Any],
    ) -> Optional[RecordIndex]:
        if index is None and "data" in values:
            index = _build_record_index(values["data"])
        return index

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


class BlobRecord(Record):
    data: blob_record_data


class RecordCollection(_DataElement):
    records: Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]
    ensemble_size: Optional[int] = None

    @property
    def record_type(self) -> Optional[RecordType]:
        return self.records[0].record_type

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if ensemble_size is None and "records" in values:
            ensemble_size = len(values["records"])
        assert ensemble_size is not None and ensemble_size > 0
        return ensemble_size

    @root_validator(skip_on_failure=True)
    def ensure_consistent_ensemble(
        cls, ensemble_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert "records" in ensemble_record and "ensemble_size" in ensemble_record
        assert len(ensemble_record["records"]) == ensemble_record["ensemble_size"]
        record_type = ensemble_record["records"][0].record_type
        for record in ensemble_record["records"][1:]:
            if record.record_type != record_type:
                raise ValueError("Ensemble records must have a uniform record type")
        return ensemble_record


class RecordCollectionMap(_DataElement):
    ensemble_records: Mapping[str, RecordCollection]
    record_names: Optional[Tuple[str, ...]] = None
    ensemble_size: Optional[int] = None

    @validator("record_names", pre=True, always=True)
    def record_names_validator(
        cls, record_names: Optional[Tuple[str, ...]], values: Dict[str, Any]
    ) -> Optional[Tuple[str, ...]]:
        if record_names is None and "ensemble_records" in values:
            ensemble_records = values["ensemble_records"]
            record_names = tuple(ensemble_records.keys())
        return record_names

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if (
            ensemble_size is None
            and "ensemble_records" in values
            and "record_names" in values
        ):
            record_names = values["record_names"]
            assert len(record_names) > 0
            ensemble_records = values["ensemble_records"]
            assert len(ensemble_records) > 0
            first_record = ensemble_records[record_names[0]]
            try:
                ensemble_size = first_record.ensemble_size
            except AttributeError:
                ensemble_size = len(first_record["records"])
        assert ensemble_size is not None and ensemble_size > 0
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
    async def transmit_data(self, data: record_data) -> None:
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
        self._uri: Optional[str] = None
        self._record_type: Optional[RecordType] = None

    def _set_transmitted(self, uri: Path, record_type: Optional[RecordType]) -> None:
        super()._set_transmitted_state()
        self._uri = str(uri)
        self._record_type = record_type

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._TYPE

    async def _transmit(self, record: Record) -> None:
        storage_uri = self._storage_path / self._concrete_key
        if isinstance(record, NumericalRecord):
            contents = json.dumps(record.data)
            async with aiofiles.open(storage_uri, mode="w") as f:
                await f.write(contents)
        elif isinstance(record, BlobRecord):
            async with aiofiles.open(storage_uri, mode="wb") as f:  # type: ignore
                await f.write(record.data)  # type: ignore
        else:
            raise TypeError(f"Record type not supported {type(record)}")
        self._set_transmitted(storage_uri, record_type=record.record_type)

    async def transmit_data(self, data: record_data) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        return await self._transmit(record.get_instance())

    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record: Union[NumericalRecord, BlobRecord]
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                record = NumericalRecord(data=json.loads(contents))
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                record = BlobRecord(data=contents)
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )
        return await self._transmit(record)

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type != RecordType.BYTES:
            async with aiofiles.open(str(self._uri)) as f:
                contents = await f.read()
                if self._record_type == RecordType.MAPPING_INT_FLOAT:
                    data = json.loads(contents, object_hook=parse_json_key_as_int)
                else:
                    data = json.loads(contents)
                return NumericalRecord(data=data)
        else:
            async with aiofiles.open(str(self._uri), mode="rb") as f:  # type: ignore
                data = await f.read()
                return BlobRecord(data=data)

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        await _copy(self._uri, str(location))


class InMemoryRecordTransmitter(RecordTransmitter):
    _TYPE: RecordTransmitterType = RecordTransmitterType.in_memory

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._record: Record

    def _set_transmitted(self, record: Record) -> None:
        super()._set_transmitted_state()
        self._record = record

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._TYPE

    @abstractmethod
    async def transmit_data(self, data: record_data) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        self._set_transmitted(Record(data=data))

    @abstractmethod
    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record: Record
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                record = NumericalRecord(data=json.loads(contents))
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                record = BlobRecord(data=contents)
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )
        self._set_transmitted(record)

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        return self._record

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        if self._record.record_type != RecordType.BYTES:
            async with aiofiles.open(str(location), mode="w") as f:
                await f.write(json.dumps(self._record.data))
        else:
            async with aiofiles.open(str(location), mode="wb") as f:  # type: ignore
                await f.write(self._record.data)  # type: ignore
