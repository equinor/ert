import pathlib
import shutil
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    cast,
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
    PositiveInt,
    root_validator,
    validator,
)
from ert.serialization import get_serializer

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


class RecordCollectionType(str, Enum):
    NON_UNIFORM = "NON_UNIFORM"
    UNIFORM = "UNIFORM"


_RecordTupleType = Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]


class RecordCollection:
    def __init__(
        self,
        records: Tuple[Record, ...],
        ensemble_size: Optional[PositiveInt] = None,
        collection_type: RecordCollectionType = RecordCollectionType.NON_UNIFORM,
    ):
        if len(records) < 1:
            raise ValueError("At least one record must be provided")
        if collection_type == RecordCollectionType.UNIFORM:
            if len(records) > 1:
                raise ValueError("Multiple records provided for a uniform record")
            if ensemble_size is None:
                raise ValueError("Ensemble size missing for uniform record")
            self._records = cast(_RecordTupleType, records * ensemble_size)
            self._ensemble_size = ensemble_size
        else:
            if ensemble_size is not None and ensemble_size != len(records):
                raise ValueError("Ensemble size does not match the record count")
            for record in records:
                if record.record_type != records[0].record_type:
                    raise ValueError("Ensemble records must have a uniform record type")
            self._records = cast(_RecordTupleType, records)
            self._ensemble_size = len(self._records)
        self._collection_type = collection_type

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__

    @property
    def records(self) -> _RecordTupleType:
        return self._records

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @property
    def record_type(self) -> RecordType:
        assert self._records[0].record_type is not None  # mypy needs this
        return self._records[0].record_type

    @property
    def collection_type(self) -> RecordCollectionType:
        return self._collection_type


class RecordTransmitterState(Enum):
    transmitted = auto()
    not_transmitted = auto()


class RecordTransmitterType(Enum):
    in_memory = auto()
    ert_storage = auto()
    shared_disk = auto()


class RecordTransmitter:
    def __init__(self, transmitter_type: RecordTransmitterType) -> None:
        self._state = RecordTransmitterState.not_transmitted
        self._uri: str = ""
        self._record_type: Optional[RecordType] = None
        self._transmitter_type: RecordTransmitterType = transmitter_type

    def _set_transmitted_state(
        self, uri: str, record_type: Optional[RecordType]
    ) -> None:
        self._state = RecordTransmitterState.transmitted
        self._uri = uri
        self._record_type = record_type

    def is_transmitted(self) -> bool:
        return self._state == RecordTransmitterState.transmitted

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._transmitter_type

    @abstractmethod
    async def _load_numerical_record(self) -> NumericalRecord:
        pass

    @abstractmethod
    async def _load_blob_record(self) -> BlobRecord:
        pass

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type != RecordType.BYTES:
            return await self._load_numerical_record()
        return await self._load_blob_record()

    @abstractmethod
    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        pass

    @abstractmethod
    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        pass

    async def transmit_record(self, record: Record) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if isinstance(record, NumericalRecord):
            uri = await self._transmit_numerical_record(record)
        elif isinstance(record, BlobRecord):
            uri = await self._transmit_blob_record(record)
        else:
            raise TypeError(f"Record type not supported {type(record)}")
        self._set_transmitted_state(uri, record_type=record.record_type)

    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as fb:
                contents_b: bytes = await fb.read()
                blob_record = BlobRecord(data=contents_b)
            uri = await self._transmit_blob_record(blob_record)
            self._set_transmitted_state(uri, blob_record.record_type)
        else:
            serializer = get_serializer(mime)
            async with aiofiles.open(str(file), mode="rt", encoding="utf-8") as ft:
                contents_t: str = await ft.read()
                num_record = NumericalRecord(data=serializer.decode(contents_t))
            uri = await self._transmit_numerical_record(num_record)
            self._set_transmitted_state(uri, num_record.record_type)

    async def dump(self, location: Path, mime: str) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        record = await self.load()
        if isinstance(record, NumericalRecord):
            async with aiofiles.open(str(location), mode="wt", encoding="utf-8") as ft:
                await ft.write(get_serializer(mime).encode(record.data))
        else:
            async with aiofiles.open(str(location), mode="wb") as fb:
                await fb.write(record.data)  # type: ignore


class SharedDiskRecordTransmitter(RecordTransmitter):
    _INTERNAL_MIME_TYPE = "application/x-yaml"

    def __init__(self, name: str, storage_path: Path):
        super().__init__(RecordTransmitterType.shared_disk)
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._concrete_key = f"{name}_{uuid.uuid4()}"
        self._storage_uri = self._storage_path / self._concrete_key

    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        contents = get_serializer(
            SharedDiskRecordTransmitter._INTERNAL_MIME_TYPE
        ).encode(record.data)
        async with aiofiles.open(self._storage_uri, mode="wt", encoding="utf-8") as f:
            await f.write(contents)
        return str(self._storage_uri)

    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        async with aiofiles.open(self._storage_uri, mode="wb") as f:
            await f.write(record.data)
        return str(self._storage_uri)

    async def _load_numerical_record(self) -> NumericalRecord:
        async with aiofiles.open(str(self._uri), mode="rt", encoding="utf-8") as f:
            contents = await f.read()
        serializer = get_serializer(SharedDiskRecordTransmitter._INTERNAL_MIME_TYPE)
        data = serializer.decode(contents)
        return NumericalRecord(data=data)

    async def _load_blob_record(self) -> BlobRecord:
        async with aiofiles.open(str(self._uri), mode="rb") as f:
            data = await f.read()
        return BlobRecord(data=data)

    async def dump(self, location: Path, mime: str) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        if (
            self._record_type == RecordType.BYTES
            or mime == SharedDiskRecordTransmitter._INTERNAL_MIME_TYPE
        ):
            await _copy(self._uri, str(location))
        else:
            record = await self._load_numerical_record()
            contents = get_serializer(mime).encode(record.data)
            async with aiofiles.open(location, mode="wt", encoding="utf-8") as f:
                await f.write(contents)


class InMemoryRecordTransmitter(RecordTransmitter):
    def __init__(self, name: str):
        super().__init__(RecordTransmitterType.in_memory)
        self._name = name
        self._record: Record

    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        self._record = record
        return "in_memory"

    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        self._record = record
        return "in_memory"

    async def _load_numerical_record(self) -> NumericalRecord:
        return NumericalRecord(data=self._record.data)

    async def _load_blob_record(self) -> BlobRecord:
        return BlobRecord(data=self._record.data)


def load_collection_from_file(
    file_path: pathlib.Path, mime: str, ensemble_size: int = 1
) -> RecordCollection:
    if mime == "application/octet-stream":
        with open(file_path, "rb") as fb:
            return RecordCollection(
                records=(BlobRecord(data=fb.read()),),
                ensemble_size=ensemble_size,
                collection_type=RecordCollectionType.UNIFORM,
            )

    with open(file_path, "rt", encoding="utf-8") as f:
        raw_ensrecord = get_serializer(mime).decode_from_file(f)
    return RecordCollection(
        records=tuple(NumericalRecord(data=raw_record) for raw_record in raw_ensrecord)
    )
