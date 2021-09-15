import base64
import pathlib
import pickle
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
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
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


class RecordTransmitterState(str, Enum):
    transmitted = "transmitted"
    not_transmitted = "not_transmitted"


class RecordTransmitterType(str, Enum):
    in_memory = "in_memory"
    ert_storage = "ert_storage"
    shared_disk = "shared_disk"


class _RawRecordTransmitter(NamedTuple):
    """Crude schema for all record transmitters."""

    state: RecordTransmitterState
    uri: str
    record_type: RecordType
    transmitter_type: RecordTransmitterType
    data: Dict[str, Any]


class RecordTransmitter:
    def __init__(self, transmitter_type: RecordTransmitterType) -> None:
        self._state = RecordTransmitterState.not_transmitted
        self._uri: str = ""
        self._record_type: Optional[RecordType] = None
        self._transmitter_type: RecordTransmitterType = transmitter_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return self.__dict__ == o.__dict__
        return False

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

    @abstractmethod
    def _special_serialization_data(self) -> Dict[str, Any]:
        """Provide a dict with idiosyncratic data about this specific, concrete
        transmitter."""
        pass

    @classmethod
    @abstractmethod
    def _from_special_serialization_data(
        cls, data: Dict[str, Any]
    ) -> "RecordTransmitter":
        """From a dict describing the idiosyncratic data for this specific,
        concrete transmitter, return such a transmitter if the required data
        are present."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return _RawRecordTransmitter(
            transmitter_type=self._transmitter_type,
            state=self._state,
            record_type=self._record_type,
            uri=self._uri,
            data=self._special_serialization_data(),
        )._asdict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordTransmitter":
        raw = _RawRecordTransmitter(**data)
        transmitter = cls._from_special_serialization_data(raw.data)
        transmitter._state = RecordTransmitterState(raw.state)
        if raw.record_type:
            transmitter._record_type = RecordType(raw.record_type)
        transmitter._uri = raw.uri
        return transmitter


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
        # import requests
        # requests.get("http://example.org")
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

    def _special_serialization_data(self) -> Dict[str, Any]:
        return {
            "storage_path": str(self._storage_path),
            "concrete_key": self._concrete_key,
            "storage_uri": str(self._storage_uri),
        }

    @classmethod
    def _from_special_serialization_data(
        cls, data: Dict[str, Any]
    ) -> "RecordTransmitter":
        transmitter = cls("", Path(data["storage_path"]))
        transmitter._concrete_key = data["concrete_key"]
        transmitter._storage_uri = Path(data["storage_uri"])
        return transmitter


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

    def _special_serialization_data(self) -> Dict[str, Any]:
        data = {
            "name": self._name,
        }

        # pickling does not run __init__, which means the _record attribute may
        # or may not exist
        if hasattr(self, "_record"):
            data["record"] = (base64.b64encode(pickle.dumps(self._record))).decode()
        return data

    @classmethod
    def _from_special_serialization_data(
        cls, data: Dict[str, Any]
    ) -> "RecordTransmitter":
        transmitter = cls(data["name"])
        if "record" in data:
            transmitter._record = pickle.loads(base64.b64decode(data["record"]))
        return transmitter


def load_collection_from_file(
    file_path: pathlib.Path, mime: str, ens_size: int = 1
) -> RecordCollection:
    if mime == "application/octet-stream":
        with open(file_path, "rb") as fb:
            return RecordCollection(
                records=[BlobRecord(data=fb.read())] * ens_size,
            )

    with open(file_path, "rt", encoding="utf-8") as f:
        raw_ensrecord = get_serializer(mime).decode_from_file(f)
    return RecordCollection(
        records=[NumericalRecord(data=raw_record) for raw_record in raw_ensrecord]
    )
