import json
import shutil
import typing
import uuid
from abc import abstractmethod, ABC
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
    ValidationError,
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


class Record(ABC, _DataElement):
    # There should be an abstract property for data. However, pydantic does not
    # support that, it will complain about shadowing the field. That can be
    # solved using a Field alias, but then the derived classes become abstract,
    # since they do not have a data field. Defining a non-abstract data property
    # instead also does not work in combination with pydantic: the base property
    # will be called when accessing record.data.
    #
    # The current solution is to define no data property here at all. This has
    # the disadvantage that we are reverting to duck-typing  when we access the
    # data field in a record of unknown type (record.data). Not a problem for
    # Python, but it breaks mypy. Hence, in such cases, a 'type: ignore' needs
    # to be used.
    #
    # @property
    # @abstractmethod
    # def data(self) -> record_data:
    #     "Return the record data"

    @property
    @abstractmethod
    def record_type(self) -> RecordType:
        "Return the record type"


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

    @property
    def record_type(self) -> RecordType:
        if isinstance(self.data, list):
            if not self.data:
                return RecordType.LIST_FLOAT
            if isinstance(self.data[0], (int, float)):
                return RecordType.LIST_FLOAT
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


class BlobRecord(Record):
    data: blob_record_data

    @property
    def record_type(self) -> RecordType:
        if isinstance(self.data, bytes):
            return RecordType.BYTES
        raise TypeError(
            f"Not able to deduce record type from data was: {type(self.data)}"
        )


class EnsembleRecord(_DataElement):
    records: Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]
    ensemble_size: Optional[int] = None

    @property
    def record_type(self) -> RecordType:
        return self.records[0].record_type

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if ensemble_size == None and "records" in values:
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


# The use of this, rather ugly, function could possibly be avoided by moving some
# of the functionality of the transmitters into the Record classes.
def make_record(
    data: Any, index: Optional[RecordIndex] = None
) -> Union[NumericalRecord, BlobRecord]:
    validation_errors = []
    try:
        return NumericalRecord(data=data, index=index)
    except ValidationError as err:
        validation_errors.append(err)
    try:
        return BlobRecord(data=data)
    except ValidationError as err:
        validation_errors.append(err)
    # This function only fails with multiple validation errors, not sure how to
    # raise those.
    raise ValueError(validation_errors)


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
        record = make_record(data)
        return await self._transmit(record)

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
    # TODO: these fields should be Union[NumericalRecord, BlobRecord], but that
    # does not work until https://github.com/cloudpipe/cloudpickle/issues/403
    # has been released.
    _data: Optional[record_data] = None
    _index: Optional[RecordIndex] = None

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def _set_transmitted(self, record: Record) -> None:
        super()._set_transmitted_state()
        # See the Record class for the reason of the 'type ignore'.
        self._data = record.data  # type: ignore
        if isinstance(record, NumericalRecord):
            self._index = record.index

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._TYPE

    @abstractmethod
    async def transmit_data(self, data: record_data) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = make_record(data)
        self._set_transmitted(record)

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
        assert self._data is not None
        return make_record(self._data, index=self._index)

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        if self._data is None:
            raise ValueError("cannot dump Record with no data")
        record = make_record(data=self._data, index=self._index)
        if record.record_type != RecordType.BYTES:
            async with aiofiles.open(str(location), mode="w") as f:
                await f.write(json.dumps(self._data))
        else:
            async with aiofiles.open(str(location), mode="wb") as f:  # type: ignore
                await f.write(self._data)  # type: ignore
