import io
import pathlib
import tarfile
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from concurrent import futures
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
from beartype import beartype
from beartype.roar import BeartypeException  # type: ignore
from pydantic import PositiveInt

from ert.serialization import get_serializer
from ert_shared.async_utils import get_event_loop

number = Union[int, float]
numerical_record_data = Union[
    List[number],
    Dict[str, number],
    Dict[int, number],
]
blob_record_data = bytes
record_data = Union[numerical_record_data, blob_record_data]
record_collection = Tuple["Record", ...]
RecordIndex = Union[Tuple[int, ...], Tuple[str, ...]]


@beartype
def _build_record_index(
    data: numerical_record_data,
) -> Tuple[Any, ...]:
    if isinstance(data, MutableMapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class RecordValidationError(Exception):
    pass


class RecordType(str, Enum):
    LIST_FLOAT = "LIST_FLOAT"
    MAPPING_INT_FLOAT = "MAPPING_INT_FLOAT"
    MAPPING_STR_FLOAT = "MAPPING_STR_FLOAT"
    BYTES = "BYTES"


class Record(ABC):
    @property
    @abstractmethod
    def data(self) -> record_data:
        pass

    @property
    @abstractmethod
    def record_type(self) -> RecordType:
        pass


class BlobRecord(Record):
    def __init__(self, data: blob_record_data) -> None:
        self._record_type = RecordType.BYTES

        try:
            self._data = self._validate_data(data)
        except BeartypeException as e:
            raise RecordValidationError(str(e))

    @beartype
    def _validate_data(self, data: blob_record_data) -> blob_record_data:
        return data

    @property
    def data(self) -> blob_record_data:
        return self._data

    @property
    def record_type(self) -> RecordType:
        return RecordType.BYTES

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return self.__dict__ == o.__dict__
        return False


class NumericalRecord(Record):
    def __init__(
        self, data: numerical_record_data, index: Optional[RecordIndex] = None
    ) -> None:
        if isinstance(data, (set, frozenset, deque, tuple)):
            data = [val for _, val in enumerate(data)]
        try:
            self._validate_data(data)
        except BeartypeException as e:
            raise RecordValidationError(str(e))
        self._data = data

        if index is None:
            index = _build_record_index(data)
        self._validate_index(data, index)
        self._index = index

        self._type = self._infer_type(data)

    @beartype
    def _validate_data(self, data: numerical_record_data) -> None:
        # beartype does not do deep validation on dicts, so we do non-nested validation
        # TODO: remove once https://github.com/beartype/beartype/issues/53 is done
        if isinstance(data, dict):
            key_type, val_type = (
                type(next(iter(data.keys()))),
                type(next(iter(data.values()))),
            )
            for key, val in data.items():
                if not isinstance(key, key_type) or not isinstance(key, (int, str)):
                    raise RecordValidationError(f"unexpected key type {type(key)}")
                if not isinstance(val, val_type) or not isinstance(val, (int, float)):
                    raise RecordValidationError(f"unexpected value type {type(val)}")

    def _validate_index(
        self, data: numerical_record_data, index: Optional[RecordIndex] = None
    ) -> None:
        norm_record_index = _build_record_index(data)
        if norm_record_index != index:
            raise RecordValidationError(
                f"inconsistent index {norm_record_index} vs {index}"
            )
        try:
            idx_type = type(next(iter(index)))
        except StopIteration:
            return
        for idx in index:
            if not isinstance(idx, idx_type):
                raise RecordValidationError(
                    f"unexpected index type {type(idx)}, expected {idx_type}"
                )

    def _infer_type(self, data: numerical_record_data) -> RecordType:
        if isinstance(data, (list, tuple)):
            if not data or isinstance(data[0], (int, float)):
                return RecordType.LIST_FLOAT
        elif isinstance(data, Mapping):
            if not data:
                return RecordType.MAPPING_STR_FLOAT
            from_ = list(data.keys())[0]
            if isinstance(from_, (int, float)):
                return RecordType.MAPPING_INT_FLOAT
            if isinstance(from_, str):
                return RecordType.MAPPING_STR_FLOAT
        raise RecordValidationError(f"unexpected data type {type(data)}")

    @property
    def index(self) -> RecordIndex:
        return self._index

    @property
    def data(self) -> record_data:
        return self._data

    @property
    def record_type(self) -> RecordType:
        return self._type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return self.__dict__ == o.__dict__
        return False


class RecordCollectionType(str, Enum):
    NON_UNIFORM = "NON_UNIFORM"
    UNIFORM = "UNIFORM"


_RecordTupleType = Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]


class RecordCollection:
    """Storage container for records of the same record-type, always non-empty.

    If the collection type is uniform, all records are identical. The typical
    use case for a uniform collection is a record that is constant over an
    ensemble.
    """

    def __init__(
        self,
        records: Tuple[Record, ...],
        length: Optional[PositiveInt] = None,
        collection_type: RecordCollectionType = RecordCollectionType.NON_UNIFORM,
    ):
        if len(records) < 1:
            raise ValueError("At least one record must be provided")
        if collection_type == RecordCollectionType.UNIFORM:
            if len(records) > 1:
                raise ValueError("Multiple records provided for a uniform record")
            if length is None:
                raise ValueError("Length missing for uniform record")
            self._records = cast(_RecordTupleType, records * length)
            self._length = length
        else:
            if length is not None and length != len(records):
                raise ValueError(
                    f"Requested length ({length}) does not match the record count ({len(records)})"
                )
            for record in records:
                if record.record_type != records[0].record_type:
                    raise ValueError(
                        "Record collections must have a uniform record type"
                    )
            self._records = cast(_RecordTupleType, records)
            self._length = len(self._records)
        self._collection_type = collection_type

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False

    def __len__(self) -> int:
        return self._length

    @property
    def records(self) -> _RecordTupleType:
        return self._records

    @property
    def record_type(self) -> RecordType:
        assert self._records[0].record_type is not None  # mypy needs this
        return self._records[0].record_type

    @property
    def collection_type(self) -> RecordCollectionType:
        return self._collection_type


async def make_tar(file_path: pathlib.Path) -> bytes:
    """Walk the files under a filepath and encapsulate the file(s)
    in a tar package."""
    _executor = futures.ThreadPoolExecutor()
    return await get_event_loop().run_in_executor(_executor, _sync_make_tar, file_path)


def _sync_make_tar(file_path: pathlib.Path) -> bytes:
    tar_obj = io.BytesIO()
    with tarfile.open(fileobj=tar_obj, mode="w") as tar:
        tar.add(file_path, arcname="")
    return tar_obj.getvalue()


async def load_collection_from_file(
    file_path: pathlib.Path,
    mime: str,
    length: int = 1,
    is_directory: bool = False,
) -> RecordCollection:
    if mime == "application/octet-stream":
        if is_directory:
            return RecordCollection(
                records=(BlobRecord(data=await make_tar(file_path)),),
                length=length,
                collection_type=RecordCollectionType.UNIFORM,
            )
        else:
            async with aiofiles.open(file_path, "rb") as filebinary:
                return RecordCollection(
                    records=(BlobRecord(data=await filebinary.read()),),
                    length=length,
                    collection_type=RecordCollectionType.UNIFORM,
                )
    raw_ensrecord = await get_serializer(mime).decode_from_path(file_path)
    return RecordCollection(
        records=tuple(NumericalRecord(data=raw_record) for raw_record in raw_ensrecord)
    )
