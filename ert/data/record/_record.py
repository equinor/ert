import pathlib
from abc import ABC
from collections import deque
from enum import Enum
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

from beartype import beartype
from beartype.roar import BeartypeException  # type: ignore

import ert

number = Union[int, float]
numerical_record_data = Union[
    List[number],
    Dict[str, number],
    Dict[int, number],
]
blob_record_data = bytes
record_data = Union[numerical_record_data, blob_record_data]
record_collection = Tuple["Record", ...]
RecordIndex = Tuple[Union[int, str], ...]


def _build_record_index(
    data: numerical_record_data,
) -> RecordIndex:
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
    def data(self) -> record_data:
        pass

    @property
    def record_type(self) -> RecordType:
        pass

    @staticmethod
    def create(
        spec_or_record: Union[
            Dict[str, bytes], Dict[str, numerical_record_data], "Record"
        ]
    ) -> "Record":
        if isinstance(spec_or_record, Record):
            return spec_or_record
        data = spec_or_record["data"]
        if isinstance(data, bytes):
            return BlobRecord(data)
        else:
            index = spec_or_record.get("index")
            if not isinstance(index, int):
                index = None
            return NumericalRecord(data=data, index=index)


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
        if isinstance(o, BlobRecord):
            return self.__dict__ == o.__dict__
        return False


class NumericalRecord(Record):
    def __init__(
        self, data: numerical_record_data, index: Optional[RecordIndex] = None
    ) -> None:
        if isinstance(data, (set, frozenset, deque, tuple)):
            data = [val for _, val in enumerate(data)]

        try:
            self._data = self._validate_data(data)
        except BeartypeException as e:
            raise RecordValidationError(str(e))

        self._type = self._infer_type()
        self._index = self._validate_index(index)

    @beartype
    def _validate_data(self, data: numerical_record_data) -> numerical_record_data:
        # beartype does not do deep validation on dicts, so we do non-nested validation
        if isinstance(data, dict):
            for key, val in data.items():
                if not isinstance(key, (str, int)):
                    raise RecordValidationError(f"unexpected key type {type(key)}")
                if not isinstance(val, (int, float)):
                    raise RecordValidationError(f"unexpected value type {type(val)}")
        return data

    def _validate_index(self, index: Optional[RecordIndex] = None) -> RecordIndex:
        if index is None:
            return _build_record_index(self._data)

        norm_record_index = _build_record_index(self._data)
        assert (
            norm_record_index == index
        ), f"inconsistent index {norm_record_index} vs {index}"
        return index

    def _infer_type(self) -> RecordType:
        data = self._data
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
        if isinstance(o, NumericalRecord):
            return self.__dict__ == o.__dict__
        return False


class RecordCollection:
    def __init__(
        self, records: record_collection, ensemble_size: Optional[int] = None
    ) -> None:
        coerced_records = tuple(Record.create(record) for record in records)
        records, ens_size, record_type = self._validate_data(
            coerced_records, ensemble_size
        )
        self._records = records
        self._ensemble_size = ens_size
        self._record_type = record_type

    @beartype
    def _validate_data(
        self, records: record_collection, ensemble_size: Optional[int]
    ) -> Tuple[record_collection, int, RecordType]:
        if not records:
            raise RecordValidationError("no records")

        if ensemble_size is None:
            ensemble_size = len(records)

        if ensemble_size != len(records):
            raise RecordValidationError(
                f"ensemble size mismatch: {ensemble_size}/{len(records)}"
            )

        record_type = records[0].record_type
        for index, record in enumerate(records[1:]):
            if record.record_type != record_type:
                raise RecordValidationError(
                    f"Ensemble records must be homogenously type {record_type}, found"
                    + f" {record.record_type} at index {index}"
                )
        return (records, ensemble_size, record_type)

    @property
    def records(self) -> Tuple[Record, ...]:
        return self._records

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @property
    def record_type(self) -> Optional[RecordType]:
        return self.records[0].record_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, RecordCollection):
            return self.__dict__ == o.__dict__
        return False


def load_collection_from_file(
    file_path: pathlib.Path, mime: str, ens_size: int = 1
) -> RecordCollection:
    if mime == "application/octet-stream":
        with open(file_path, "rb") as fb:
            return RecordCollection(
                records=tuple(BlobRecord(data=fb.read()) for _ in range(ens_size))
            )

    with open(file_path, "rt", encoding="utf-8") as f:
        raw_ensrecord = ert.serialization.get_serializer(mime).decode_from_file(f)
    return RecordCollection(
        records=tuple(NumericalRecord(data=raw_record) for raw_record in raw_ensrecord)
    )
