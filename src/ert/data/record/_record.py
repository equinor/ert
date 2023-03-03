import json
import warnings
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from beartype import beartype
from beartype.roar import BeartypeDecorHintPepDeprecationWarning, BeartypeException
from pydantic import PositiveInt

# Mute PEP-585 warnings from Python 3.9:
warnings.simplefilter(action="ignore", category=BeartypeDecorHintPepDeprecationWarning)

number = Union[int, float]
numerical_record_data = Union[
    number, List[number], Dict[str, number], Dict[int, number]
]
blob_record_data = bytes
record_data = Union[
    numerical_record_data,
    blob_record_data,
]
record_collection = Tuple["Record", ...]
RecordIndex = Union[Tuple[int, ...], Tuple[str, ...]]


@beartype
def _build_record_index(
    data: numerical_record_data,
) -> Tuple[Any, ...]:
    if isinstance(data, (int, float)):
        return ()
    if isinstance(data, MutableMapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class RecordValidationError(Exception):
    pass


class RecordType(str, Enum):
    # NumericalRecord types
    SCALAR_FLOAT = "SCALAR_FLOAT"
    LIST_FLOAT = "LIST_FLOAT"
    MAPPING_INT_FLOAT = "MAPPING_INT_FLOAT"
    MAPPING_STR_FLOAT = "MAPPING_STR_FLOAT"
    # BlobRecord type
    BYTES = "BYTES"
    # RecordTree types
    NUMERICAL_TREE = "NUMERICAL_TREE"
    BLOB_TREE = "BLOB_TREE"


class Record(ABC):
    """The :class:`Record` class is an abstract class that
    includes record data and the record type. It represents a
    basic data `unit` used in ert.

    The record needs be either :class:`BlobRecord`,
    :class:`NumericalRecord`, :class:`NumericalRecordTree` or
    :class:`BlobRecordTree` type derived from `Record` class.
    """

    @property
    @abstractmethod
    def data(self) -> record_data:
        pass

    @property
    @abstractmethod
    def record_type(self) -> RecordType:
        pass


class BlobRecord(Record):
    """The :class:`BlobRecord` is an implementation of the Record class
    that treats any stream of bytes as binary data.
    """

    def __init__(self, data: blob_record_data) -> None:
        """Creates a BlobRecord instance from the binary record data.
        It automatically assigns :class:`RecordType` as :class:`RecordType.BYTES`

        Args:
            data: bytes type object

        Raises:
            RecordValidationError: Raises when data is of wrong type; ie.
                not bytes type object.
        """
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
        """Returns :class:`RecordType` set to :class:`RecordType.BYTES`."""
        return RecordType.BYTES

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return self.__dict__ == o.__dict__
        return False


class NumericalRecord(Record):
    """The :class:`NumericalRecord` is an implementation of the Record class
    that handles a scalar or an indexed list of numerical data.
    """

    def __init__(
        self, data: numerical_record_data, index: Optional[RecordIndex] = None
    ) -> None:
        """Creates a NumericalRecord instance.
        It automatically assigns :class:`RecordType` as either
        :class:`RecordType.SCALAR_FLOAT`, :class:`RecordType.LIST_FLOAT`,
        :class:`RecordType.MAPPING_INT_FLOAT` or :class:`RecordType.MAPPING_STR_FLOAT`.
        The data index is either automatically inferred from the data or
        can be provided as a parameter. Index should not be specified for scalar.
        Scalar data will always be stored as floats internally.

        Args:
            data: numerics that cannot be None
            index: data indices. If None is given, then the index will
                be an 0-indexed enumeration of the elements in the data

        Raises:
            RecordValidationError: Raises when
                data is a wrong type, or index was not build
                or provided correctly
        """
        if isinstance(data, (set, frozenset, deque, tuple)):
            data = [val for _, val in enumerate(data)]
        try:
            self._validate_data(data)
        except BeartypeException as e:
            raise RecordValidationError(str(e))

        self._data: numerical_record_data
        if isinstance(data, int):
            # This is for consistency with how serializers may
            # handle scalar data:
            self._data = float(data)
        else:
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

    def _validate_index(self, data: numerical_record_data, index: RecordIndex) -> None:
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
        if isinstance(data, (int, float)):
            return RecordType.SCALAR_FLOAT
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
    def data(self) -> numerical_record_data:
        return self._data

    @property
    def record_type(self) -> RecordType:
        """Returns the :class:`RecordType` of the record data, which is either
        :class:`RecordType.LIST_FLOAT`, :class:`RecordType.MAPPING_INT_FLOAT`,
        :class:`RecordType.MAPPING_STR_FLOAT` or :class:`RecordType.SCALAR_FLOAT`.
        """
        return self._type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return self.__dict__ == o.__dict__
        return False


RecordGen = TypeVar("RecordGen", BlobRecord, NumericalRecord)


class RecordTree(Record, Generic[RecordGen]):
    """The :class:`RecordTree` represents a base abstract class for manipulating
    multiple Record objects represented as a hierarchical structure of records
    where the leaves are the actual records (ie. BlobRecords or NumericalRecords)
    and the internal nodes are in essence namespaces for the particular subtree.

    There are two implementations: :class:`BlobRecordTree`
    and :class:`NumericalRecordTree`. For instance to initialize BlobRecordTree
    one can use:
    ert.data.BlobRecordTree(record_tree={
            "key_A:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            "key_B:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            "group_OP2": {
                "key_AA:OP2": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
                "key_BA:OP2": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            },
            },
    )
    """

    def __init__(self, record_dict: Dict[str, Any]) -> None:
        """Create instance of the RecordTree from a dictionary.
        The flattened version of the dict is retrieved via
        :func:`RecordTree.flat_record_dict` property.

        Args:
            record_dict: hierarchical representation of records,
                see :class:`RecordTree`.

        Raises:
            RecordValidationError: Raises when leaf Records
                are not of the same type.
        """
        self._record_type: RecordType = RecordType.BYTES
        self._flat_record_dict: Dict[str, RecordGen] = self._flatten_record_dict(
            record_dict
        )
        try:
            self._validate_data(self._flat_record_dict)
        except BeartypeException as e:
            raise RecordValidationError(str(e))

    @beartype
    def _validate_data(self, flat_record_dict: Dict[str, RecordGen]) -> None:
        # beartype does not do deep validation on dicts, so we do non-nested validation
        # TODO: remove once https://github.com/beartype/beartype/issues/53 is done
        if flat_record_dict:
            _record_type = list(flat_record_dict.values())[0].record_type
            for val in flat_record_dict.values():
                if val.record_type != _record_type:
                    raise RecordValidationError(
                        "RecordTree needs same record types "
                        f"{_record_type}!={val.record_type}"
                    )
        else:
            raise RecordValidationError("No records found in RecordTree")

    def _flatten_record_dict(
        self,
        record_dict: Dict[str, Any],
        root: str = "",
    ) -> Dict[str, RecordGen]:
        flat_record_dict: Dict[str, RecordGen] = {}
        for record_name, record in record_dict.items():
            if isinstance(record, dict):
                flat_record_dict.update(
                    self._flatten_record_dict(record, f"{root}{record_name}/")
                )
            elif isinstance(self, BlobRecordTree) and isinstance(record, BlobRecord):
                flat_record_dict[f"{root}{record_name}"] = record
            elif isinstance(self, NumericalRecordTree) and isinstance(
                record, NumericalRecord
            ):
                flat_record_dict[f"{root}{record_name}"] = record
            else:
                raise RecordValidationError(
                    "RecordTree needs same record types "
                    f"_{type(record)}!={self.record_type}"
                )
        return flat_record_dict

    @property
    def data(self) -> record_data:
        """Internally RecordTree is represented as BlobRecord; therefore
        this property in RecordTree shouldn't be used.
        The actual BlobRecord data contains URIs for its leaf node records,
        but this implementation is to be regarded as private.

        Returns:
            record_data: binary encoded json string containing `RECORD_TREE`
        """
        #
        return json.dumps("RECORD_TREE").encode("utf-8")

    @property
    def record_type(self) -> RecordType:
        """Returns either :class:`RecordType.BLOB_TREE` or
        :class:`RecordType.NUMERICAL_TREE`

        Returns:
            RecordType: RecordType.BLOB_TREE or RecordType.NUMERICAL_TREE
        """
        return RecordType.BYTES

    @property
    def flat_record_dict(self) -> Dict[str, RecordGen]:
        """Returns a flattened dictionary, where each `key` represents a path
        and `value` represents the actual Record object. Character `/` is used
        to delineate branches in the original tree in the `key`.
        """
        return self._flat_record_dict


class BlobRecordTree(RecordTree[BlobRecord]):
    """The :class:`BlobRecordTree` is an implementation of the RecordTree class
    that handles a hierarchical representation of BlobRecords.
    """

    @property
    def record_type(self) -> RecordType:
        """Returns :class:`RecordType` set to :class:`RecordType.BLOB_TREE`."""

        return RecordType.BLOB_TREE


class NumericalRecordTree(RecordTree[NumericalRecord]):
    """The :class:`NumericalRecordTree` is an implementation of the RecordTree class
    that handles a hierarchical representation of NumericalRecords.
    """

    @property
    def record_type(self) -> RecordType:
        """Returns :class:`RecordType` set to :class:`RecordType.NUMERICAL_TREE`."""
        return RecordType.NUMERICAL_TREE


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
        """Create RecordCollection instance from the given tuple of records.
        In case of UNIFORM collections the length must be provided.

        Args:
            records: Input tuple of records
            length: The size of collection, which defaults to None
            collection_type: Type of collection, which
                defaults to :py:class:`RecordCollectionType.NON_UNIFORM`

        Raises:
            ValueError: Raises value error in case when: 1) there are no records
                provided, 2) multiple records are given or length is missing when
                type is :class:`RecordCollectionType.UNIFORM`, 3) length of record
                list does not match the length provided in case of
                :class:`RecordCollectionType.NON_UNIFORM` collection,
                4) the record type varies within collection.
        """
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
                    f"Requested length ({length}) does not match "
                    f"the record count ({len(records)})"
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
        """Returns the type of the record(s) within the collection."""
        assert self._records[0].record_type is not None  # mypy needs this
        return self._records[0].record_type

    @property
    def collection_type(self) -> RecordCollectionType:
        """Returns the collection type, which is either
        :class:`RecordCollectionType.NON_UNIFORM` or
        :class:`RecordCollectionType.UNIFORM`.
        """
        return self._collection_type
