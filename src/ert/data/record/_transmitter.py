import json
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
from pydantic import StrictBytes, StrictFloat, StrictInt, StrictStr

from ert.serialization import get_serializer

from ._record import (
    BlobRecord,
    BlobRecordTree,
    NumericalRecord,
    NumericalRecordTree,
    Record,
    RecordType,
)

strict_number = Union[StrictInt, StrictFloat]
numerical_record_data = Union[
    List[strict_number],
    Dict[StrictStr, strict_number],
    Dict[StrictInt, strict_number],
]
blob_record_data = StrictBytes
record_data = Union[numerical_record_data, blob_record_data]
transmitter_factory = Callable[..., "RecordTransmitter"]


class RecordTransmitterState(Enum):
    transmitted = auto()
    not_transmitted = auto()


class RecordTransmitterType(Enum):
    in_memory = auto()
    ert_storage = auto()
    shared_disk = auto()


def _unflatten_record_dict(
    flat_record_dict: Union[Dict[str, BlobRecord], Dict[str, NumericalRecord]]
) -> Dict[str, Any]:
    record_dict: Dict[str, Any] = {}
    for rec_path, rec in flat_record_dict.items():
        keys = rec_path.split("/")
        sub_rec_name = keys[-1]
        branch = record_dict
        for key in keys[:-1]:
            if key not in branch:
                branch[key] = {}
            branch = branch[key]
        branch[sub_rec_name] = rec
    return record_dict


class RecordTransmitter:
    """:class:`RecordTransmitter` represents the base class for loading
    and transmitting :class:`Record`. The Transmitter is one of the
    following implementations :class:`SharedDiskRecordTransmitter`,
    :class:`InMemoryRecordTransmitter` or
    `StorageRecordTransmitter`.
    """

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
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _load_blob_record(self) -> BlobRecord:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _get_recordtree_transmitters(
        self,
        trans_records: Dict[str, str],
        record_type: RecordType,
        path: Optional[str] = None,
    ) -> Dict[str, "RecordTransmitter"]:
        raise NotImplementedError("not implemented")

    async def _load_numerical_recordtree(
        self,
        transmitters: Dict[str, "RecordTransmitter"],
    ) -> NumericalRecordTree:
        flat_record_dict = {
            rec_path: await transmitter._load_numerical_record()
            for rec_path, transmitter in transmitters.items()
        }
        record_dict = _unflatten_record_dict(flat_record_dict)
        return NumericalRecordTree(record_dict=record_dict)

    async def _load_blob_recordtree(
        self,
        transmitters: Dict[str, "RecordTransmitter"],
    ) -> BlobRecordTree:
        record_path_dict = {
            rec_path: await transmitter._load_blob_record()
            for rec_path, transmitter in transmitters.items()
        }
        record_dict = _unflatten_record_dict(record_path_dict)
        return BlobRecordTree(record_dict=record_dict)

    async def load(self) -> Record:
        """Loads the transmitted Record to the memory.
        Based on the :class:`RecordType` it creates :class:`BlobRecord`,
        :class:`NumericalRecord`, :class:`NumericalRecordTree` or
        :class:`BlobRecordTree` instance.

        Raises:
            RuntimeError: Raises when the Record was not transmitted yet.
        """
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type == RecordType.NUMERICAL_TREE:
            record = await self._load_blob_record()
            sub_records = json.loads(record.data.decode("utf-8"))
            transmitters = await self._get_recordtree_transmitters(
                sub_records, RecordType.MAPPING_STR_FLOAT
            )
            return await self._load_numerical_recordtree(transmitters)
        elif self._record_type == RecordType.BLOB_TREE:
            record = await self._load_blob_record()
            sub_records = json.loads(record.data.decode("utf-8"))
            transmitters = await self._get_recordtree_transmitters(
                sub_records, RecordType.BYTES
            )
            return await self._load_blob_recordtree(transmitters)
        elif self._record_type != RecordType.BYTES:
            return await self._load_numerical_record()
        return await self._load_blob_record()

    @abstractmethod
    async def _transmit_recordtree(
        self, record: Union[NumericalRecordTree, BlobRecordTree]
    ) -> str:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        raise NotImplementedError("not implemented")

    async def transmit_record(self, record: Record) -> None:
        """Transmits a Record object.

        Args:
            record: Record object needs to be of :class:`BlobRecord`,
                :class:`NumericalRecord`, :class:`NumericalRecordTree`
                or :class:`BlobRecordTree` type.

        Raises:
            RuntimeError: Raises when the Record was already transmitted.
            TypeError: Raises when the Record is of a different type then
                types listed in args.
        """
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if isinstance(record, NumericalRecord):
            uri = await self._transmit_numerical_record(record)
        elif isinstance(record, BlobRecord):
            uri = await self._transmit_blob_record(record)
        elif isinstance(record, (NumericalRecordTree, BlobRecordTree)):
            uri = await self._transmit_recordtree(record)
        else:
            raise TypeError(f"Record type not supported {type(record)}")
        self._set_transmitted_state(uri, record_type=record.record_type)


class SharedDiskRecordTransmitter(RecordTransmitter):
    """:class:`SharedDiskRecordTransmitter` represents :class:`RecordTransmitter`
    implementation that handles transmitting Records to a disk and back.
    """

    _INTERNAL_MIME_TYPE = "application/x-yaml"

    def __init__(self, name: str, storage_path: Path):
        """Creates instance of `SharedDiskRecordTransmitter`.

        Args:
            name: Represents the Record name
            storage_path: Location for handling record-disk operations
        """
        super().__init__(RecordTransmitterType.shared_disk)
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._concrete_key = f"{name}_{uuid.uuid4()}"
        self._storage_uri = self._storage_path / self._concrete_key

    async def _get_recordtree_transmitters(
        self,
        trans_records: Dict[str, str],
        record_type: RecordType,
        path: Optional[str] = None,
    ) -> Dict[str, RecordTransmitter]:
        transmitters: Dict[str, RecordTransmitter] = {}
        for record_path, record_uri in trans_records.items():
            if path is None or path in record_path:
                record_name = record_path.split("/")[-1]
                transmitter = SharedDiskRecordTransmitter(
                    record_name, self._storage_path
                )
                transmitter._set_transmitted_state(record_uri, record_type)
                transmitters[record_path] = transmitter
        return transmitters

    async def _transmit_recordtree(
        self, record: Union[NumericalRecordTree, BlobRecordTree]
    ) -> str:
        data = {}
        for rec_path in record.flat_record_dict:
            rec_key = rec_path.split("/")[-1]
            transmitter = SharedDiskRecordTransmitter(
                name=rec_key,
                storage_path=self._storage_path,
            )
            await transmitter.transmit_record(record.flat_record_dict[rec_path])
            data[rec_path] = transmitter._uri
        await self._transmit_blob_record(
            BlobRecord(data=json.dumps(data).encode("utf-8"))
        )
        return str(self._storage_uri)

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


class InMemoryRecordTransmitter(RecordTransmitter):
    """:class:`InMemoryRecordTransmitter` represents :class:`RecordTransmitter`
    implementation that handles transmitting Records to memory and back.
    """

    def __init__(self, name: str):
        """Creates instance of `InMemoryRecordTransmitter`.

        Args:
            name: Represents the Record name
        """
        super().__init__(RecordTransmitterType.in_memory)
        self._name = name
        self._record: Record
        self._sub_transmitters: Dict[str, RecordTransmitter] = {}

    async def _get_recordtree_transmitters(
        self,
        trans_records: Dict[str, str],
        record_type: RecordType,
        path: Optional[str] = None,
    ) -> Dict[str, RecordTransmitter]:
        transmitters = {
            record_path: self._sub_transmitters[record_path]
            for record_path in trans_records
            if path is None or path in record_path
        }
        return transmitters

    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        self._record = record
        return "in_memory"

    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        self._record = record
        return "in_memory"

    async def _transmit_recordtree(
        self, record: Union[NumericalRecordTree, BlobRecordTree]
    ) -> str:
        data = {}
        for rec_path in record.flat_record_dict:
            rec_key = rec_path.split("/")[-1]
            transmitter = InMemoryRecordTransmitter(name=rec_key)
            await transmitter.transmit_record(record.flat_record_dict[rec_path])
            data[rec_path] = transmitter._uri
            self._sub_transmitters[rec_path] = transmitter
        self._record = BlobRecord(data=json.dumps(data).encode("utf-8"))
        return "in_memory"

    async def _load_numerical_record(self) -> NumericalRecord:
        if not isinstance(self._record, NumericalRecord):
            raise TypeError("loading numerical from blob record")
        return self._record

    async def _load_blob_record(self) -> BlobRecord:
        if not isinstance(self._record, BlobRecord):
            raise TypeError("loading blob from numerical record")
        return self._record
