import shutil
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
import aiofiles

# Type hinting for wrap must be turned off until (1) is resolved.
# (1) https://github.com/Tinche/aiofiles/issues/8
from aiofiles.os import wrap  # type: ignore
from pydantic import (
    StrictBytes,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from ert.serialization import get_serializer

from ._record import Record, RecordType, NumericalRecord, BlobRecord

_copy = wrap(shutil.copy)

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
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _load_blob_record(self) -> BlobRecord:
        raise NotImplementedError("not implemented")

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type != RecordType.BYTES:
            return await self._load_numerical_record()
        return await self._load_blob_record()

    @abstractmethod
    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        raise NotImplementedError("not implemented")

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
            _record_data = await serializer.decode_from_path(file)
            num_record = NumericalRecord(data=_record_data)
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
        if not isinstance(self._record, NumericalRecord):
            raise TypeError("loading numerical from blob record")
        return self._record

    async def _load_blob_record(self) -> BlobRecord:
        if not isinstance(self._record, BlobRecord):
            raise TypeError("loading blob from numerical record")
        return self._record
