import asyncio
import json
import shutil
import typing
import uuid
from abc import abstractmethod
from enum import Enum, auto
from functools import partial, wraps
from pathlib import Path
from typing import Awaitable, List, Mapping, Tuple, Union

import aiofiles
from aiofiles.os import wrap
from pydantic import BaseModel, root_validator


_copy = wrap(shutil.copy)


class _DataElement(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


def _build_record_index(data):
    if isinstance(data, Mapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class Record(_DataElement):
    data: Union[List[float], Mapping[int, float], Mapping[str, float], List[bytes]]
    index: Union[Tuple[int, ...], Tuple[str, ...]]

    def __init__(self, *, data, index=None, **kwargs):
        if index is None:
            index = _build_record_index(data)
        super().__init__(data=data, index=index, **kwargs)

    @root_validator
    def ensure_consistent_index(cls, record):
        assert "data" in record and "index" in record
        norm_record_index = _build_record_index(record["data"])
        assert norm_record_index == record["index"]
        return record


class EnsembleRecord(_DataElement):
    records: Tuple[Record, ...]
    ensemble_size: int

    def __init__(self, *, records, ensemble_size=None, **kwargs):
        if ensemble_size == None:
            ensemble_size = len(records)
        super().__init__(records=records, ensemble_size=ensemble_size, **kwargs)

    @root_validator
    def ensure_consistent_ensemble_size(cls, ensemble_record):
        assert "records" in ensemble_record and "ensemble_size" in ensemble_record
        assert len(ensemble_record["records"]) == ensemble_record["ensemble_size"]
        return ensemble_record


class MultiEnsembleRecord(_DataElement):
    ensemble_records: Mapping[str, EnsembleRecord]
    ensemble_size: int
    record_names: Tuple[str, ...]

    def __init__(
        self, *, ensemble_records, record_names=None, ensemble_size=None, **kwargs
    ):
        if record_names is None:
            record_names = list(ensemble_records.keys())
        if ensemble_size is None:
            first_record = ensemble_records[record_names[0]]
            try:
                ensemble_size = first_record.ensemble_size
            except AttributeError:
                ensemble_size = len(first_record["records"])

        super().__init__(
            ensemble_records=ensemble_records,
            ensemble_size=ensemble_size,
            record_names=record_names,
            **kwargs,
        )

    @root_validator
    def ensure_consistent_ensemble_size(cls, multi_ensemble_record):
        ensemble_size = multi_ensemble_record["ensemble_size"]
        for ensemble_record in multi_ensemble_record["ensemble_records"].values():
            if ensemble_size != ensemble_record.ensemble_size:
                raise AssertionError("Inconsistent ensemble record size")
        return multi_ensemble_record

    @root_validator
    def ensure_consistent_record_names(cls, multi_ensemble_record):
        assert "record_names" in multi_ensemble_record
        record_names = tuple(multi_ensemble_record["ensemble_records"].keys())
        assert multi_ensemble_record["record_names"] == record_names
        return multi_ensemble_record

    def __len__(self):
        return len(self.record_names)


class RecordTransmitterState(Enum):
    transmitted = auto()
    not_transmitted = auto()


class RecordTransmitterType(Enum):
    in_memory = auto()
    ert_storage = auto()
    shared_disk = auto()


class RecordTransmitter:
    def __init__(self, type_: RecordTransmitterType):
        self._type = type_

        # TODO: implement state machine?
        self._state = RecordTransmitterState.not_transmitted

    @abstractmethod
    def stream(self) -> asyncio.StreamReader:
        # maybe private? Doesn't really make sense to give the user a stream?
        pass

    def set_transmitted(self):
        self._state = RecordTransmitterState.transmitted

    def is_transmitted(self):
        return self._state == RecordTransmitterState.transmitted

    @abstractmethod
    async def dump(
        self, location: Path, format: str = "json"
    ) -> None:  # Should be RecordReference ?
        # the result of this awaitable will be set to a RecordReference
        # that has the folder into which this record was dumped
        pass

    @abstractmethod
    async def load(self, format: str = "json") -> "asyncio.Future[Record]":
        pass

    @abstractmethod
    async def transmit(
        self,
        data_or_file: typing.Union[
            Path,
            Union[List[float], Mapping[int, float], Mapping[str, float], List[bytes]],
        ],
    ) -> None:
        pass


class SharedDiskRecordTransmitter(RecordTransmitter):
    TYPE: RecordTransmitterType = RecordTransmitterType.shared_disk

    def __init__(self, name: str, storage_path: Path):
        super().__init__(type_=self.TYPE)
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._concrete_key = f"{name}_{uuid.uuid4()}"
        self._uri: typing.Optional[str] = None

    def set_transmitted(self, uri: Path):
        super().set_transmitted()
        self._uri = str(uri)

    async def transmit(
        self,
        data_or_file: typing.Union[
            Path, List[float], Mapping[int, float], Mapping[str, float], List[bytes]
        ],
        mime="text/json",
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if isinstance(data_or_file, Path) or isinstance(data_or_file, str):
            async with aiofiles.open(str(data_or_file), mode="r") as f:
                contents = await f.read()
                record = Record(data=json.loads(contents))
        else:
            record = Record(data=data_or_file)

        storage_uri = self._storage_path / self._concrete_key
        async with aiofiles.open(storage_uri, mode="w") as f:
            if mime == "text/json":
                contents = json.dumps(record.data)
                await f.write(contents)
            elif mime == "application/x-python-code":
                # XXX: An opaque record is a list of bytes... yes
                # sonso or dan or jond: do something about this
                await f.write(record.data[0].decode())
            else:
                raise ValueError(f"unsupported mime {mime}")
        self.set_transmitted(storage_uri)

    async def load(self) -> Record:
        if self._state != RecordTransmitterState.transmitted:
            raise RuntimeError("cannot load untransmitted record")
        async with aiofiles.open(str(self._uri)) as f:
            contents = await f.read()
            return Record(data=json.loads(contents))

    # TODO: should use Path
    async def dump(self, location: str):
        if self._state != RecordTransmitterState.transmitted:
            raise RuntimeError("cannot dump untransmitted record")
        await _copy(self._uri, location)


class InMemoryRecordTransmitter(RecordTransmitter):
    TYPE: RecordTransmitterType = RecordTransmitterType.in_memory

    def __init__(self, name: str):
        super().__init__(type_=self.TYPE)
        self._name = name
        self._record = None

    def set_transmitted(self, record: Record):
        super().set_transmitted()
        self._record = record

    async def transmit(
        self,
        data_or_file: typing.Union[
            Path, List[float], Mapping[int, float], Mapping[str, float], List[bytes]
        ],
        mime="text/json",
    ):
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if isinstance(data_or_file, Path) or isinstance(data_or_file, str):
            async with aiofiles.open(data_or_file) as f:
                contents = await f.read()
                record = Record(data=json.loads(contents))
        else:
            record = Record(data=data_or_file)
        self.set_transmitted(record=record)

    async def load(self):
        return self._record

    # TODO: should use Path
    async def dump(self, location: str, format: str = "text/json"):
        if format is None:
            format = "text/json"
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        async with aiofiles.open(location, mode="w") as f:
            if format == "text/json":
                await f.write(json.dumps(self._record.data))
            elif format == "application/x-python-code":
                # XXX: An opaque record is a list of bytes... yes
                # sonso or dan or jond: do something about this
                await f.write(self._record.data[0].decode())
            else:
                raise ValueError(f"unsupported mime {format}")
