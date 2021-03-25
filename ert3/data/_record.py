import json
import shutil
import typing
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import aiofiles
from aiofiles.os import wrap  # type: ignore
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

    def _set_transmitted(self):
        self._state = RecordTransmitterState.transmitted

    def is_transmitted(self):
        return self._state == RecordTransmitterState.transmitted

    @abstractmethod
    async def dump(
        self, location: Path, mime: str = "text/json"
    ) -> None:  # Should be RecordReference ?
        # the result of this awaitable will be set to a RecordReference
        # that has the folder into which this record was dumped
        pass

    @abstractmethod
    async def load(self) -> Record:
        pass

    @abstractmethod
    async def transmit_data(
        self,
        data: Union[List[float], Mapping[int, float], Mapping[str, float], List[bytes]],
        mime="text/json",
    ) -> None:
        pass

    @abstractmethod
    async def transmit_file(
        self,
        file: Path,
        mime="text/json",
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
        super()._set_transmitted()
        self._uri = str(uri)

    async def _transmit(self, record: Record, mime: str):
        storage_uri = self._storage_path / self._concrete_key
        async with aiofiles.open(storage_uri, mode="w") as f:
            if mime == "text/json":
                contents = json.dumps(record.data)
                await f.write(contents)
            elif mime == "application/x-python-code":
                if isinstance(record.data, list) and isinstance(record.data[0], bytes):
                    await f.write(record.data[0].decode())
                else:
                    raise TypeError(f"unexpected record data type {type(record.data)}")
            else:
                raise ValueError(f"unsupported mime {mime}")
        self.set_transmitted(storage_uri)

    async def transmit_data(
        self,
        data: Union[List[float], Mapping[int, float], Mapping[str, float], List[bytes]],
        mime="text/json",
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        return await self._transmit(record, mime)

    async def transmit_file(
        self,
        file: Path,
        mime="text/json",
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("File already transmitted")
        async with aiofiles.open(str(file), mode="r") as f:
            contents = await f.read()
            record = Record(data=json.loads(contents))
        return await self._transmit(record, mime)

    async def load(self) -> Record:
        if self._state != RecordTransmitterState.transmitted:
            raise RuntimeError("cannot load untransmitted record")
        async with aiofiles.open(str(self._uri)) as f:
            contents = await f.read()
            return Record(data=json.loads(contents))

    # TODO: should use Path
    async def dump(self, location: Path, mime: str = "text/json") -> None:
        if self._state != RecordTransmitterState.transmitted:
            raise RuntimeError("cannot dump untransmitted record")
        await _copy(self._uri, str(location))


class InMemoryRecordTransmitter(RecordTransmitter):
    TYPE: RecordTransmitterType = RecordTransmitterType.in_memory
    # TODO: this field should be Record, but that does not work until
    # https://github.com/cloudpipe/cloudpickle/issues/403 has been released.
    _data: Optional[Any] = None
    _index: Optional[Any] = None

    def __init__(self, name: str):
        super().__init__(type_=self.TYPE)
        self._name = name

    def set_transmitted(self, record: Record):
        super()._set_transmitted()
        self._data = record.data
        self._index = record.index

    @abstractmethod
    async def transmit_data(
        self,
        data: Union[List[float], Mapping[int, float], Mapping[str, float], List[bytes]],
        mime="text/json",
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        self.set_transmitted(record=record)

    @abstractmethod
    async def transmit_file(
        self,
        file: Path,
        mime="text/json",
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        async with aiofiles.open(str(file)) as f:
            contents = await f.read()
            record = Record(data=json.loads(contents))
        self.set_transmitted(record=record)

    async def load(self):
        return Record(data=self._data, index=self._index)

    async def dump(self, location: Path, mime: str = "text/json"):
        if mime is None:
            mime = "text/json"
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        if self._data is None:
            raise ValueError("Cannot dump Record with no data")

        async with aiofiles.open(str(location), mode="w") as f:
            if mime == "text/json":
                await f.write(json.dumps(self._data))
            elif mime == "application/x-python-code":
                if isinstance(self._data, list) and isinstance(self._data[0], bytes):
                    await f.write(self._data[0].decode())
                else:
                    raise TypeError(f"unexpected record data type {type(self._data)}")
            else:
                raise ValueError(f"unsupported mime {mime}")
