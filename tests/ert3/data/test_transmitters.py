import contextlib
import json
import pathlib
import pickle
import tempfile
from typing import Callable, ContextManager

import cloudpickle
import pytest
from utils import tmp

from ert.data import (
    InMemoryRecordTransmitter,
    RecordTransmitter,
    SharedDiskRecordTransmitter,
)


@contextlib.contextmanager
def shared_disk_factory_context():
    tmp_path = tempfile.TemporaryDirectory()
    tmp_storage_path = pathlib.Path(tmp_path.name) / ".shared-storage"
    tmp_storage_path.mkdir(parents=True)

    def shared_disk_factory(name: str) -> SharedDiskRecordTransmitter:
        return SharedDiskRecordTransmitter(
            name=name,
            storage_path=tmp_storage_path,
        )

    try:
        yield shared_disk_factory
    finally:
        tmp_path.cleanup()


@contextlib.contextmanager
def in_memory_factory_context():
    def in_memory_factory(name: str) -> InMemoryRecordTransmitter:
        return InMemoryRecordTransmitter(name=name)

    yield in_memory_factory


factory_params = pytest.mark.parametrize(
    ("record_transmitter_factory_context"),
    (
        in_memory_factory_context,
        shared_disk_factory_context,
    ),
)

simple_records = pytest.mark.parametrize(
    ("data_in", "expected_data", "application_type"),
    (
        ([1, 2, 3], [1, 2, 3], "application/json"),
        ((1.0, 10.0, 42, 999.0), [1.0, 10.0, 42, 999.0], "application/json"),
        ([], [], "application/json"),
        ([12.0], [12.0], "application/json"),
        ({1, 2, 3}, [1, 2, 3], "application/json"),
        ({"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "c": 2}, "application/json"),
        ({0: 10, 100: 0}, {0: 10, 100: 0}, "application/json"),
        ([b"\x00"], [b"\x00"], "application/octet-stream"),
        ([b"\xF0\x9F\xA6\x89"], [b"\xF0\x9F\xA6\x89"], "application/octet-stream"),
    ),
)


@pytest.mark.asyncio
@simple_records
@factory_params
async def test_simple_record_transmit(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
    application_type,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_data(data_in)
        assert transmitter.is_transmitted()
        with pytest.raises(RuntimeError, match="Record already transmitted"):
            await transmitter.transmit_data([1, 2, 3])


@pytest.mark.asyncio
@simple_records
@factory_params
async def test_simple_record_transmit_from_file(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
    application_type,
):
    filename = "record.file"
    with record_transmitter_factory_context() as record_transmitter_factory, tmp():
        transmitter = record_transmitter_factory(name="some_name")
        if application_type == "application/json":
            with open(filename, "w") as f:
                json.dump(expected_data, f)
        else:
            with open(filename, "wb") as f:
                f.write(expected_data[0])
        await transmitter.transmit_file(filename, mime=application_type)
        assert transmitter.is_transmitted()
        with pytest.raises(RuntimeError, match="Record already transmitted"):
            await transmitter.transmit_file(filename, mime=application_type)


@pytest.mark.asyncio
@simple_records
@factory_params
async def test_simple_record_transmit_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
    application_type,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_data(data_in)

        record = await transmitter.load()
        assert record.data == expected_data


@pytest.mark.asyncio
@simple_records
@factory_params
async def test_simple_record_transmit_and_dump(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
    application_type,
):
    with record_transmitter_factory_context() as record_transmitter_factory, tmp():
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_data(data_in)

        await transmitter.dump("record.json")
        if application_type == "application/json":
            with open("record.json") as f:
                assert json.dumps(expected_data) == f.read()
        else:
            with open("record.json", "rb") as f:
                assert expected_data[0] == f.read()


@pytest.mark.asyncio
@simple_records
@factory_params
async def test_simple_record_transmit_pickle_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
    application_type,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        await transmitter.transmit_data(data_in)
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        record = await transmitter.load()

        assert record.data == expected_data


@pytest.mark.asyncio
@factory_params
async def test_load_untransmitted_record(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        with pytest.raises(RuntimeError, match="cannot load untransmitted record"):
            _ = await transmitter.load()


@pytest.mark.asyncio
@factory_params
async def test_dump_untransmitted_record(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        with pytest.raises(RuntimeError, match="cannot dump untransmitted record"):
            await transmitter.dump("some.file")
