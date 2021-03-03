import contextlib
import json
import os
import pathlib
import pickle
import tempfile
from typing import Callable, ContextManager

import cloudpickle
import pytest
from ert3.data import (
    InMemoryRecordTransmitter,
    SharedDiskRecordTransmitter,
    RecordTransmitter,
)
from tests.utils import tmpdir


@contextlib.contextmanager
def prefect_factory_context():
    tmp_path = tempfile.TemporaryDirectory()
    tmp_storage_path = pathlib.Path(tmp_path.name) / ".shared-storage"
    tmp_storage_path.mkdir(parents=True, exist_ok=True)

    def prefect_storage_factory(name: str) -> SharedDiskRecordTransmitter:
        return SharedDiskRecordTransmitter(
            name=name,
            storage_path=tmp_storage_path,
        )

    try:
        yield prefect_storage_factory
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
        prefect_factory_context,
    ),
)

simple_records = pytest.mark.parametrize(
    ("data_in", "expected_data"),
    (
        ([1, 2, 3], [1.0, 2.0, 3.0]),
        ((1.0, 10.0, 42, 999.0), [1.0, 10.0, 42.0, 999.0]),
        ([], []),
        ([12.0], [12.0]),
        ({1, 2, 3}, [1.0, 2.0, 3.0]),
        ({"a": 0, "b": 1, "c": 2}, {"a": 0.0, "b": 1.0, "c": 2.0}),
        ({0: 10, 100: 0}, {0: 10.0, 100: 0.0}),
    ),
)


@simple_records
@factory_params
def test_simple_record_transmit(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter.transmit(data_in)
        assert transmitter.is_transmitted()
        with pytest.raises(RuntimeError, match="Record already transmitted"):
            transmitter.transmit(data_or_file=[1, 2, 3])


@simple_records
@factory_params
def test_simple_record_transmit_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter.transmit(data_in)

        record = transmitter.load()
        assert record.data == expected_data


@simple_records
@factory_params
@tmpdir(None)
def test_simple_record_transmit_and_dump(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter.transmit(data_in)

        transmitter.dump("record.json")
        with open("record.json") as f:
            expected_data = json.loads(json.dumps(expected_data))
            assert expected_data == json.load(f)


@simple_records
@factory_params
def test_simple_record_transmit_pickle_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    data_in,
    expected_data,
):
    with record_transmitter_factory_context() as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        transmitter.transmit(data_in)
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        record = transmitter.load()

        assert record.data == expected_data
