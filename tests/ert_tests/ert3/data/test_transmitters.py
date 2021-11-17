import contextlib
import json
import pathlib
import pickle
import tempfile
from typing import Callable, ContextManager

import aiofiles
import cloudpickle
import pytest
from ert_utils import tmp

import ert
from ert.data import BlobRecord, NumericalRecord, RecordTransmitter

simple_records = pytest.mark.parametrize(
    ("record_in", "expected_data"),
    (
        (
            NumericalRecord(data=[1, 2, 3]),
            [1, 2, 3],
        ),
        (
            NumericalRecord(data=(1.0, 10.0, 42.0, 999.0)),
            [1.0, 10.0, 42.0, 999.0],
        ),
        (
            NumericalRecord(data=[]),
            [],
        ),
        (
            NumericalRecord(data=[12.0]),
            [12.0],
        ),
        (
            NumericalRecord(data={1, 2, 3}),
            [1, 2, 3],
        ),
        (
            NumericalRecord(data={"a": 0, "b": 1, "c": 2}),
            {"a": 0, "b": 1, "c": 2},
        ),
        (
            NumericalRecord(data={0: 10, 100: 0}),
            {0: 10, 100: 0},
        ),
        (
            BlobRecord(data=b"\x00"),
            b"\x00",
        ),
        (
            BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            b"\xF0\x9F\xA6\x89",
        ),
    ),
)

mime_types = pytest.mark.parametrize(
    ("mime_type"),
    tuple(("application/octet-stream",) + ert.serialization.registered_types()),
)


@pytest.mark.asyncio
@simple_records
async def test_simple_record_transmit(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    record_in,
    expected_data,
    storage_path,
):
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_record(record_in)
        assert transmitter.is_transmitted()
        with pytest.raises(RuntimeError, match="Record already transmitted"):
            await transmitter.transmit_record(NumericalRecord(data=[1, 2, 3]))


@pytest.mark.asyncio
@simple_records
@mime_types
async def test_simple_record_transmit_from_file(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    record_in,
    expected_data,
    mime_type,
    storage_path,
):
    if isinstance(record_in, BlobRecord) and mime_type != "application/octet-stream":
        pytest.skip(f"unsupported serialization of opaque record to {mime_type}")
    if (
        isinstance(record_in, NumericalRecord)
        and mime_type == "application/octet-stream"
    ):
        pytest.skip(f"unsupported serialization of num record to {mime_type}")
    filename = "record.file"
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory, tmp():
        transmitter = record_transmitter_factory(name="some_name")
        if mime_type == "application/octet-stream":
            async with aiofiles.open(filename, "wb") as fb:
                await fb.write(expected_data)
        else:
            await ert.serialization.get_serializer(mime_type).encode_to_path(
                expected_data, filename
            )
        await transmitter.transmit_file(filename, mime=mime_type)
        assert transmitter.is_transmitted()
        with pytest.raises(RuntimeError, match="Record already transmitted"):
            await transmitter.transmit_file(filename, mime=mime_type)


@pytest.mark.asyncio
@simple_records
async def test_simple_record_transmit_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    record_in,
    expected_data,
    storage_path,
):
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_record(record_in)

        record = await transmitter.load()
        assert record.data == expected_data


@pytest.mark.asyncio
@simple_records
@mime_types
async def test_simple_record_transmit_and_dump(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    record_in,
    expected_data,
    mime_type,
    storage_path,
):
    if isinstance(record_in, BlobRecord) and mime_type != "application/octet-stream":
        pytest.skip(f"unsupported serialization of opaque record to {mime_type}")
    if (
        isinstance(record_in, NumericalRecord)
        and mime_type == "application/octet-stream"
    ):
        pytest.skip(f"unsupported serialization of num record to {mime_type}")
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory, tmp():
        transmitter = record_transmitter_factory(name="some_name")
        await transmitter.transmit_record(record_in)

        await transmitter.dump("record", mime_type)
        if mime_type == "application/octet-stream":
            with open("record", "rb") as f:
                assert expected_data == f.read()
        else:
            with open("record", "rt", encoding="utf-8") as f:
                assert (
                    ert.serialization.get_serializer(mime_type).encode(expected_data)
                    == f.read()
                )


@pytest.mark.asyncio
@simple_records
async def test_simple_record_transmit_pickle_and_load(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    record_in,
    expected_data,
    storage_path,
):
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        await transmitter.transmit_record(record_in)
        transmitter = pickle.loads(cloudpickle.dumps(transmitter))
        record = await transmitter.load()

        assert record.data == expected_data


@pytest.mark.asyncio
async def test_load_untransmitted_record(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
):
    with record_transmitter_factory_context(
        storage_path=None
    ) as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        with pytest.raises(RuntimeError, match="cannot load untransmitted record"):
            _ = await transmitter.load()


@pytest.mark.asyncio
async def test_dump_untransmitted_record(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
):
    with record_transmitter_factory_context(
        storage_path=None
    ) as record_transmitter_factory:
        transmitter = record_transmitter_factory(name="some_name")
        with pytest.raises(RuntimeError, match="cannot dump untransmitted record"):
            await transmitter.dump("some.file", "text/whatever")
