import contextlib
import os
import pathlib
from typing import Callable, ContextManager, List

import pytest
from ert_utils import tmp

from ert.data import (
    BlobRecord,
    ExecutableRecordTransformation,
    FileRecordTransformation,
    RecordTransformation,
    RecordTransmitter,
    TarRecordTransformation,
)
from ert.data.record._record import _sync_make_tar


@contextlib.contextmanager
def file_factory_context(tmpdir):
    def file_factory(files: List[str]) -> None:
        for file in files:
            dir_path = pathlib.Path(tmpdir) / file
            dir_path.parent.mkdir(parents=True, exist_ok=True)
            dir_path.touch()

    yield file_factory


@contextlib.contextmanager
def record_factory_context(tmpdir):
    def record_factory(is_dir: bool):
        if is_dir:
            dir_path = pathlib.Path(tmpdir) / "resources" / "test_dir"
            _files = [dir_path / "a.txt", dir_path / "b.txt"]
            with file_factory_context(tmpdir) as file_factory:
                file_factory(_files)
            # Using the private sync version for simplicity in tests:
            tardata = _sync_make_tar(dir_path)
            return BlobRecord(data=tardata)
        else:
            return BlobRecord(data=b"\xF0\x9F\xA6\x89")

    yield record_factory


transformation_params = pytest.mark.parametrize(
    ("transformation_class, location, mime, is_dir, res_files_dumped"),
    (
        (
            FileRecordTransformation,
            "test.blob",
            "application/octet-stream",
            False,
            ["test.blob"],
        ),
        (
            TarRecordTransformation,
            "test_dir",
            "application/octet-stream",
            True,
            ["test_dir/a.txt", "test_dir/b.txt"],
        ),
        (
            ExecutableRecordTransformation,
            "test.blob",
            "application/octet-stream",
            False,
            ["bin/test.blob"],
        ),
    ),
)


@pytest.mark.asyncio
@transformation_params
async def test_atomic_transformation_input(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    transformation_class: RecordTransformation,
    location: str,
    mime: str,
    is_dir: bool,
    res_files_dumped: List[str],
    storage_path,
    tmp_path,
):
    runpath = pathlib.Path(".")
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory, record_factory_context(
        tmp_path
    ) as record_factory, tmp():
        record_in = record_factory(is_dir=is_dir)
        transmitter = record_transmitter_factory(name="trans_custom")
        await transmitter.transmit_record(record_in)
        assert transmitter.is_transmitted()
        transformation = transformation_class()
        record = await transmitter.load()
        await transformation.transform_input(
            record, mime, runpath, pathlib.Path(location)
        )

        for file in res_files_dumped:
            assert os.path.isfile(os.path.join(runpath, file))


@pytest.mark.asyncio
@transformation_params
async def test_atomic_transformation_output(
    record_transmitter_factory_context: ContextManager[
        Callable[[str], RecordTransmitter]
    ],
    transformation_class: RecordTransformation,
    location: str,
    mime: str,
    is_dir: bool,
    res_files_dumped: List[str],
    storage_path,
    tmp_path,
):
    with record_transmitter_factory_context(
        storage_path=storage_path
    ) as record_transmitter_factory, file_factory_context(
        tmp_path
    ) as file_factory, tmp():
        runpath = tmp_path
        file_factory(files=res_files_dumped)
        transmitter = record_transmitter_factory(name="trans_custom")
        assert transmitter.is_transmitted() is False
        transformation = transformation_class()
        if not is_dir:
            location = res_files_dumped[0]

        record = await transformation.transform_output(mime, runpath / location)
        await transmitter.transmit_record(record)
        assert transmitter.is_transmitted()

        blob_record = await transmitter.load()
        assert isinstance(blob_record, BlobRecord)
