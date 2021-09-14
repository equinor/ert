import pytest
import contextlib
import pathlib
import os
from typing import Callable, ContextManager, List

from ert_utils import tmp
from ert.data import (
    RecordTransformation,
    FileRecordTransformation,
    TarRecordTransformation,
    ExecutableRecordTransformation,
    path_to_bytes,
    RecordTransmitter,
    BlobRecord,
)


@contextlib.contextmanager
def record_factory_context(tmpdir):
    def record_factory(is_dir: bool):
        if is_dir:
            dir_path = pathlib.Path(tmpdir) / "resources" / "test_dir"
            dir_path.mkdir(parents=True, exist_ok=True)
            _files = [dir_path / "a.txt", dir_path / "b.txt"]
            for file in _files:
                file.touch()
            return BlobRecord(data=path_to_bytes(dir_path))
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
async def test_atomic_transformation(
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
        await transformation.transform(transmitter, mime, runpath, location)

        for file in res_files_dumped:
            assert os.path.isfile(os.path.join(runpath, file))
