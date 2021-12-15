import io
import stat
import tarfile
from abc import ABC, abstractmethod
from concurrent import futures
from pathlib import Path
from typing import Tuple

import aiofiles
from ert.data import BlobRecord, NumericalRecord, Record
from ert.serialization import get_serializer
from ert_shared.async_utils import get_event_loop

_BIN_FOLDER = "bin"


def _prepare_location(base_path: Path, location: Path) -> None:
    """Ensure relative, and if the location is within a folder, create those
    folders."""
    if location.is_absolute():
        raise ValueError(f"location {location} must be relative")
    abs_path = base_path / location
    if not abs_path.parent.exists():
        abs_path.parent.mkdir(parents=True, exist_ok=True)


def _sync_make_tar(file_path: Path) -> bytes:
    tar_obj = io.BytesIO()
    with tarfile.open(fileobj=tar_obj, mode="w") as tar:
        tar.add(file_path, arcname="")
    return tar_obj.getvalue()


async def _make_tar(file_path: Path) -> bytes:
    # Walk the files under a filepath and encapsulate them in a tar package.
    executor = futures.ThreadPoolExecutor()
    return await get_event_loop().run_in_executor(executor, _sync_make_tar, file_path)


class RecordTransformation(ABC):
    @abstractmethod
    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        pass

    @abstractmethod
    async def transform_output(self, mime: str, location: Path) -> Record:
        pass


class FileRecordTransformation(RecordTransformation):
    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        if not isinstance(record, (NumericalRecord, BlobRecord)):
            raise TypeError("Record type must be a NumericalRecord or BlobRecord")

        _prepare_location(runpath, location)
        await _save_record_to_file(record, runpath / location, mime)

    async def transform_output(self, mime: str, location: Path) -> Record:
        return await _load_record_from_file(location, mime)

    async def transform_output_sequence(
        self, mime: str, location: Path
    ) -> Tuple[Record, ...]:
        if mime == "application/octet-stream":
            raise TypeError("Output record types must be NumericalRecord")
        raw_ensrecord = await get_serializer(mime).decode_from_path(location)
        return tuple(NumericalRecord(data=raw_record) for raw_record in raw_ensrecord)


class TarRecordTransformation(RecordTransformation):
    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        if not isinstance(record, BlobRecord):
            raise TypeError("Record type must be a BlobRecord")

        with tarfile.open(fileobj=io.BytesIO(record.data), mode="r") as tar:
            _prepare_location(runpath, location)
            tar.extractall(runpath / location)

    async def transform_output(self, mime: str, location: Path) -> Record:

        return BlobRecord(data=await _make_tar(location))


class ExecutableRecordTransformation(RecordTransformation):
    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        if not isinstance(record, BlobRecord):
            raise TypeError("Record type must be a BlobRecord")

        # pre-make bin folder if necessary
        base_path = Path(runpath / _BIN_FOLDER)
        base_path.mkdir(parents=True, exist_ok=True)

        # create file(s)
        _prepare_location(base_path, location)
        await _save_record_to_file(record, base_path / location, mime)

        # post-process if necessary
        path = base_path / location
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC)

    async def transform_output(self, mime: str, location: Path) -> Record:
        return await _load_record_from_file(location, mime)


async def _load_record_from_file(
    file: Path,
    mime: str,
) -> Record:
    if mime == "application/octet-stream":
        async with aiofiles.open(str(file), mode="rb") as fb:
            contents_b: bytes = await fb.read()
            return BlobRecord(data=contents_b)
    else:
        serializer = get_serializer(mime)
        _record_data = await serializer.decode_from_path(file)
        return NumericalRecord(data=_record_data)


async def _save_record_to_file(record: Record, location: Path, mime: str) -> None:
    if isinstance(record, NumericalRecord):
        async with aiofiles.open(str(location), mode="wt", encoding="utf-8") as ft:
            await ft.write(get_serializer(mime).encode(record.data))
    else:
        async with aiofiles.open(str(location), mode="wb") as fb:
            await fb.write(record.data)  # type: ignore
