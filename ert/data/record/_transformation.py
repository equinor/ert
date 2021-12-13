from abc import ABC, abstractmethod
from pathlib import Path
import tarfile
import io
import stat
import aiofiles

from ert.data import RecordTransmitter, BlobRecord, make_tar, Record, NumericalRecord
from ert.serialization import get_serializer

_BIN_FOLDER = "bin"


def _prepare_location(base_path: Path, location: Path) -> None:
    """Ensure relative, and if the location is within a folder, create those
    folders."""
    if location.is_absolute():
        raise ValueError(f"location {location} must be relative")
    abs_path = base_path / location
    if not abs_path.parent.exists():
        abs_path.parent.mkdir(parents=True, exist_ok=True)


class RecordTransformation(ABC):
    @abstractmethod
    async def transform_input(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        raise NotImplementedError("not implemented")

    @abstractmethod
    async def transform_output(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        raise NotImplementedError("not implemented")


class FileRecordTransformation(RecordTransformation):
    async def transform_input(
        self,
        transmitter: RecordTransmitter,
        mime: str,
        runpath: Path,
        location: Path,
    ) -> None:
        _prepare_location(runpath, location)
        record = await transmitter.load()
        await _dump(record, runpath / location, mime)

    async def transform_output(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        await transmitter.transmit_file(runpath / location, mime)


class TarRecordTransformation(RecordTransformation):
    async def transform_input(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        record = await transmitter.load()
        if isinstance(record, BlobRecord):
            with tarfile.open(fileobj=io.BytesIO(record.data), mode="r") as tar:
                _prepare_location(runpath, location)
                tar.extractall(runpath / location)
        else:
            raise TypeError("Record needs to be a BlobRecord type!")

    async def transform_output(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        blob_record = BlobRecord(data=await make_tar(runpath / location))
        await transmitter.transmit_record(blob_record)


class ExecutableRecordTransformation(RecordTransformation):
    async def transform_input(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        # pre-make bin folder if neccessary
        base_path = Path(runpath / _BIN_FOLDER)
        base_path.mkdir(parents=True, exist_ok=True)

        # create file(s)
        _prepare_location(base_path, location)
        record = await transmitter.load()
        await _dump(record, base_path / location, mime)

        # post-proccess if neccessary
        path = base_path / location
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC)

    async def transform_output(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        await transmitter.transmit_file(runpath / location, mime)


async def _dump(record: Record, location: Path, mime: str) -> None:
    if isinstance(record, NumericalRecord):
        async with aiofiles.open(str(location), mode="wt", encoding="utf-8") as ft:
            await ft.write(get_serializer(mime).encode(record.data))
    else:
        async with aiofiles.open(str(location), mode="wb") as fb:
            await fb.write(record.data)  # type: ignore
