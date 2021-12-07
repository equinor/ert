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
    """Ensure relative, and if the location is within a folder create those
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
    """Walk the files under a filepath and encapsulate the file(s)
    in a tar package.

    Args:
        file_path: The path to convert to tar.

    Returns:
        bytes: bytes from in memory tar object.
    """
    executor = futures.ThreadPoolExecutor()
    return await get_event_loop().run_in_executor(executor, _sync_make_tar, file_path)


class RecordTransformation(ABC):
    """:class:`RecordTransformation` is an abstract class that handles
    custom save and load operations on Records to and from disk.
    """

    @abstractmethod
    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        pass

    @abstractmethod
    async def transform_output(self, mime: str, location: Path) -> Record:
        pass


class FileRecordTransformation(RecordTransformation):
    """:class:`FileRecordTransformation` is :class:`RecordTransformation`
    implementation which provides basic Record to disk and disk to Record
    functionality.
    """

    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        """Transforms a Record to disk on the given location via a serializer
        given in the mime type.

        Args:
            record: a record object to save to disk
            mime: mime type to fetch the corresponding serializer
            runpath: the root of the path
            location: filename to save the Record to
        """
        if not isinstance(record, (NumericalRecord, BlobRecord)):
            raise TypeError("Record type must be a NumericalRecord or BlobRecord")

        _prepare_location(runpath, location)
        await _save_record_to_file(record, runpath / location, mime)

    async def transform_output(self, mime: str, location: Path) -> Record:
        """Transfroms a file to Record from the given location via
        a serializer given in the mime type.

        Args:
            mime: mime type to fetch the corresponding serializer
            runpath: the root of the path
            location: filename to load the Record from

        Returns:
            Record: the object is either :class:`BlobRecord` or
                :class:`NumericalRecord`
        """
        return await _load_record_from_file(location, mime)

    async def transform_output_sequence(
        self, mime: str, location: Path
    ) -> Tuple[Record, ...]:
        if mime == "application/octet-stream":
            raise TypeError("Output record types must be NumericalRecord")
        raw_ensrecord = await get_serializer(mime).decode_from_path(location)
        return tuple(NumericalRecord(data=raw_record) for raw_record in raw_ensrecord)


class TarRecordTransformation(RecordTransformation):
    """:class:`TarRecordTransformation` is :class:`RecordTransformation`
    implementation which provides creating a tar object from a given location
    into a BlobRecord :func:`TarRecordTransformation.transform_output` and
    extracting tar object (:class:`BlobRecord`) to the given location.
    """

    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        """Transforms BlobRecord (tar object) to disk, ie. extracting tar object
        on the given location.

        Args:
            record: BlobRecord object, where :func:`BlobRecord.data`
                is the binary tar object
            mime: mime type is ignored in this case
            runpath: the root of the path
            location: directory name to extract the tar object into

        Raises:
            TypeError: Raises when the Record (loaded via transmitter)
                is not BlobRecord
        """
        if not isinstance(record, BlobRecord):
            raise TypeError("Record type must be a BlobRecord")

        with tarfile.open(fileobj=io.BytesIO(record.data), mode="r") as tar:
            _prepare_location(runpath, location)
            tar.extractall(runpath / location)

    async def transform_output(self, mime: str, location: Path) -> Record:
        """Transfroms directory from the given location into a :class:`BlobRecord`
        object.

        Args:
            mime: mime type is ignored
            runpath: the root of the path
            location: directory name to create tar representation from

        Returns:
            Record: returns :class:`BlobRecord` object that is a
                binary representation of the final tar object.
        """
        return BlobRecord(data=await _make_tar(location))


class ExecutableRecordTransformation(RecordTransformation):
    """:class:`ExecutableRecordTransformation` is :class:`RecordTransformation`
    implementation which provides creating an executable file; ie. when
    storing a Record to the file.
    """

    async def transform_input(
        self, record: Record, mime: str, runpath: Path, location: Path
    ) -> None:
        """Transforms a Record to disk on the given location via
        via a serializer given in the mime type. Additionally, it makes
        executable from the file

        Args:
            record: a record object to save to disk that becomes executable
            mime: mime type to fetch the corresponding serializer
            runpath: the root of the path
            location: filename to save the Record to
        """
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
        """Transforms a file to Record from the given location via
        a serializer given in the mime type..

        Args:
            mime: mime type to fetch the corresponding serializer
            runpath: the root of the path
            location: filename to load the Record from

        Returns:
            Record: return object of :class:`BlobRecord` type.
        """
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
