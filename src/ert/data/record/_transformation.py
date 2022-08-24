import io
import stat
import tarfile
from abc import ABC, abstractmethod
from concurrent import futures
from enum import Flag, auto
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from ecl.summary import EclSum

from ert.async_utils import get_event_loop
from ert.data import BlobRecord, NumericalRecord, NumericalRecordTree, Record
from ert.exceptions import FileExistsException
from ert.serialization import get_serializer, has_serializer

_BIN_FOLDER = "bin"


class TransformationDirection(Flag):
    """Defines how transformations map from files to records, and/or the inverse."""

    NONE = auto()
    """Transformation cannot transform in any direction."""
    FROM_RECORD = auto()
    """Transformation is able to transform from a record to e.g. a file."""
    TO_RECORD = auto()
    """Transformation is able to transform to a record from e.g. a file."""
    BIDIRECTIONAL = FROM_RECORD | TO_RECORD
    """Transformation is able to transform in both directions."""

    @classmethod
    def from_direction(cls, direction_string: str) -> "TransformationDirection":
        """Return a direction from string."""
        if direction_string == "from_record":
            return cls.FROM_RECORD
        elif direction_string == "to_record":
            return cls.TO_RECORD
        elif direction_string == "bidirectional":
            return cls.BIDIRECTIONAL
        elif direction_string == "none":
            return cls.NONE
        raise ValueError(f"unknown TransformationDirection for '{direction_string}'")

    def __str__(self) -> str:
        if self == self.__class__.FROM_RECORD:
            return "from_record"
        elif self == self.__class__.TO_RECORD:
            return "to_record"
        elif self == self.__class__.BIDIRECTIONAL:
            return "bidirectional"
        elif self == self.__class__.NONE:
            return "none"
        raise ValueError


def _prepare_location(root_path: Path, location: Path) -> None:
    """Ensure relative, and if the location is within a folder create those
    folders."""
    if location.is_absolute():
        raise ValueError(f"location {location} must be relative")
    abs_path = root_path / location
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


class TransformationType(Flag):
    """:class:`TransformationType` indicates what kind of record a transformation is
    likely to produce."""

    BINARY = auto()
    """Indicates transformation will likely produce binary data."""
    NUMERICAL = auto()
    """Indicates transformation will likely produce numerical data."""

    def __str__(self) -> str:
        if self & TransformationType.BINARY:
            return "binary"
        return "numerical"


class RecordTransformation(ABC):
    """:class:`RecordTransformation` is an abstract class that handles
    custom save and load operations on Records to and from disk.

    The direction parameter can be used to point the transformation in a
    specific direction. This direction will then be compared to the
    transformation's DIRECTION, which constraints the transformation to a
    specific direction, no direction or both. This aids validation where
    direction is known, and actual transformation is deferred. E.g. if a
    :class:`EclSumTransformation` is created with the
    :class:`TransformationDirection.FROM_RECORD`, only transformations from
    a record will be allowed.
    """

    DIRECTION = TransformationDirection.NONE
    """Indicates in what direction(s) this transformation can transform. When a new
    transformation instance is created, its ``direction`` parameter will be compared to
    this value. See :class:`TransformationDirection`."""

    def __init__(
        self,
        direction: Optional[TransformationDirection] = None,
    ) -> None:
        super().__init__()
        if not direction:
            direction = self.DIRECTION
        if direction not in self.DIRECTION:
            raise ValueError(f"{self} cannot transform in direction: {str(direction)}")
        self.direction = direction

    @abstractmethod
    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        pass

    @abstractmethod
    async def to_record(self, root_path: Path = Path()) -> Record:
        pass

    @property
    @abstractmethod
    def type(self) -> TransformationType:
        pass


class FileTransformation(RecordTransformation):
    """:class:`FileTransformation` is a :class:`RecordTransformation` serving as base
    for all transformations bound to files.
    """

    MIME = "application/octet-stream"
    """The mime associated with this transformation."""

    def __init__(
        self,
        location: Path,
        mime: Optional[str] = None,
        direction: Optional[TransformationDirection] = None,
    ) -> None:
        """
        Args:
            location: The path of the file to which this transformation is bound.
            mime: the media type of the file to which this transformation is bound.
            direction: the :class:`TransformationDirection` of this
                transformation indicating in which direction this transformation will
                be used.
        """
        super().__init__(direction)
        if not mime:
            mime = self.MIME
        self.mime = mime
        self.location = location
        self._type = self._infer_record_type()

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        raise NotImplementedError("not implemented")

    async def to_record(self, root_path: Path = Path()) -> Record:
        raise NotImplementedError("not implemented")

    def _infer_record_type(self) -> TransformationType:
        if has_serializer(self.mime) or self.mime == EclSumTransformation.MIME:
            return TransformationType.NUMERICAL
        return TransformationType.BINARY

    @property
    def type(self) -> TransformationType:
        """Return the type of this transformation. The type is inferred by
        looking at whether 1) the mime is (de)serializable, and/or 2) the
        transformation has a known type.
        """
        return self._type


class CopyTransformation(FileTransformation):
    """:class:`CopyTransformation` is a :class:`FileTransformation` that copies files
    using a blob record intermediary.
    """

    DIRECTION = TransformationDirection.BIDIRECTIONAL
    """Files can be copied both from a record and to a record."""

    MIME = "application/octet-stream"
    """A file is always treated as a stream of bytes."""

    def __init__(
        self,
        location: Path,
        direction: Optional[TransformationDirection] = None,
    ) -> None:
        """
        Args:
            location: The path of the file to which this transformation is bound.
            direction: the :class:`TransformationDirection` of this
                transformation indicating in which direction this transformation will
                be used.
        """
        super().__init__(location, self.MIME, direction)
        if location.is_dir():
            raise RuntimeError(
                f"cannot copy directory {location}: use the 'directory' transformation "
                + "instead"
            )

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        """Copies a Record to disk based on the location associated with this
        transformation, location rooted to ``root_path`` if supplied.

        Args:
            record: a record object to copy to disk
            root_path: an optional root of the location associated with this
                transformation
        """
        if not isinstance(record, BlobRecord):
            raise TypeError(f"Record copied to {self.location}, must be a BlobRecord")

        _prepare_location(root_path, self.location)
        await _save_record_to_file(record, root_path / self.location, self.MIME)

    async def to_record(self, root_path: Path = Path()) -> Record:
        """Copies the location associated with this transformation to a record.

        Args:
            root_path: an optional root of the location associated with this
                transformation

        Returns:
            Record: a :class:`BlobRecord`
        """
        path = root_path / self.location
        if path.is_dir():
            raise RuntimeError(
                f"cannot copy directory {path}: use the 'directory' "
                + "transformation instead"
            )
        return await _load_record_from_file(path, self.MIME)

    @property
    def type(self) -> TransformationType:
        """The CopyTransformation always has a binary type."""
        return TransformationType.BINARY


class SerializationTransformation(FileTransformation):
    """:class:`SerializationTransformation` is :class:`FileTransformation`
    implementation which provides basic Record to disk and disk to Record
    functionality.
    """

    DIRECTION = TransformationDirection.BIDIRECTIONAL
    """This transformation can by default transform bidirectionally."""

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        """Transforms a Record to disk based on the location and MIME type associated
        with this transformation, location rooted to ``root_path`` if supplied.

        Args:
            record: a record object to save to disk
            root_path: an optional root of the location associated with this
                transformation
        """
        if not isinstance(record, (NumericalRecord, BlobRecord)):
            raise TypeError("Record type must be a NumericalRecord or BlobRecord")

        _prepare_location(root_path, self.location)
        await _save_record_to_file(record, root_path / self.location, self.mime)

    async def to_record(self, root_path: Path = Path()) -> Record:
        """Transforms the location associated with this transformation to a record.

        Args:
            root_path: an optional root of the location associated with this
                transformation

        Returns:
            Record: the object is either :class:`BlobRecord` or
                :class:`NumericalRecord`
        """
        return await _load_record_from_file(root_path / self.location, self.mime)


class TarTransformation(FileTransformation):
    """:class:`TarTransformation` is a :class:`FileTransformation`
    implementation which provides creating a tar object from a given location
    into a BlobRecord :func:`TarTransformation.to_record` and
    extracting tar object (:class:`BlobRecord`) to the given location.
    """

    MIME = "application/x-directory"

    DIRECTION = TransformationDirection.BIDIRECTIONAL
    """This transformation can by default transform bidirectionally."""

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        """Transforms BlobRecord (tar object) to disk, ie. extracting tar object
        on the given location.

        Args:
            record: BlobRecord object, where :func:`BlobRecord.data`
                is the binary tar object
            root_path: an optional root of the location associated with this
                transformation

        Raises:
            TypeError: Raises when the Record (loaded via transmitter)
                is not BlobRecord
        """
        if not isinstance(record, BlobRecord):
            raise TypeError("Record type must be a BlobRecord")

        with tarfile.open(fileobj=io.BytesIO(record.data), mode="r") as tar:
            _prepare_location(root_path, self.location)
            tar.extractall(root_path / self.location)

    async def to_record(self, root_path: Path = Path()) -> Record:
        """Transfroms directory from the given location into a :class:`BlobRecord`
        object.

        Args:
            root_path: an optional root of the location associated with this
                transformation

        Returns:
            Record: returns :class:`BlobRecord` object that is a
                binary representation of the final tar object.
        """
        return BlobRecord(data=await _make_tar(root_path / self.location))


class ExecutableTransformation(SerializationTransformation):
    """:class:`ExecutableTransformation` is :class:`SerializationTransformation`
    implementation which provides creating an executable file; ie. when
    storing a Record to the file.
    """

    DIRECTION = TransformationDirection.BIDIRECTIONAL
    """This transformation can by default transform bidirectionally."""

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        """Transforms a Record to the location associated with this transformation,
        with ``root_path / "bin"`` serving as location root if supplied. Additionally,
        this transformation sets the executable bit on the file on disk.

        Args:
            record: a record object to save to disk that becomes executable
            root_path: an optional root of the location associated with this
                transformation
        """
        if not isinstance(record, BlobRecord):
            raise TypeError("Record type must be a BlobRecord")

        # pre-make bin folder if necessary
        root_path = Path(root_path / _BIN_FOLDER)
        root_path.mkdir(parents=True, exist_ok=True)

        # create file(s)
        _prepare_location(root_path, self.location)
        await _save_record_to_file(record, root_path / self.location, self.mime)

        # post-process if necessary
        path = root_path / self.location
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC)

    async def to_record(self, root_path: Path = Path()) -> Record:
        """Transforms an executable file from the location associated with this
        transformation, to a record.

        Args:
            root_path: an optional root of the location associated with this
                transformation

        Returns:
            Record: return object of :class:`BlobRecord` type.
        """
        return await _load_record_from_file(
            root_path / self.location, "application/octet-stream"
        )


async def _load_record_from_file(file: Path, mime: str) -> Record:
    if mime == "application/octet-stream":
        async with aiofiles.open(str(file), mode="rb") as fb:
            contents_b: bytes = await fb.read()
            return BlobRecord(data=contents_b)
    else:
        serializer = get_serializer(mime)
        _record_data = await serializer.decode_from_path(file)
        return NumericalRecord(data=_record_data)


async def _save_record_to_file(record: Record, location: Path, mime: str) -> None:
    if location.exists():
        raise FileExistsException(f"transformation failed: {location} exists")
    if isinstance(record, NumericalRecord):
        async with aiofiles.open(str(location), mode="wt", encoding="utf-8") as ft:
            await ft.write(get_serializer(mime).encode(record.data))
    else:
        async with aiofiles.open(str(location), mode="wb") as fb:
            assert isinstance(record.data, bytes)  # mypy
            await fb.write(record.data)


class TreeSerializationTransformation(SerializationTransformation):
    """
    Write all leaf records of a NumericalRecordTree to individual files.
    """

    DIRECTION = TransformationDirection.FROM_RECORD
    """This transformation can by default only transform from records."""

    def __init__(
        self,
        location: Path,
        mime: Optional[str] = None,
        sub_path: Optional[str] = None,
        direction: Optional[TransformationDirection] = None,
    ):
        if sub_path is not None:
            raise NotImplementedError("Extracting sub-trees not implemented")
        super().__init__(location, mime, direction)

    async def from_record(self, record: Record, root_path: Path = Path()) -> None:
        if not isinstance(record, NumericalRecordTree):
            raise TypeError("Only NumericalRecordTrees can be transformed.")
        for key, leaf_record in record.flat_record_dict.items():
            location_key = f"{key}-{self.location}"
            path = root_path / location_key
            if path.exists():
                raise FileExistsException(f"writing tree to file failed: {path} exists")
            await get_serializer(self.mime).encode_to_path(leaf_record.data, path=path)

    async def to_record(self, root_path: Path = Path()) -> Record:
        raise NotImplementedError


class EclSumTransformation(FileTransformation):
    """Transform binary output from Eclipse into a NumericalRecordTree."""

    MIME = "application/x-eclipse.summary"

    DIRECTION = TransformationDirection.TO_RECORD
    """This transformation can by default only transform to a record."""

    def __init__(
        self,
        location: Path,
        smry_keys: List[str],
        direction: TransformationDirection = TransformationDirection.TO_RECORD,  # noqa  # pylint: disable=C0301
    ):
        """
        Args:
            location: Path location of eclipse load case, passed as load_case to EclSum.
            smry_keys: List (non-empty) of Eclipse summary vectors (must be present) to
                include when transforming from Eclipse binary files. Wildcards are not
                supported.
            direction: the direction in which we intend to transform.
        """
        super().__init__(location, self.MIME, direction)
        if not smry_keys:
            raise ValueError("smry_keys must be non-empty")
        self._smry_keys = smry_keys

    async def to_record(self, root_path: Path = Path()) -> Record:
        executor = futures.ThreadPoolExecutor()
        record_dict = await get_event_loop().run_in_executor(
            executor,
            _sync_eclsum_to_record,
            root_path / self.location,
            self._smry_keys,
        )
        return NumericalRecordTree(record_dict=record_dict)

    @property
    def smry_keys(self) -> List[str]:
        return self._smry_keys


def _sync_eclsum_to_record(
    location: Path, smry_keys: List[str]
) -> Dict[str, NumericalRecord]:
    eclsum = EclSum(str(location))
    record_dict = {}
    for key in smry_keys:
        record_dict[key] = NumericalRecord(
            data=dict(zip(map(str, eclsum.dates), map(float, eclsum.numpy_vector(key))))
        )
    return record_dict
