from abc import ABC, abstractmethod
from pathlib import Path
import tarfile
import io
import stat
import json
from ert import serialization

from ert.data import (
    RecordTransmitter,
    BlobRecord,
    make_tar,
    NumericalRecord,
    NumericalRecordTree,
)
from ert.serialization import get_serializer
from ert.serialization._serializer import _ecl_sum_serializer
from typing import Dict, List, Optional

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
        await transmitter.dump(runpath / location, mime)

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
        await transmitter.dump(base_path / location, mime)

        # post-proccess if neccessary
        path = base_path / location
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC)

    async def transform_output(
        self, transmitter: RecordTransmitter, mime: str, runpath: Path, location: Path
    ) -> None:
        await transmitter.transmit_file(runpath / location, mime)

    class SummaryInputTransformation(RecordTransformation):
        def __init__(self, keys: List[str], sub_path: Optional[str] = None):
            self._keys = keys
            self._sub_path = sub_path

        async def transform_input(
            self,
            transmitter: RecordTransmitter,
            mime: str,
            runpath: Path,
            location: Path,
        ) -> None:
            record = transmitter.load()
            if isinstance(record, BlobRecord):
                sub_records: Dict[str, str] = json.loads(record.data.decode("utf-8"))
                transmitters = await transmitter.get_recordtree_transmitters(
                    sub_records, RecordType.MAPPING_STR_FLOAT, self._sub_path
                )
                for rec_path, sub_transmitter in transmitters.items():
                    rec_name = rec_path.split("/")[-1]
                    sub_transmitter.dump(runpath / rec_name, "application/x-yaml")
                await transmitter.dump(runpath / location, mime)
            else:
                raise TypeError(
                    "The underlying RecordTree needs to be a BlobRecord type!"
                )
            pass

        async def transform_output(
            self,
            transmitter: RecordTransmitter,
            mime: str,
            runpath: Path,
            location: Path,
        ) -> None:
            serializer = get_serializer(mime)
            if isinstance(serializer, _ecl_sum_serializer):
                record_tree = {
                    key: NumericalRecord(
                        data=serializer.decode_from_path(
                            str(runpath / location), key=key
                        )
                    )
                    for key in self._keys
                }
                await transmitter.transmit_record(
                    NumericalRecordTree(record_tree=record_tree)
                )
