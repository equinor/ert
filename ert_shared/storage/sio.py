import os
import sys
from pathlib import Path
from typing import Generator, Optional
from pydantic import BaseModel
import numpy as np

import struct


# From ert/enkf/fs_driver.hpp
FS_MAGIC_ID = 123998
CURRENT_FS_VERSION = 107  # 105 and 106 are compatible apparently?

# ert/enkf/fs_types.hpp
BLOCK_FS_DRIVER_ID = 3001  # only one that's valid today

# res_util/block_fs.cpp
MOUNT_MAP_MAGIC_INT = 8861290
INDEX_MAGIC_INT = 1213775
INDEX_FORMAT_VERSION = 1
NODE_IN_USE = 1431655765
NODE_FREE = -1431655766
NODE_WRITE_ACTIVE = 77162
NODE_END_TAG = 16711935


class Index(BaseModel):
    name: Optional[str] = None
    node_offset: int = 0
    node_length: int = 0
    data_offset: int = 0
    data_length: int = 0


class ByteStream:
    def __init__(self, stream):
        self.stream = stream

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def read(self, count) -> bytes:
        data = self.stream.read(count)
        if len(data) < count:
            raise StopIteration
        return data

    def ecl_string(self) -> str:
        count = self.i32()
        if count < 0:
            return ""
        return self.read(count + 1)[:-1].decode()

    def ecl_array(self, dtype: str) -> np.ndarray:
        count = self.i32()
        assert count >= 0

        t = np.dtype(dtype)
        b = self.read(t.itemsize * count)
        return np.frombuffer(b, dtype=t)

    def ecl_stringlist(self) -> list[str]:
        count = self.i32()
        assert count >= 0

        return [self.ecl_string() for _ in range(count)]

    def assert_eof(self) -> None:
        assert self.stream.read(1) == b""


class Ensemble:
    def __init__(self, path: Path) -> None:
        self.path = path

        self._check_fstab()

    def _check_fstab(self) -> None:
        fstab = ByteStream((self.path / "ert_fstab").open("rb"))

        magic = fstab.u64()
        assert magic == FS_MAGIC_ID

        version = fstab.u32()
        assert version == CURRENT_FS_VERSION

        driver_id = fstab.u32()
        assert driver_id == BLOCK_FS_DRIVER_ID

        try:
            while True:
                self._read_fs(fstab)
        except StopIteration:
            ...

        for i in range(32):
            self.read_data("FORECAST", i)

    def _read_fs(self, fstab: ByteStream) -> None:
        driver_type = fstab.u32()
        # assert driver_type == 1
        num_fs = fstab.u32()
        mountfile_fmt = fstab.ecl_string()

    def read_time_map(self) -> None:
        stream = ByteStream((self.path / "files" / "time-map").open("rb"))
        return stream.ecl_array("datetime64[s]")

    def read_summary_key_set(self) -> None:
        stream = ByteStream((self.path / "files" / "summary-key-set").open("rb"))
        return stream.ecl_stringlist()

    def read_data(self, name: str, mod: int = 0) -> None:
        mnt = ByteStream((self.path / "Ensemble" / f"mod_{mod}" / f"{name}.mnt").open("rb"))
        assert mnt.u32() == MOUNT_MAP_MAGIC_INT
        version = mnt.u32()
        mnt.assert_eof()

        index = ByteStream(
            (self.path / "Ensemble" / f"mod_{mod}" / f"{name}.index").open("rb")
        )
        assert index.u32() == INDEX_MAGIC_INT
        assert index.u32() == INDEX_FORMAT_VERSION
        index.u64()  # datetime64[s] of the data file

        num_active_nodes = index.i32()
        print("num_active_nodes", num_active_nodes)
        indices: list[Index] = []
        for i in range(num_active_nodes):
            node_name = index.ecl_string()
            status = index.i32()
            node_offset = index.i64()
            node_length = index.i32()
            data_offset = index.i32()
            data_length = index.i32()

            assert status == NODE_IN_USE
            assert node_offset >= 0
            assert node_length >= 0
            assert data_offset >= 0
            assert data_length >= 0

            indices.append(
                Index(
                    name=node_name,
                    node_offset=node_offset,
                    node_length=node_length,
                    data_offset=data_offset,
                    data_length=data_length,
                )
            )

        num_free_nodes = index.i32()
        print("num_free_nodes", num_free_nodes)
        for _ in range(num_free_nodes):
            status = index.i32()
            node_offset = index.i64()
            node_length = index.i32()
            data_offset = index.i32()
            data_length = index.i32()

            assert status == NODE_FREE
            assert node_offset >= 0
            assert node_length >= 0
            assert data_offset >= 0
            assert data_length >= 0

            indices.append(
                Index(
                    name=None,
                    node_offset=node_offset,
                    node_length=node_length,
                    data_offset=data_offset,
                    data_length=data_length,
                )
            )
        index.assert_eof()

        data = ByteStream(
            (self.path / "Ensemble" / f"mod_{mod}" / f"{name}.data_{version}").open("rb")
        )
        while True:
            status = data.i32()
            key = None
            if status == NODE_IN_USE:
                key = data.ecl_string()
            elif status == NODE_FREE:
                key = None
            elif status == NODE_END_TAG:
                break
            else:
                raise ValueError(f"Unknown status {status}")

            print("node_size", data.i32())
            if status == NODE_IN_USE:
                print("data_size", data.i32())


def each_ensemble() -> Generator[Ensemble, None, None]:
    root = Path("storage/snake_oil/ensemble")
    for path in root.glob("*/"):
        if path.is_dir():
            yield Ensemble(path)
            sys.exit(0)


def main() -> None:
    for ens in each_ensemble():
        pass


if __name__ == "__main__":
    main()
