import os
import io
import sys
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Generator, ItemsView, Optional
from pydantic import BaseModel
import numpy as np
import math
import zlib

import struct


# From ert/enkf/fs_driver.hpp
FS_MAGIC_ID = 123998
CURRENT_FS_VERSION = 107  # 105 and 106 are compatible apparently?

# ert/enkf/fs_types.hpp
BLOCK_FS_DRIVER_ID = 3001  # only one that's valid today

# res_util/block_fs.cpp
MOUNT_MAP_MAGIC_INT = 8861290
INDEX_FORMAT_VERSION = 1
NODE_IN_USE = 1431655765
NODE_FREE = -1431655766
NODE_WRITE_ACTIVE = 77162
NODE_END_TAG = 16711935


def read_exact(stream: BinaryIO, count: int) -> bytes:
    assert count > 0
    data = stream.read(count)
    if len(data) != count:
        raise IOError(
            f"Premature EOF: Attempted to read {count} bytes, but {len(data)} remained"
        )
    return data


def read_unpack(stream: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    data = read_exact(stream, size)
    return struct.unpack(fmt, data)


def read_eclstring(stream: BinaryIO) -> str:
    """
    A libecl string consists of a 32-bit signed little endian integer 'n'
    followed by 'n' ASCII characters.
    """
    (count,) = read_unpack(stream, "<i")
    return read_exact(stream, count + 1)[:-1].decode()


@np.vectorize
def _func(x: float) -> float:
    min_ = 0.1
    max_ = 0.5
    y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))
    return y * (max_ - min_) + min_


def read_data(data: bytes) -> np.ndarray:
    (file_type,) = struct.unpack("<i", data[:4])
    if file_type == 104:  # FIELD
        data = zlib.decompress(data[4:])
        return np.frombuffer(data)  # , dtype=np.float64)  # Either int, float or double
    if file_type == 107 or file_type == 102:
        # Either GEN_KW (107) or MULTFLT (102, deprecated since 2009)
        return np.frombuffer(data[4:], dtype=np.float64)
    if file_type == 110:  # SUMMARY
        size, default_value = struct.unpack("<id", data[4:16])
        # default_value is an ecl thing that we ignore
        assert size * 8 + 16 == len(data)
        return np.frombuffer(data[16:], dtype=np.float64)
    if file_type == 113:  # GEN_DATA
        size, report_step = struct.unpack("<ii", data[4:12])
        # decompress with type
    if file_type == 114:  # SURFACE
        return np.frombuffer(data[4:], dtype=np.float64)
    if file_type == 115:  # CONTAINER
        # doesn't implement read_from_buffer
        raise NotImplementedError
    if file_type == 116:  # EXT_PARAM
        # unsure how to read this type
        raise NotImplementedError


class NodeStatus(Enum):
    in_use = 0
    free = 1
    invalid = 2


class NodeIndex(BaseModel):
    """
    Index of "node" in the data file
    """

    node_offset: int = 0  # Offset to the start of the "node"
    node_length: int = 0  # Size preallocated to this node
    data_offset: int = 0  # Offset from the start of the "node" to the start of data
    data_length: int = 0  # Size of data contained in the "node"


class IndexFile:
    """
    Loader class for block_fs .index files
    """

    # From res_utl/block_fs.cpp, INDEX_MAGIC_INT
    _FILE_MAGIC = 1213775

    # From res_utl/block_fs.cpp, INDEX_FORMAT_VERSION
    _FILE_VERSION = 1

    def __init__(self, stream: BinaryIO) -> None:
        self._used_indices: dict[str, NodeIndex] = {}
        self._unused_indices: list[NodeIndex] = []

        self._read_file(stream)

    def __getitem__(self, key: str) -> NodeIndex:
        return self._used_indices[key]

    def _read_file(self, stream: BinaryIO) -> None:
        # struct header {
        #   int magic;   // == _FILE_MAGIC
        #   int version; // == _FILE_VERSION
        #   time_t mod;
        # }
        magic, version, time, nactive = read_unpack(stream, "<iiQi")
        assert magic == self._FILE_MAGIC
        assert version == self._FILE_VERSION
        # assert time == ???

        for _ in range(nactive):
            key = read_eclstring(stream)
            status, index = self._read_node(stream)
            assert status == NodeStatus.in_use
            assert key not in self._used_indices
            print(key, index)
            self._used_indices[key] = index

        (nfree,) = read_unpack(stream, "<i")
        for _ in range(nfree):
            status, index = self._read_node(stream)
            assert status == NodeStatus.free
            self._unused_indices.append(index)

        # Check EOF
        assert stream.read(1) == b""

    def _read_node(self, stream: BinaryIO) -> tuple[NodeStatus, NodeIndex]:
        # struct node {
        #   int status;
        #   uint64_t node_offset;
        #   int node_length;
        #   int data_offset;
        #   int data_length;
        # }
        raw_status, node_offset, node_length, data_offset, data_length = read_unpack(
            stream, "<iqiii"
        )
        if raw_status == NODE_IN_USE:
            status = NodeStatus.in_use
        elif raw_status == NODE_FREE:
            status = NodeStatus.free
        else:
            status = NodeStatus.invalid
        return (
            status,
            NodeIndex(
                node_offset=node_offset,
                node_length=node_length,
                data_offset=data_offset,
                data_length=data_length,
            ),
        )


class DataFile:
    """
    Loader class for block_fs .data_N files
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def __getitem__(self, index: NodeIndex) -> bytes:
        self._stream.seek(index.node_offset + index.data_offset + 8)
        return read_exact(self._stream, index.data_length)
        # return np.frombuffer(data, dtype=np.float32)


# class Ensemble:
#     def __init__(self, path: Path) -> None:
#         self.path = path

#         self._check_fstab()

#     def _check_fstab(self) -> None:
#         fstab = ByteStream((self.path / "ert_fstab").open("rb"))

#         magic = fstab.u64()
#         assert magic == FS_MAGIC_ID

#         version = fstab.u32()
#         assert version == CURRENT_FS_VERSION

#         driver_id = fstab.u32()
#         assert driver_id == BLOCK_FS_DRIVER_ID

#         try:
#             while True:
#                 self._read_fs(fstab)
#         except StopIteration:
#             ...

#         for i in range(32):
#             self.read_data("FORECAST", i)

#     def read_data(self, name: str, mod: int) -> None:
#         index = IndexFile()


def each_ensemble() -> None:
    root = Path("storage/snake_oil/ensemble")
    for path in root.glob("*/"):
        if not path.is_dir():
            continue

        for mod in range(32):
            base = str(path / f"Ensemble/mod_{mod}/PARAMETER")
            index_file = IndexFile(open(f"{base}.index", "rb"))
            data_file = DataFile(open(f"{base}.data_0", "rb"))

            for key, index in index_file._used_indices.items():
                print("Trying to read", key)
                data = data_file[index]
                read_data(data)

            sys.exit(0)

    #     if path.is_dir():
    #         yield Ensemble(path)
    #         sys.exit(0)


def main() -> None:
    each_ensemble()


if __name__ == "__main__":
    main()
