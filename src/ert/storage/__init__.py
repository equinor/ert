from __future__ import annotations

import os
from typing import Literal, Union, overload

from ert.storage._protocol import (
    EnsembleAccessor,
    EnsembleReader,
    ExperimentAccessor,
    ExperimentReader,
    StorageAccessor,
    StorageReader,
)
from ert.storage.local_storage import local_storage_needs_migration


@overload
def open_storage(
    path: Union[str, os.PathLike[str]], mode: Literal["r"] = "r"
) -> StorageReader:
    ...


@overload
def open_storage(
    path: Union[str, os.PathLike[str]], mode: Literal["w"]
) -> StorageAccessor:
    ...


def open_storage(
    path: Union[str, os.PathLike[str]], mode: Literal["r", "w"] = "r"
) -> Union[StorageReader, StorageAccessor]:
    # pylint: disable=unused-import,import-outside-toplevel
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

    if mode == "r":
        return LocalStorageReader(path)
    else:
        return LocalStorageAccessor(path)


__all__ = [
    "EnsembleReader",
    "EnsembleAccessor",
    "ExperimentReader",
    "ExperimentAccessor",
    "StorageReader",
    "StorageAccessor",
    "open_storage",
    "local_storage_needs_migration",
]
