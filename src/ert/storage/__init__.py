from __future__ import annotations

import os
from typing import Literal, Union, overload

from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
from ert.storage.local_experiment import LocalExperimentAccessor, LocalExperimentReader
from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

# Alias types. The Local* variants are meant to co-exist with Remote* classes
# that connect to a remote ERT Storage Server, as well as an in-memory Memory*
# variant for testing. The `open_storage` factory is to return the correct type.
# However, currently there is only one implementation, so to keep things simple
# we simply alias these types. In the future these will probably be subclasses
# of typing.Protocol
StorageReader = LocalStorageReader
StorageAccessor = LocalStorageAccessor
ExperimentReader = LocalExperimentReader
ExperimentAccessor = LocalExperimentAccessor
EnsembleReader = LocalEnsembleReader
EnsembleAccessor = LocalEnsembleAccessor


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
]
