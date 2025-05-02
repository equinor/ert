from __future__ import annotations

import os
from pathlib import Path

from .local_ensemble import LocalEnsemble
from .local_experiment import LocalExperiment
from .local_storage import LocalStorage
from .mode import Mode, ModeLiteral

# Alias types. The Local* variants are meant to co-exist with Remote* classes
# that connect to a remote ERT Storage Server, as well as an in-memory Memory*
# variant for testing. The `open_storage` factory is to return the correct type.
# However, currently there is only one implementation, so to keep things simple
# we simply alias these types. In the future these will probably be subclasses
# of typing.Protocol
Storage = LocalStorage
Ensemble = LocalEnsemble
Experiment = LocalExperiment

# Kept here to avoid having to have to rewrite a lot of files
StorageReader = LocalStorage
StorageAccessor = LocalStorage
ExperimentReader = LocalExperiment
ExperimentAccessor = LocalExperiment
EnsembleReader = LocalEnsemble
EnsembleAccessor = LocalEnsemble


class ErtStorageException(Exception):
    pass


def open_storage(
    path: str | os.PathLike[str], mode: ModeLiteral | Mode = "r"
) -> Storage:
    try:
        return LocalStorage(Path(path), Mode(mode))
    except Exception as err:
        raise ErtStorageException(
            f"Failed to open storage: {path} with error: {err}"
        ) from err


__all__ = [
    "Ensemble",
    "Experiment",
    "Mode",
    "Storage",
    "open_storage",
]
