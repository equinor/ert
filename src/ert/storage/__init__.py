from __future__ import annotations

import os
from pathlib import Path

from .local_ensemble import LocalEnsemble, load_realization_parameters_and_responses
from .local_experiment import ExperimentState, ExperimentStatus, LocalExperiment
from .local_storage import LocalStorage, local_storage_set_ert_config
from .mode import Mode, ModeLiteral
from .realization_storage_state import RealizationStorageState

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


class ErtStoragePermissionError(ErtStorageException):
    pass


def open_storage(
    path: str | os.PathLike[str], mode: ModeLiteral | Mode = "r"
) -> Storage:
    _ = LocalStorage.check_migration_needed(Path(path))

    try:
        return LocalStorage(Path(path), Mode(mode))
    except PermissionError as err:
        raise ErtStoragePermissionError(
            "Permission error when accessing storage at: "
            f"{path} with mode: '{mode}'. Error: {err}"
        ) from err
    except Exception as err:
        raise ErtStorageException(
            f"Failed to open storage: {path} with error: {err}"
        ) from err


__all__ = [
    "Ensemble",
    "Experiment",
    "ExperimentState",
    "ExperimentStatus",
    "Mode",
    "RealizationStorageState",
    "Storage",
    "load_realization_parameters_and_responses",
    "local_storage_set_ert_config",
    "open_storage",
]
