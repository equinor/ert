from __future__ import annotations

from .local_ensemble import (
    LocalEnsemble,
    load_realization_parameters_and_responses,
)
from .local_experiment import ExperimentState, ExperimentStatus, LocalExperiment
from .local_storage import LocalStorage, local_storage_set_ert_config, open_storage
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


__all__ = [
    "Ensemble",
    "Experiment",
    "ExperimentState",
    "ExperimentStatus",
    "Mode",
    "ModeLiteral",
    "RealizationStorageState",
    "Storage",
    "load_realization_parameters_and_responses",
    "local_storage_set_ert_config",
    "open_storage",
]
