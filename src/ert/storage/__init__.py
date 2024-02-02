from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.local_experiment import LocalExperiment
from ert.storage.local_storage import LocalStorage
from ert.storage.mode import Mode, ModeLiteral

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


def open_storage(
    path: Union[str, os.PathLike[str]], mode: Union[ModeLiteral, Mode] = "r"
) -> Storage:
    return LocalStorage(Path(path), Mode(mode))


__all__ = [
    "Ensemble",
    "Experiment",
    "Storage",
    "Mode",
    "open_storage",
]
