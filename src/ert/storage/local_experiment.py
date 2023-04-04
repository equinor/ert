from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Optional, Union
from uuid import UUID

if TYPE_CHECKING:
    from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader


class LocalExperimentReader:
    def __init__(self, storage: LocalStorageReader, uuid: UUID) -> None:
        self._storage: Union[LocalStorageReader, LocalStorageAccessor] = storage
        self._id = uuid

    @property
    def ensembles(self) -> Generator[LocalEnsembleReader, None, None]:
        yield from (
            ens for ens in self._storage.ensembles if ens.experiment_id == self.id
        )

    @property
    def id(self) -> UUID:
        return self._id


class LocalExperimentAccessor(LocalExperimentReader):
    def __init__(self, storage: LocalStorageAccessor, uuid: UUID) -> None:
        self._storage: LocalStorageAccessor = storage
        self._id = uuid

    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: str,
        prior_ensemble: Optional[LocalEnsembleReader] = None,
    ) -> LocalEnsembleAccessor:
        return self._storage.create_ensemble(
            self._id,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

    @property
    def ensembles(self) -> Generator[LocalEnsembleAccessor, None, None]:
        yield from super().ensembles  # type: ignore
