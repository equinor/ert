from __future__ import annotations

import os
import sys
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Optional,
    Type,
    Union,
    no_type_check,
)
from uuid import UUID, uuid4

from filelock import FileLock, Timeout

from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
from ert.storage.local_experiment import LocalExperimentAccessor, LocalExperimentReader

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if TYPE_CHECKING:
    from ecl.summary import EclSum

    from ert._c_wrappers.enkf.ensemble_config import ParameterConfiguration


class LocalStorageReader:
    def __init__(
        self, path: Union[str, os.PathLike[str]], refcase: Optional[EclSum] = None
    ) -> None:
        self.path = Path(path).absolute()

        self._experiments: Union[
            Dict[UUID, LocalExperimentReader], Dict[UUID, LocalExperimentAccessor]
        ]
        self._ensembles: Union[
            Dict[UUID, LocalEnsembleReader], Dict[UUID, LocalEnsembleAccessor]
        ]
        self._refcase = refcase

        self.refresh()

    def refresh(self) -> None:
        self._ensembles = self._load_ensembles()  # type: ignore
        self._experiments = self._load_experiments()

    def close(self) -> None:
        for ensemble in self._ensembles.values():
            ensemble.close()
        self._ensembles.clear()
        self._experiments.clear()

    def to_accessor(self) -> LocalStorageAccessor:
        raise TypeError(str(type(self)))

    def get_experiment(self, uuid: UUID) -> LocalExperimentReader:
        return self._experiments[uuid]

    def get_ensemble(self, uuid: UUID) -> LocalEnsembleReader:
        return self._ensembles[uuid]

    def get_ensemble_by_name(
        self, name: str
    ) -> Union[LocalEnsembleReader, LocalEnsembleAccessor]:
        for ens in self._ensembles.values():
            if ens.name == name:
                return ens
        raise KeyError(f"Ensemble with name '{name}' not found")

    @property
    def experiments(self) -> Generator[LocalExperimentReader, None, None]:
        yield from self._experiments.values()

    @property
    def ensembles(self) -> Generator[LocalEnsembleReader, None, None]:
        yield from self._ensembles.values()

    @no_type_check
    def _load_ensembles(self):
        try:
            ensembles = []
            for ensemble_path in (self.path / "ensembles").iterdir():
                ensemble = self._load_ensemble(ensemble_path)
                ensembles.append(ensemble)

            # Make sure that the ensembles are sorted by name in reverse. Given
            # multiple ensembles with a common name, iterating over the ensemble
            # dictionary will yield the newest ensemble first.
            ensembles = sorted(ensembles, key=lambda x: x.started_at, reverse=True)
            return {
                x.id: x
                for x in sorted(ensembles, key=lambda x: x.started_at, reverse=True)
            }
        except FileNotFoundError:
            return {}

    def _load_ensemble(self, path: Path) -> Any:
        return LocalEnsembleReader(self, path, refcase=self._refcase)

    def _load_experiments(self) -> Dict[UUID, LocalExperimentReader]:
        experiment_ids = {ens.experiment_id for ens in self._ensembles.values()}
        return {exp_id: self._load_experiment(exp_id) for exp_id in experiment_ids}

    def _load_experiment(self, uuid: UUID) -> LocalExperimentReader:
        return LocalExperimentReader(self, uuid, self._experiment_path(uuid))

    def _ensemble_path(self, ensemble_id: UUID) -> Path:
        return self.path / "ensembles" / str(ensemble_id)

    def _experiment_path(self, experiment_id: UUID) -> Path:
        return self.path / "experiments" / str(experiment_id)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception: Exception,
        exception_type: Type[Exception],
        traceback: TracebackType,
    ) -> None:
        self.close()


class LocalStorageAccessor(LocalStorageReader):
    LOCK_TIMEOUT = 5

    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        super().__init__(path)

        self.path.mkdir(parents=True, exist_ok=True)

        self._lock = FileLock(self.path / "storage.lock")
        try:
            self._lock.acquire(timeout=self.LOCK_TIMEOUT)
        except Timeout:
            raise TimeoutError(
                f"Not able to acquire lock for: {self.path}."
                " You may already be running ERT,"
                " or another user is using the same ENSPATH."
            )

    def close(self) -> None:
        super().close()

        if self._lock.is_locked:
            self._lock.release()
            (self.path / "storage.lock").unlink()

    def to_accessor(self) -> LocalStorageAccessor:
        return self

    def create_experiment(
        self, parameters: Optional[ParameterConfiguration] = None
    ) -> LocalExperimentAccessor:
        exp_id = uuid4()
        path = self._experiment_path(exp_id)
        path.mkdir(parents=True, exist_ok=False)
        exp = LocalExperimentAccessor(self, exp_id, path, parameters=parameters)
        self._experiments[exp.id] = exp
        return exp

    def create_ensemble(
        self,
        experiment: Union[LocalExperimentReader, LocalExperimentAccessor, UUID],
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: Optional[str] = None,
        prior_ensemble: Optional[Union[LocalEnsembleReader, UUID]] = None,
        refcase: Optional[EclSum] = None,
    ) -> LocalEnsembleAccessor:
        if isinstance(experiment, UUID):
            experiment_id = experiment
        else:
            experiment_id = experiment.id

        uuid = uuid4()
        path = self._ensemble_path(uuid)
        path.mkdir(parents=True, exist_ok=False)

        prior_ensemble_id: Optional[UUID] = None
        if isinstance(prior_ensemble, UUID):
            prior_ensemble_id = prior_ensemble
        elif isinstance(prior_ensemble, LocalEnsembleReader):
            prior_ensemble_id = prior_ensemble.id

        ens = LocalEnsembleAccessor.create(
            self,
            path,
            uuid,
            ensemble_size=ensemble_size,
            experiment_id=experiment_id,
            iteration=iteration,
            name=str(name),
            prior_ensemble_id=prior_ensemble_id,
            refcase=refcase,
        )
        self._ensembles[ens.id] = ens
        return ens

    def _load_experiment(self, uuid: UUID) -> LocalExperimentAccessor:
        return LocalExperimentAccessor(self, uuid, self._experiment_path(uuid))

    def _load_ensemble(self, path: Path) -> LocalEnsembleAccessor:
        return LocalEnsembleAccessor(self, path, refcase=self._refcase)
