from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    no_type_check,
    overload,
)
from uuid import UUID, uuid4

from filelock import FileLock, Timeout
from pydantic import BaseModel

from ert.config import ErtConfig
from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
from ert.storage.local_experiment import LocalExperimentAccessor, LocalExperimentReader

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if TYPE_CHECKING:
    from ert.config.parameter_config import ParameterConfig
    from ert.storage import (
        EnsembleAccessor,
        EnsembleReader,
        ExperimentAccessor,
        ExperimentReader,
        StorageAccessor,
    )


_LOCAL_STORAGE_VERSION = 1


class _Index(BaseModel):
    version: int = _LOCAL_STORAGE_VERSION


class LocalStorageReader:
    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        self.path = Path(path).absolute()

        self._experiments: Dict[UUID, Any]
        self._ensembles: Dict[UUID, Any]
        self._index: _Index

        self.refresh()

    def refresh(self) -> None:
        self._index = self._load_index()
        self._ensembles = self._load_ensembles()  # type: ignore
        self._experiments = self._load_experiments()

    def close(self) -> None:
        for ensemble in self._ensembles.values():
            ensemble.close()
        self._ensembles.clear()
        self._experiments.clear()

    def to_accessor(self) -> StorageAccessor:
        raise TypeError(str(type(self)))

    @overload
    def get_experiment(self, uuid: UUID, mode: Literal["r"] = "r") -> ExperimentReader:
        ...

    @overload
    def get_experiment(self, uuid: UUID, mode: Literal["w"]) -> ExperimentAccessor:
        ...

    def get_experiment(
        self, uuid: UUID, mode: Literal["r", "w"] = "r"
    ) -> Union[ExperimentReader, ExperimentAccessor]:
        assert mode == "r"
        return self._experiments[uuid]

    @overload
    def get_ensemble(self, uuid: UUID, mode: Literal["r"] = "r") -> EnsembleReader:
        ...

    @overload
    def get_ensemble(self, uuid: UUID, mode: Literal["w"]) -> EnsembleAccessor:
        ...

    def get_ensemble(
        self, uuid: UUID, mode: Literal["r", "w"] = "r"
    ) -> Union[EnsembleReader, EnsembleAccessor]:
        assert mode == "r"
        return self._ensembles[uuid]

    @overload
    def get_ensemble_by_name(
        self, name: str, mode: Literal["r"] = "r"
    ) -> EnsembleReader:
        ...

    @overload
    def get_ensemble_by_name(self, name: str, mode: Literal["w"]) -> EnsembleAccessor:
        ...

    def get_ensemble_by_name(
        self, name: str, mode: Literal["r", "w"] = "r"
    ) -> Union[EnsembleReader, EnsembleAccessor]:
        assert mode == "r"
        for ens in self._ensembles.values():
            if ens.name == name:
                return ens
        raise KeyError(f"Ensemble with name '{name}' not found")

    @property
    def experiments(self) -> Generator[ExperimentReader, None, None]:
        yield from self._experiments.values()

    @property
    def ensembles(self) -> Generator[EnsembleReader, None, None]:
        yield from self._ensembles.values()

    def _load_index(self) -> _Index:
        try:
            return _Index.parse_file(self.path / "index.json")
        except FileNotFoundError:
            return _Index()

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
        return LocalEnsembleReader(self, path)

    def _load_experiments(self) -> Dict[UUID, Any]:
        experiment_ids = {ens.experiment_id for ens in self._ensembles.values()}
        return {exp_id: self._load_experiment(exp_id) for exp_id in experiment_ids}

    def _load_experiment(self, uuid: UUID) -> LocalExperimentReader:
        return LocalExperimentReader(self, uuid, self._experiment_path(uuid))

    def _ensemble_path(self, ensemble_id: UUID) -> Path:
        return self.path / "ensembles" / str(ensemble_id)

    def _experiment_path(self, experiment_id: UUID) -> Path:
        return self.path / "experiments" / str(experiment_id)

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

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        ignore_migration_check: bool = False,
    ) -> None:
        self.path = Path(path)
        if not ignore_migration_check and local_storage_needs_migration(self.path):
            from ert.storage.migration.block_fs import migrate  # pylint: disable=C0415

            migrate(self.path)

        self.path.mkdir(parents=True, exist_ok=True)

        # ERT 4 checks that this file exists and if it exists tells the user
        # that their ERT storage is incompatible
        try:
            (self.path / ".fs_version").symlink_to("index.json")
        except FileExistsError:
            pass

        self._lock = FileLock(self.path / "storage.lock")
        try:
            self._lock.acquire(timeout=self.LOCK_TIMEOUT)
        except Timeout:
            raise TimeoutError(
                f"Not able to acquire lock for: {self.path}."
                " You may already be running ERT,"
                " or another user is using the same ENSPATH."
            )

        super().__init__(path)

        self._save_index()

    def close(self) -> None:
        self._save_index()
        super().close()

        if self._lock.is_locked:
            self._lock.release()
            (self.path / "storage.lock").unlink()

    def to_accessor(self) -> StorageAccessor:
        return self

    def create_experiment(
        self, parameters: Optional[Sequence[ParameterConfig]] = None
    ) -> ExperimentAccessor:
        exp_id = uuid4()
        path = self._experiment_path(exp_id)
        path.mkdir(parents=True, exist_ok=False)
        exp = LocalExperimentAccessor(self, exp_id, path, parameters=parameters)
        self._experiments[exp.id] = exp
        return exp

    def create_ensemble(
        self,
        experiment: Union[ExperimentReader, UUID],
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: Optional[str] = None,
        prior_ensemble: Optional[Union[EnsembleReader, UUID]] = None,
    ) -> EnsembleAccessor:
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
        )
        self._ensembles[ens.id] = ens
        return ens

    def _save_index(self) -> None:
        if not hasattr(self, "_index"):
            return

        with open(self.path / "index.json", mode="w", encoding="utf-8") as f:
            print(self._index.json(), file=f)

    def _load_experiment(self, uuid: UUID) -> LocalExperimentAccessor:
        return LocalExperimentAccessor(self, uuid, self._experiment_path(uuid))

    def _load_ensemble(self, path: Path) -> LocalEnsembleAccessor:
        return LocalEnsembleAccessor(self, path)


def local_storage_needs_migration(path: os.PathLike[str]) -> bool:
    """
    Checks whether the path points to a LocalStorage that is in need of
    migration. Returns true if the LocalStorage is outdated OR if the path does
    not point to a LocalStorage directory (eg. it is a URL that points to a ERT
    Storage Server)
    """
    path = Path(path)

    if not path.exists():
        return False

    try:
        with open("index.json", encoding="utf-8") as f:
            version = json.load(f)["version"]
        if version == _LOCAL_STORAGE_VERSION:
            return False
        elif version < _LOCAL_STORAGE_VERSION:
            return True
        elif version > _LOCAL_STORAGE_VERSION:
            raise NotImplementedError("Incompatible ERT Local Storage")
    except KeyError as exc:
        raise NotImplementedError("Incompatible ERT Local Storage") from exc
    except FileNotFoundError:
        pass

    return _is_block_storage(path)


_migration_ert_config: Optional[ErtConfig] = None


def local_storage_set_ert_config(ert_config: Optional[ErtConfig]) -> None:
    """
    Set the ErtConfig for migration hints
    """
    global _migration_ert_config  # pylint: disable=W0603
    _migration_ert_config = ert_config


def local_storage_get_ert_config() -> ErtConfig:
    assert (
        _migration_ert_config is not None
    ), "Use 'local_storage_set_ert_config' before retrieving the config"
    return _migration_ert_config


def _is_block_storage(path: Path) -> bool:
    """Looks for ert_fstab in subdirectories"""
    for subpath in path.iterdir():
        if subpath.name.startswith("_"):
            continue

        if (subpath / "ert_fstab").exists():
            return True

    return False
