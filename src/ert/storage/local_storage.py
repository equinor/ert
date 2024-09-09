from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    MutableSequence,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import UUID, uuid4

import xarray as xr
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from ert.config import ErtConfig
from ert.shared import __version__
from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.local_experiment import LocalExperiment
from ert.storage.mode import (
    BaseMode,
    Mode,
    require_write,
)
from ert.storage.realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    from ert.config import ParameterConfig, ResponseConfig

logger = logging.getLogger(__name__)

_LOCAL_STORAGE_VERSION = 7


class _Migrations(BaseModel):
    ert_version: str = __version__
    timestamp: datetime = Field(default_factory=datetime.now)
    name: str
    version_range: Tuple[int, int]


class _Index(BaseModel):
    version: int = _LOCAL_STORAGE_VERSION
    migrations: MutableSequence[_Migrations] = Field(default_factory=list)


class LocalStorage(BaseMode):
    """
    A class representing the local storage for ERT experiments and ensembles.

    This class manages the file-based storage system used by ERT to store
    experiments and ensembles.
    It includes functionality to handle versioning, migrations, and concurrency
    through file locks.
    """

    LOCK_TIMEOUT = 5
    EXPERIMENTS_PATH = "experiments"
    ENSEMBLES_PATH = "ensembles"

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        mode: Mode,
        *,
        ignore_migration_check: bool = False,
    ) -> None:
        """
        Initializes the LocalStorage instance.

        Parameters
        ----------
        path : {str, path-like}
            The file system path to the storage.
        mode : Mode
            The access mode for the storage (read/write).
        ignore_migration_check : bool
            If True, skips migration checks during initialization.
        """

        super().__init__(mode)
        self.path = Path(path).absolute()

        self._experiments: Dict[UUID, LocalExperiment]
        self._ensembles: Dict[UUID, LocalEnsemble]
        self._index: _Index

        try:
            version = _storage_version(self.path)
        except FileNotFoundError as err:
            # No index json, will have a problem if other components of storage exists
            errors = []
            if (self.path / self.EXPERIMENTS_PATH).exists():
                errors.append(
                    f"experiments path: {(self.path / self.EXPERIMENTS_PATH)}"
                )
            if (self.path / self.ENSEMBLES_PATH).exists():
                errors.append(f"ensemble path: {self.path / self.ENSEMBLES_PATH}")
            if errors:
                raise ValueError(f"No index.json, but found: {errors}") from err
            version = _LOCAL_STORAGE_VERSION

        if version > _LOCAL_STORAGE_VERSION:
            raise RuntimeError(
                f"Cannot open storage '{self.path}': Storage version {version} is newer than the current version {_LOCAL_STORAGE_VERSION}, upgrade ert to continue, or run with a different ENSPATH"
            )
        if self.can_write:
            self._acquire_lock()
            if version < _LOCAL_STORAGE_VERSION and not ignore_migration_check:
                self._migrate(version)
            self._index = self._load_index()
            self._ensure_fs_version_exists()
            self._save_index()
        elif version < _LOCAL_STORAGE_VERSION:
            raise RuntimeError(
                f"Cannot open storage '{self.path}' in read-only mode: Storage version {version} is too old. Run ert to initiate migration."
            )
        self.refresh()

    def refresh(self) -> None:
        """
        Reloads the index, experiments, and ensembles from the storage.

        This method is used to refresh the state of the storage to reflect any
        changes made to the underlying file system since the storage was last
        accessed.
        """

        self._index = self._load_index()
        self._ensembles = self._load_ensembles()
        self._experiments = self._load_experiments()

    def get_experiment(self, uuid: UUID) -> LocalExperiment:
        """
        Retrieves an experiment by UUID.

        Parameters
        ----------
        uuid : UUID
            The UUID of the experiment to retrieve.

        Returns
        -------
        local_experiment : LocalExperiment
            The experiment associated with the given UUID.
        """

        return self._experiments[uuid]

    def get_experiment_by_name(self, name: str) -> LocalExperiment:
        """
        Retrieves an experiment by name.
        Parameters
        ----------
        name : str
            The name of the experiment to retrieve.
        Returns
        -------
        local_experiment : LocalExperiment
            The experiment associated with the given name.
        Raises
        ------
        KeyError
            If no experiment with the given name is found.
        """
        for exp in self._experiments.values():
            if exp.name == name:
                return exp
        raise KeyError(f"Experiment with name '{name}' not found")

    def get_ensemble(self, uuid: Union[UUID, str]) -> LocalEnsemble:
        """
        Retrieves an ensemble by UUID.

        Parameters
        ----------
        uuid : UUID
            The UUID of the ensemble to retrieve.

        Returns
        local_ensemble : LocalEnsemble
            The ensemble associated with the given UUID.
        """
        if isinstance(uuid, str):
            uuid = UUID(uuid)
        return self._ensembles[uuid]

    @property
    def experiments(self) -> Generator[LocalExperiment, None, None]:
        yield from self._experiments.values()

    @property
    def ensembles(self) -> Generator[LocalEnsemble, None, None]:
        yield from self._ensembles.values()

    def _load_index(self) -> _Index:
        try:
            return _Index.model_validate_json(
                (self.path / "index.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return _Index()

    def _load_ensembles(self) -> Dict[UUID, LocalEnsemble]:
        if not (self.path / "ensembles").exists():
            return {}
        ensembles: List[LocalEnsemble] = []
        for ensemble_path in (self.path / "ensembles").iterdir():
            try:
                ensemble = LocalEnsemble(self, ensemble_path, self.mode)
                ensembles.append(ensemble)
            except FileNotFoundError:
                logger.exception(
                    "Failed to load an ensemble from path: %s", ensemble_path
                )
                continue
        # Make sure that the ensembles are sorted by name in reverse. Given
        # multiple ensembles with a common name, iterating over the ensemble
        # dictionary will yield the newest ensemble first.
        return {
            x.id: x for x in sorted(ensembles, key=lambda x: x.started_at, reverse=True)
        }

    def _load_experiments(self) -> Dict[UUID, LocalExperiment]:
        experiment_ids = {ens.experiment_id for ens in self._ensembles.values()}
        return {
            exp_id: LocalExperiment(self, self._experiment_path(exp_id), self.mode)
            for exp_id in experiment_ids
        }

    def _ensemble_path(self, ensemble_id: UUID) -> Path:
        return self.path / self.ENSEMBLES_PATH / str(ensemble_id)

    def _experiment_path(self, experiment_id: UUID) -> Path:
        return self.path / self.EXPERIMENTS_PATH / str(experiment_id)

    def __enter__(self) -> LocalStorage:
        return self

    def __exit__(
        self,
        exception: Exception,
        exception_type: Type[Exception],
        traceback: TracebackType,
    ) -> None:
        self.close()

    @require_write
    def _ensure_fs_version_exists(self) -> None:
        # ERT 4 checks that this file exists and if it exists tells the user
        # that their ERT storage is incompatible
        with contextlib.suppress(FileExistsError):
            (self.path / ".fs_version").symlink_to("index.json")

    @require_write
    def _acquire_lock(self) -> None:
        self._lock = FileLock(self.path / "storage.lock")
        try:
            self._lock.acquire(timeout=self.LOCK_TIMEOUT)
        except Timeout as e:
            raise TimeoutError(
                f"Not able to acquire lock for: {self.path}."
                " You may already be running ERT,"
                " or another user is using the same ENSPATH."
            ) from e

    def close(self) -> None:
        """
        Closes the storage, releasing any acquired locks and saving the index.

        This method should be called to cleanly close the storage, especially
        when it was opened in write mode. Failing to call this method may leave
        a lock file behind, which would interfere with subsequent access to
        the storage.
        """

        self._ensembles.clear()
        self._experiments.clear()

        if not self.can_write:
            return

        self._save_index()
        self._release_lock()

    def _release_lock(self) -> None:
        if self._lock.is_locked:
            self._lock.release()
            (self.path / "storage.lock").unlink()

    @require_write
    def create_experiment(
        self,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, xr.Dataset]] = None,
        simulation_arguments: Optional[Dict[Any, Any]] = None,
        name: Optional[str] = None,
    ) -> LocalExperiment:
        """
        Creates a new experiment in the storage.

        Parameters
        ----------
        parameters : list of ParameterConfig, optional
            The parameters for the experiment.
        responses : list of ResponseConfig, optional
            The responses for the experiment.
        observations : dict of str to Dataset, optional
            The observations for the experiment.
        simulation_arguments : SimulationArguments, optional
            The simulation arguments for the experiment.
        name : str, optional
            The name of the experiment.

        Returns
        -------
        local_experiment : LocalExperiment
            The newly created experiment.
        """

        exp_id = uuid4()
        path = self._experiment_path(exp_id)
        path.mkdir(parents=True, exist_ok=False)

        exp = LocalExperiment.create(
            self,
            exp_id,
            path,
            parameters=parameters,
            responses=responses,
            observations=observations,
            simulation_arguments=simulation_arguments,
            name=name,
        )

        self._experiments[exp.id] = exp
        return exp

    @require_write
    def create_ensemble(
        self,
        experiment: Union[LocalExperiment, UUID],
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: Optional[str] = None,
        prior_ensemble: Union[LocalEnsemble, UUID, None] = None,
    ) -> LocalEnsemble:
        """
        Creates a new ensemble in the storage.

        Raises a ValueError if the ensemble size is larger than the prior
        ensemble.

        Parameters
        ----------
        experiment : {LocalExperiment, UUID}
            The experiment for which the ensemble is created.
        ensemble_size : int
            The number of realizations in the ensemble.
        iteration : int, optional
            The iteration index for the ensemble.
        name : str, optional
            The name of the ensemble.
        prior_ensemble : {LocalEnsemble, UUID}, optional
            An optional ensemble to use as a prior.

        Returns
        -------
        local_ensemble : LocalEnsemble
            The newly created ensemble.
        """

        experiment_id = experiment if isinstance(experiment, UUID) else experiment.id

        uuid = uuid4()
        path = self._ensemble_path(uuid)
        path.mkdir(parents=True, exist_ok=False)

        prior_ensemble_id: Optional[UUID] = None
        if isinstance(prior_ensemble, UUID):
            prior_ensemble_id = prior_ensemble
        elif isinstance(prior_ensemble, LocalEnsemble):
            prior_ensemble_id = prior_ensemble.id
        prior_ensemble = (
            self.get_ensemble(prior_ensemble_id) if prior_ensemble_id else None
        )
        if prior_ensemble and ensemble_size > prior_ensemble.ensemble_size:
            raise ValueError(
                f"New ensemble ({ensemble_size}) must be of equal or "
                f"smaller size than parent ensemble ({prior_ensemble.ensemble_size})"
            )
        ens = LocalEnsemble.create(
            self,
            path,
            uuid,
            ensemble_size=ensemble_size,
            experiment_id=experiment_id,
            iteration=iteration,
            name=str(name),
            prior_ensemble_id=prior_ensemble_id,
        )
        if prior_ensemble:
            for realization, state in enumerate(prior_ensemble.get_ensemble_state()):
                if state in [
                    RealizationStorageState.LOAD_FAILURE,
                    RealizationStorageState.PARENT_FAILURE,
                    RealizationStorageState.UNDEFINED,
                ]:
                    ens.set_failure(
                        realization,
                        RealizationStorageState.PARENT_FAILURE,
                        f"Failure from prior: {state}",
                    )

        self._ensembles[ens.id] = ens
        return ens

    @require_write
    def _add_migration_information(
        self, from_version: int, to_version: int, name: str
    ) -> None:
        self._index.migrations.append(
            _Migrations(
                version_range=(from_version, to_version),
                name=name,
            )
        )
        self._index.version = to_version
        self._save_index()

    @require_write
    def _save_index(self) -> None:
        with open(self.path / "index.json", mode="w", encoding="utf-8") as f:
            print(self._index.model_dump_json(indent=4), file=f)

    @require_write
    def _migrate(self, version: int) -> None:
        from ert.storage.migration import (  # noqa: PLC0415
            block_fs,
            to2,
            to3,
            to4,
            to5,
            to6,
            to7,
        )

        try:
            self._index = self._load_index()
            if version == 0:
                self._release_lock()
                block_fs.migrate(self.path)
                self._acquire_lock()
                self._add_migration_information(0, _LOCAL_STORAGE_VERSION, "block_fs")
            elif version < _LOCAL_STORAGE_VERSION:
                migrations = list(enumerate([to2, to3, to4, to5, to6, to7], start=1))
                for from_version, migration in migrations[version - 1 :]:
                    print(f"* Updating storage to version: {from_version+1}")
                    migration.migrate(self.path)
                    self._add_migration_information(
                        from_version, from_version + 1, migration.info
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                f"Migrating storage at {self.path} failed with: {e}", stack_info=True
            )
            raise e

    def get_unique_experiment_name(self, experiment_name: str) -> str:
        """
        Get a unique experiment name

        If an experiment with the given name exists an _0 is appended
        or _n+1 where n is the the largest postfix found for the given experiment name
        """
        if not experiment_name:
            return self.get_unique_experiment_name("default")

        if experiment_name not in [e.name for e in self.experiments]:
            return experiment_name

        if (
            len(
                same_prefix := [
                    e.name
                    for e in self.experiments
                    if e.name.startswith(experiment_name + "_")
                ]
            )
            > 0
        ):
            return (
                experiment_name
                + "_"
                + str(max(int(e[e.rfind("_") + 1 :]) for e in same_prefix) + 1)
            )
        else:
            return experiment_name + "_0"


def _storage_version(path: Path) -> int:
    if not path.exists():
        return _LOCAL_STORAGE_VERSION
    try:
        with open(path / "index.json", encoding="utf-8") as f:
            return int(json.load(f)["version"])
    except KeyError as exc:
        raise NotImplementedError("Incompatible ERT Local Storage") from exc
    except FileNotFoundError:
        if _is_block_storage(path):
            return 0
        else:
            raise


_migration_ert_config: Optional[ErtConfig] = None


def local_storage_set_ert_config(ert_config: Optional[ErtConfig]) -> None:
    """
    Set the ErtConfig for migration hints.

    This function sets a global ErtConfig instance which may be used by
    migration scripts to access configuration details during the migration
    process.

    Parameters
    ----------
    ert_config : Optional[ErtConfig]
        The ErtConfig instance to be used for migrations.
    """

    global _migration_ert_config  # noqa: PLW0603
    _migration_ert_config = ert_config


def local_storage_get_ert_config() -> ErtConfig:
    """
    Retrieves the ErtConfig instance previously set for migrations.

    This function should be called after `local_storage_set_ert_config` has
    been used to set the ErtConfig instance.

    Raises an AssertionError uf the ErtConfig has not been set before calling
    this function.

    Returns
    -------
    ert_config : ErtConfig
        The ErtConfig instance.
    """

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
