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
    from ert.run_models.run_arguments import RunArgumentsType

logger = logging.getLogger(__name__)

_LOCAL_STORAGE_VERSION = 5


class _Migrations(BaseModel):
    ert_version: str = __version__
    timestamp: datetime = Field(default_factory=datetime.now)
    name: str
    version_range: Tuple[int, int]


class _Index(BaseModel):
    version: int = _LOCAL_STORAGE_VERSION
    migrations: MutableSequence[_Migrations] = Field(default_factory=list)


class LocalStorage(BaseMode):
    LOCK_TIMEOUT = 5

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        mode: Mode,
        *,
        ignore_migration_check: bool = False,
    ) -> None:
        super().__init__(mode)
        self.path = Path(path).absolute()

        self._experiments: Dict[UUID, LocalExperiment]
        self._ensembles: Dict[UUID, LocalEnsemble]
        self._index: _Index

        if self.can_write:
            self.path.mkdir(parents=True, exist_ok=True)
            self._migrate(ignore_migration_check)
            self._index = self._load_index()
            self._acquire_lock()
            self._ensure_fs_version_exists()
            self._save_index()
        elif (version := _storage_version(self.path)) is not None:
            if version != _LOCAL_STORAGE_VERSION:
                raise RuntimeError(
                    f"Cannot open storage '{self.path}' in read-only mode: Storage version {version} is too old"
                )

        self.refresh()

    def refresh(self) -> None:
        self._index = self._load_index()
        self._ensembles = self._load_ensembles()
        self._experiments = self._load_experiments()

    def get_experiment(self, uuid: UUID) -> LocalExperiment:
        return self._experiments[uuid]

    def get_ensemble(self, uuid: UUID) -> LocalEnsemble:
        return self._ensembles[uuid]

    def get_ensemble_by_name(self, name: str) -> Union[LocalEnsemble, LocalEnsemble]:
        for ens in self._ensembles.values():
            if ens.name == name:
                return ens
        raise KeyError(f"Ensemble with name '{name}' not found")

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
        return self.path / "ensembles" / str(ensemble_id)

    def _experiment_path(self, experiment_id: UUID) -> Path:
        return self.path / "experiments" / str(experiment_id)

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
        self._ensembles.clear()
        self._experiments.clear()

        if not self.can_write:
            return

        self._save_index()

        if self._lock.is_locked:
            self._lock.release()
            (self.path / "storage.lock").unlink()

    @require_write
    def create_experiment(
        self,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, xr.Dataset]] = None,
        simulation_arguments: Optional[RunArgumentsType] = None,
        name: Optional[str] = None,
    ) -> LocalExperiment:
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
    def _add_migration_information(self, from_version: int, name: str) -> None:
        self._index.migrations.append(
            _Migrations(
                version_range=(from_version, _LOCAL_STORAGE_VERSION),
                name=name,
            )
        )
        self._index.version = _LOCAL_STORAGE_VERSION
        self._save_index()

    @require_write
    def _save_index(self) -> None:
        with open(self.path / "index.json", mode="w", encoding="utf-8") as f:
            print(self._index.model_dump_json(), file=f)

    @require_write
    def _migrate(self, ignore_migration_check: bool) -> None:
        if ignore_migration_check:
            return
        from ert.storage.migration import (  # noqa: PLC0415
            block_fs,
            empty_summary,
            ert_kind,
            experiment_id,
            gen_kw,
            observations,
            response_info,
            update_params,
        )

        try:
            version = _storage_version(self.path)
            self._index = self._load_index()
            if version == 0:
                block_fs.migrate(self.path)
                experiment_id.migrate(self.path)
                observations.migrate(self.path)
                self._add_migration_information(0, "block_fs")
            elif version == 1:

                experiment_id.migrate(self.path)
                gen_kw.migrate(self.path)
                response_info.migrate(self.path)
                observations.migrate(self.path)
                update_params.migrate(self.path)
                ert_kind.migrate(self.path)
                self._add_migration_information(1, "gen_kw")
            elif version == 2:
                gen_kw.migrate(self.path)
                experiment_id.migrate(self.path)
                response_info.migrate(self.path)
                observations.migrate(self.path)
                update_params.migrate(self.path)
                self._add_migration_information(2, "response")
            elif version == 3:

                gen_kw.migrate(self.path)
                experiment_id.migrate(self.path)
                observations.migrate(self.path)
                update_params.migrate(self.path)
                self._add_migration_information(3, "observations")
            elif version == 4:
                gen_kw.migrate(self.path)
                experiment_id.migrate(self.path)
                update_params.migrate(self.path)
                empty_summary.migrate(self.path)
                self._add_migration_information(4, "experiment_id")
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error(f"Migrating storage at {self.path} failed with {err}")


def _storage_version(path: Path) -> Optional[int]:
    if not path.exists():
        logger.warning(f"Unknown storage version in '{path}'")
        return None
    try:
        with open(path / "index.json", encoding="utf-8") as f:
            return int(json.load(f)["version"])
    except KeyError as exc:
        raise NotImplementedError("Incompatible ERT Local Storage") from exc
    except FileNotFoundError:
        if _is_block_storage(path):
            return 0
    logger.warning(f"Unknown storage version in '{path}'")
    return None


_migration_ert_config: Optional[ErtConfig] = None


def local_storage_set_ert_config(ert_config: Optional[ErtConfig]) -> None:
    """
    Set the ErtConfig for migration hints
    """
    global _migration_ert_config  # noqa: PLW0603
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
