from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
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
    no_type_check,
)
from uuid import UUID, uuid4

import xarray as xr
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from ert.config import ErtConfig
from ert.realization_state import RealizationState
from ert.shared import __version__
from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
from ert.storage.local_experiment import LocalExperimentAccessor, LocalExperimentReader

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if TYPE_CHECKING:
    from ert.config import ParameterConfig, ResponseConfig


logger = logging.getLogger(__name__)

_LOCAL_STORAGE_VERSION = 4


class _Migrations(BaseModel):
    ert_version: str = __version__
    timestamp: datetime = Field(default_factory=datetime.now)
    name: str
    version_range: Tuple[int, int]


class _Index(BaseModel):
    version: int = _LOCAL_STORAGE_VERSION
    migrations: MutableSequence[_Migrations] = Field(default_factory=list)


class LocalStorageReader:
    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        self.path = Path(path).absolute()

        self._experiments: Union[
            Dict[UUID, LocalExperimentReader], Dict[UUID, LocalExperimentAccessor]
        ]
        self._ensembles: Union[
            Dict[UUID, LocalEnsembleReader], Dict[UUID, LocalEnsembleAccessor]
        ]
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

    def _load_experiments(self) -> Dict[UUID, LocalExperimentReader]:
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
        if not ignore_migration_check:
            try:
                version = _storage_version(self.path)
                if version == 0:
                    from ert.storage.migration import (  # pylint: disable=C0415
                        block_fs,
                        observations,
                    )

                    block_fs.migrate(self.path)
                    observations.migrate(self.path)
                    self._add_migration_information(0, "block_fs")
                elif version == 1:
                    from ert.storage.migration import (  # pylint: disable=C0415
                        gen_kw,
                        observations,
                        response_info,
                    )

                    gen_kw.migrate(self.path)
                    response_info.migrate(self.path)
                    observations.migrate(self.path)
                    self._add_migration_information(1, "gen_kw")
                elif version == 2:
                    from ert.storage.migration import (  # pylint: disable=C0415
                        observations,
                        response_info,
                    )

                    response_info.migrate(self.path)
                    observations.migrate(self.path)
                    self._add_migration_information(2, "response")
                elif version == 3:
                    from ert.storage.migration import (  # pylint: disable=C0415
                        observations,
                    )

                    observations.migrate(self.path)
                    self._add_migration_information(3, "observations")
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.error(f"Migrating storage at {self.path} failed with {err}")

        self.path.mkdir(parents=True, exist_ok=True)

        # ERT 4 checks that this file exists and if it exists tells the user
        # that their ERT storage is incompatible
        with contextlib.suppress(FileExistsError):
            (self.path / ".fs_version").symlink_to("index.json")

        self._lock = FileLock(self.path / "storage.lock")
        try:
            self._lock.acquire(timeout=self.LOCK_TIMEOUT)
        except Timeout as e:
            raise TimeoutError(
                f"Not able to acquire lock for: {self.path}."
                " You may already be running ERT,"
                " or another user is using the same ENSPATH."
            ) from e

        super().__init__(path)

        self._save_index()

    def close(self) -> None:
        self._save_index()
        super().close()

        if self._lock.is_locked:
            self._lock.release()
            (self.path / "storage.lock").unlink()

    def to_accessor(self) -> LocalStorageAccessor:
        return self

    def create_experiment(
        self,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, xr.Dataset]] = None,
    ) -> LocalExperimentAccessor:
        exp_id = uuid4()
        path = self._experiment_path(exp_id)
        path.mkdir(parents=True, exist_ok=False)
        exp = LocalExperimentAccessor(
            self,
            exp_id,
            path,
            parameters=parameters,
            responses=responses,
            observations=observations,
        )
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
    ) -> LocalEnsembleAccessor:
        experiment_id = experiment if isinstance(experiment, UUID) else experiment.id

        uuid = uuid4()
        path = self._ensemble_path(uuid)
        path.mkdir(parents=True, exist_ok=False)

        prior_ensemble_id: Optional[UUID] = None
        if isinstance(prior_ensemble, UUID):
            prior_ensemble_id = prior_ensemble
        elif isinstance(prior_ensemble, LocalEnsembleReader):
            prior_ensemble_id = prior_ensemble.id
        prior_ensemble = (
            self.get_ensemble(prior_ensemble_id) if prior_ensemble_id else None
        )
        if prior_ensemble and ensemble_size > prior_ensemble.ensemble_size:
            raise ValueError(
                f"New ensemble ({ensemble_size}) must be of equal or "
                f"smaller size than parent ensemble ({prior_ensemble.ensemble_size})"
            )
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
        if prior_ensemble:
            state_map = prior_ensemble.state_map
            for realization_nr, state in enumerate(state_map[: len(ens.state_map)]):
                if state in [
                    RealizationState.LOAD_FAILURE,
                    RealizationState.PARENT_FAILURE,
                    RealizationState.UNDEFINED,
                ]:
                    ens.state_map[realization_nr] = RealizationState.PARENT_FAILURE
        self._ensembles[ens.id] = ens
        return ens

    def _add_migration_information(self, from_version: int, name: str) -> None:
        self._index.migrations.append(
            _Migrations(
                version_range=(from_version, _LOCAL_STORAGE_VERSION),
                name=name,
            )
        )
        self._save_index()

    def _save_index(self) -> None:
        if not hasattr(self, "_index"):
            return

        with open(self.path / "index.json", mode="w", encoding="utf-8") as f:
            print(self._index.json(), file=f)

    def _load_experiment(self, uuid: UUID) -> LocalExperimentAccessor:
        return LocalExperimentAccessor(self, uuid, self._experiment_path(uuid))

    def _load_ensemble(self, path: Path) -> LocalEnsembleAccessor:
        return LocalEnsembleAccessor(self, path)


def _storage_version(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with open(path / "index.json", encoding="utf-8") as f:
            return int(json.load(f)["version"])
    except KeyError as exc:
        raise NotImplementedError("Incompatible ERT Local Storage") from exc
    except FileNotFoundError:
        if _is_block_storage(path):
            return 0
    raise ValueError("Unknown storage version")


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
