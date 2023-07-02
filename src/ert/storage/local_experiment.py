from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union
from uuid import UUID

import xtgeo

from ert._c_wrappers.enkf.config.field_config import Field
from ert._c_wrappers.enkf.config.gen_kw_config import GenKwConfig, PriorDict
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
    from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader


class LocalExperimentReader:
    def __init__(self, storage: LocalStorageReader, uuid: UUID, path: Path) -> None:
        self._storage: Union[LocalStorageReader, LocalStorageAccessor] = storage
        self._id = uuid
        self._path = path
        self._parameter_file = Path("parameter.json")

    @property
    def ensembles(self) -> Generator[LocalEnsembleReader, None, None]:
        yield from (
            ens for ens in self._storage.ensembles if ens.experiment_id == self.id
        )

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def grid_path(self) -> Optional[Path]:
        if (self._path / "grid.EGRID").exists():
            return self._path / "grid.EGRID"
        if (self._path / "grid.GRID").exists():
            return self._path / "grid.GRID"
        return None

    @property
    def mount_point(self) -> Path:
        return self._path

    @property
    def gen_kw_info(self) -> Dict[str, Any]:
        priors: Dict[str, Any]
        if Path.exists(self.mount_point / "gen-kw-priors.json"):
            with open(
                self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8"
            ) as f:
                priors = json.load(f)
        else:
            raise ValueError("No GEN_KW in experiment")
        return priors

    @property
    def parameter_info(self) -> Dict[str, Any]:
        info: Dict[str, Any]
        path = self.mount_point / self._parameter_file
        if not path.exists():
            raise ValueError(f"{str(self._parameter_file)} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        return info

    def load_gen_kw_priors(self) -> Dict[str, List[PriorDict]]:
        with open(self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8") as f:
            priors: Dict[str, List[PriorDict]] = json.load(f)
        return priors

    def get_surface(self, name: str) -> xtgeo.RegularSurface:
        return xtgeo.surface_from_file(
            str(self.mount_point / f"{name}.irap"), fformat="irap_ascii"
        )


class LocalExperimentAccessor(LocalExperimentReader):
    def __init__(
        self,
        storage: LocalStorageAccessor,
        uuid: UUID,
        path: Path,
        parameters: Optional[List[ParameterConfig]] = None,
    ) -> None:
        self._storage: LocalStorageAccessor = storage
        self._id = uuid
        self._path = path
        self._parameter_file = Path("parameter.json")

        parameter_data = {}
        parameters = [] if parameters is None else parameters
        for parameter in parameters:
            parameter.save_experiment_data(self._path)
            if isinstance(parameter, GenKwConfig):
                self.save_gen_kw_info(parameter.name, parameter.get_priors())
            elif isinstance(parameter, SurfaceConfig):
                parameter_data[parameter.name] = parameter.to_dict()
            elif isinstance(parameter, Field):
                parameter_data[parameter.name] = parameter.to_dict()

                # Grid file is shared between all FIELD keywords, so we can avoid
                # copying for each FIELD keyword.
                if parameter.grid_file is not None:
                    grid_filename = "grid" + Path(parameter.grid_file).suffix.upper()
                    if not (self._path / grid_filename).exists():
                        shutil.copy(parameter.grid_file, self._path / grid_filename)

            else:
                raise NotImplementedError("Unknown parameter type")

        with open(self.mount_point / self._parameter_file, "w", encoding="utf-8") as f:
            json.dump(parameter_data, f)

    @property
    def ensembles(self) -> Generator[LocalEnsembleAccessor, None, None]:
        yield from super().ensembles  # type: ignore

    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: str,
        prior_ensemble: Optional[LocalEnsembleReader] = None,
    ) -> LocalEnsembleAccessor:
        return self._storage.create_ensemble(
            self,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

    def save_gen_kw_info(
        self, name: str, parameter_transfer_functions: List["PriorDict"]
    ) -> None:
        priors = {}
        if Path.exists(self.mount_point / "gen-kw-priors.json"):
            with open(
                self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8"
            ) as f:
                priors = json.load(f)
        priors.update({name: parameter_transfer_functions})
        with open(self.mount_point / "gen-kw-priors.json", "w", encoding="utf-8") as f:
            json.dump(priors, f)
