from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union
from uuid import UUID

import xtgeo

from ert._c_wrappers.enkf.config.field_config import Field
from ert._c_wrappers.enkf.config.gen_kw_config import GenKwConfig, PriorDict
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.ensemble_config import ParameterConfiguration
    from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader


class LocalExperimentReader:
    def __init__(self, storage: LocalStorageReader, uuid: UUID, path: Path) -> None:
        self._storage: Union[LocalStorageReader, LocalStorageAccessor] = storage
        self._id = uuid
        self._path = path

    @property
    def ensembles(self) -> Generator[LocalEnsembleReader, None, None]:
        yield from (
            ens for ens in self._storage.ensembles if ens.experiment_id == self.id
        )

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def grid_path(self) -> Optional[str]:
        if (self._path / "grid.EGRID").exists():
            return str(self._path / "grid.EGRID")
        if (self._path / "grid.GRID").exists():
            return str(self._path / "grid.GRID")
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
    def field_info(self) -> Dict[str, Any]:
        info: Dict[str, Any]
        path = self.mount_point / "field-info.json"
        if not path.exists():
            raise ValueError("No field info exists")
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
        parameters: Optional[ParameterConfiguration] = None,
    ) -> None:
        self._storage: LocalStorageAccessor = storage

        self._id = uuid
        self._path = path

        parameters = [] if parameters is None else parameters
        for parameter in parameters:
            if isinstance(parameter, GenKwConfig):
                self.save_gen_kw_info(parameter.getKey(), parameter.get_priors())
            elif isinstance(parameter, SurfaceConfig):
                self.save_surface_info(
                    parameter.name,
                    parameter.base_surface_path,
                )
            elif isinstance(parameter, Field):
                self.save_field_info(
                    parameter.name,
                    parameter.grid_file,
                    parameter.file_format,
                    parameter.output_transformation,
                    parameter.truncation_min,
                    parameter.truncation_max,
                    parameter.nx,
                    parameter.ny,
                    parameter.nz,
                )
            else:
                raise NotImplementedError("Unknown parameter type")

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

    def save_field_info(  # pylint: disable=too-many-arguments
        self,
        name: str,
        grid_file: str,
        file_format: str,
        transfer_out: str,
        trunc_min: Optional[float],
        trunc_max: Optional[float],
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        info = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "file_format": file_format,
            "transfer_out": transfer_out,
            "truncation_min": trunc_min,
            "truncation_max": trunc_max,
        }
        # Grid file is shared between all FIELD keywords, so we can avoid
        # copying for each FIELD keyword.
        if grid_file is not None:
            grid_filename = "grid" + Path(grid_file).suffix.upper()
            if not (self._path / grid_filename).exists():
                shutil.copy(grid_file, self._path / grid_filename)

        field_info_path = self._path / "field-info.json"
        field_info = {}
        if field_info_path.exists():
            with open(field_info_path, encoding="utf-8") as f:
                field_info = json.load(f)
        field_info.update({name: info})
        with open(field_info_path, encoding="utf-8", mode="w") as f:
            json.dump(field_info, f)

    def save_surface_info(self, name: str, base_surface: str) -> None:
        surf = xtgeo.surface_from_file(base_surface, fformat="irap_ascii")
        surf.to_file(self.mount_point / f"{name}.irap", fformat="irap_ascii")

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
