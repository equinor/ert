from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import networkx as nx
import numpy as np
import xarray as xr
import xtgeo

from ert.substitutions import substitute_runpath_name

from ._str_to_bool import str_to_bool
from .field import create_flattened_cube_graph
from .parameter_config import ParameterConfig, ParameterMetadata
from .parsing import ConfigValidationError, ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


@dataclass
class SurfaceConfig(ParameterConfig):
    ncol: int
    nrow: int
    xori: float
    yori: float
    xinc: float
    yinc: float
    rotation: float
    yflip: int
    forward_init_file: str
    output_file: Path
    base_surface_path: str

    @property
    def parameter_keys(self) -> list[str]:
        return []

    @property
    def metadata(self) -> list[ParameterMetadata]:
        return [
            ParameterMetadata(
                key=self.name,
                dimensionality=2,
                transformation=None,
                userdata={
                    "data_origin": "SURFACE",
                    "nx": self.ncol,
                    "ny": self.nrow,
                },
            )
        ]

    @classmethod
    def from_config_list(cls, surface: list[str | dict[str, str]]) -> Self:
        name = cast(str, surface[0])
        options = cast(dict[str, str], surface[1])
        init_file = options.get("INIT_FILES")
        out_file = options.get("OUTPUT_FILE")
        base_surface = options.get("BASE_SURFACE")
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        errors = []
        if not out_file:
            errors.append(
                ErrorInfo("Missing required OUTPUT_FILE").set_context(surface)
            )
        if not init_file:
            errors.append(ErrorInfo("Missing required INIT_FILES").set_context(surface))
        elif not forward_init and "%d" not in init_file:
            errors.append(
                ErrorInfo(
                    "INIT_FILES must contain %d when FORWARD_INIT:FALSE"
                ).set_context(surface)
            )
        if not base_surface:
            errors.append(
                ErrorInfo("Missing required BASE_SURFACE").set_context(surface)
            )
        elif not Path(base_surface).exists():
            errors.append(
                ErrorInfo(f"BASE_SURFACE:{base_surface} not found").set_context(surface)
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        assert init_file is not None
        assert out_file is not None
        assert base_surface is not None
        try:
            surf = xtgeo.surface_from_file(
                base_surface, fformat="irap_ascii", dtype=np.float32
            )
        except Exception as err:
            raise ConfigValidationError.with_context(
                f"Could not load surface {base_surface!r}", surface
            ) from err
        return cls(
            ncol=surf.ncol,
            nrow=surf.nrow,
            xori=surf.xori,
            yori=surf.yori,
            xinc=surf.xinc,
            yinc=surf.yinc,
            rotation=surf.rotation,
            yflip=surf.yflip,
            name=name,
            forward_init=forward_init,
            forward_init_file=init_file,
            output_file=Path(out_file),
            base_surface_path=base_surface,
            update=update_parameter,
        )

    def __len__(self) -> int:
        return self.ncol * self.nrow

    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> xr.Dataset:
        file_name = substitute_runpath_name(self.forward_init_file, real_nr, iteration)
        file_path = run_path / file_name
        if not file_path.exists():
            raise ValueError(
                "Failed to initialize parameter "
                f"'{self.name}' in file {file_name}: "
                "File not found\n"
            )
        surface = xtgeo.surface_from_file(
            file_path, fformat="irap_ascii", dtype=np.float32
        )

        da = xr.DataArray(
            surface.values,
            name="values",
            dims=["x", "y"],
        )

        return da.to_dataset()

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> None:
        ds = ensemble.load_parameters(self.name, real_nr)
        assert isinstance(ds, xr.Dataset)
        data = ds["values"]

        surf = xtgeo.RegularSurface(
            ncol=self.ncol,
            nrow=self.nrow,
            xori=self.xori,
            yori=self.yori,
            xinc=self.xinc,
            yinc=self.yinc,
            rotation=self.rotation,
            yflip=self.yflip,
            values=data.values,
        )

        file_path = run_path / substitute_runpath_name(
            str(self.output_file), real_nr, ensemble.iteration
        )
        file_path.parent.mkdir(exist_ok=True, parents=True)
        surf.to_file(file_path, fformat="irap_ascii")

    def save_parameters(
        self,
        ensemble: Ensemble,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        ds = xr.Dataset(
            {
                "values": (
                    ["x", "y"],
                    data.reshape(self.ncol, self.nrow).astype("float32"),
                )
            }
        )
        ensemble.save_parameters(self.name, realization, ds)

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        ds = ensemble.load_parameters(self.name, realizations)
        assert isinstance(ds, xr.Dataset)
        ensemble_size = len(ds.realizations)
        return ds["values"].values.reshape(ensemble_size, -1).T

    def load_parameter_graph(self) -> nx.Graph[int]:
        """Graph created with nodes numbered from 0 to px*py
        corresponds to the "vectorization" or flattening of
        a 2D cube with shape (px,py) in the same way as
        reshaping such a surface into a one-dimensional array.
        The indexing scheme used to create the graph reflects
        this flattening process"""

        return create_flattened_cube_graph(px=self.ncol, py=self.nrow, pz=1)
