from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import xarray as xr
import xtgeo
from typing_extensions import Self

from ._option_dict import option_dict
from ._str_to_bool import str_to_bool
from .parameter_config import ParameterConfig
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

    @classmethod
    def from_config_list(cls, surface: List[str]) -> Self:
        options = option_dict(surface, 1)
        name = surface[0]
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

    def read_from_runpath(self, run_path: Path, real_nr: int) -> xr.Dataset:
        file_name = self.forward_init_file
        if "%d" in file_name:
            file_name = file_name % real_nr  # noqa
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
        data = ensemble.load_parameters(self.name, real_nr)["values"]

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

        file_path = run_path / self.output_file
        file_path.parent.mkdir(exist_ok=True, parents=True)
        surf.to_file(file_path, fformat="irap_ascii")

    def save_parameters(
        self,
        ensemble: Ensemble,
        group: str,
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
        ensemble.save_parameters(group, realization, ds)

    @staticmethod
    def load_parameters(
        ensemble: Ensemble, group: str, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        ds = ensemble.load_parameters(group, realizations)
        ensemble_size = len(ds.realizations)
        return ds["values"].values.reshape(ensemble_size, -1).T
