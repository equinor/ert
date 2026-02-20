from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import networkx as nx
import numpy as np
import xarray as xr
from pydantic import field_serializer
from surfio import IrapHeader, IrapSurface

from ert.config.parsing.config_errors import ConfigWarning
from ert.substitutions import substitute_runpath_name

from ._str_to_bool import str_to_bool
from .field import create_flattened_cube_graph
from .parameter_config import InvalidParameterFile, ParameterConfig
from .parsing import ConfigValidationError, ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


class SurfaceMismatchError(InvalidParameterFile):
    def __init__(
        self,
        surface_name: str,
        base_surface_path: str,
        surface_size: int,
        realization: int,
        base_surface_col: int,
        base_surface_row: int,
    ) -> None:
        base_surface_name = base_surface_path.rsplit("/", maxsplit=1)[-1]
        msg = (
            f"Saving parameters for SURFACE '{surface_name}' for realization "
            f"{int(realization)} to storage failed. "
            f"SURFACE {surface_name} (size {surface_size}) and BASE_SURFACE "
            f"{base_surface_name} (size {base_surface_row * base_surface_col}) "
            f"must have the same size."
        )
        super().__init__(msg)


ASCII_SURFACE_WARNING_MESSAGE = (
    "The default format for surfaces is changing from ASCII "
    "to binary, but Ert will temporarily still support ASCII surfaces. Binary "
    "surfaces are recommended, as it is faster to write and read from runpath. "
    "Forward model steps still importing/exporting surfaces in "
    "ASCII format need to be updated. %s"
)

logger = logging.getLogger(__name__)


class SurfaceConfig(ParameterConfig):
    type: Literal["surface"] = "surface"
    dimensionality: Literal[2] = 2
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
    file_format: Literal["ascii", "binary"] = "ascii"

    @field_serializer("output_file")
    def serialize_output_file(self, output_file: Path) -> str:
        return str(output_file)

    @field_serializer("base_surface_path")
    def serialize_base_surface_path(self, base_surface_path: Path) -> str:
        return str(base_surface_path)

    @property
    def parameter_keys(self) -> list[str]:
        return []

    @classmethod
    def from_config_list(cls, config_list: list[str | dict[str, str]]) -> Self:
        name = cast(str, config_list[0])
        options = cast(dict[str, str], config_list[1])
        init_file = options.get("INIT_FILES")
        out_file = options.get("OUTPUT_FILE")
        base_surface = options.get("BASE_SURFACE")
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        errors = []
        if not out_file:
            errors.append(
                ErrorInfo("Missing required OUTPUT_FILE").set_context(config_list)
            )
        if not init_file:
            errors.append(
                ErrorInfo("Missing required INIT_FILES").set_context(config_list)
            )
        elif not forward_init and not ("%d" in init_file or "<IENS>" in init_file):
            errors.append(
                ErrorInfo(
                    "INIT_FILES must contain %d or <IENS> when FORWARD_INIT:FALSE"
                ).set_context(config_list)
            )
        if not base_surface:
            errors.append(
                ErrorInfo("Missing required BASE_SURFACE").set_context(config_list)
            )
        elif not Path(base_surface).exists():
            errors.append(
                ErrorInfo(f"BASE_SURFACE:{base_surface} not found").set_context(
                    config_list
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        assert init_file is not None
        assert out_file is not None
        assert base_surface is not None
        file_format = "binary"
        try:
            try:
                surf = IrapSurface.from_binary_file(Path(base_surface))
                logger.info("Loaded surface in binary format")
            except Exception:
                surf = IrapSurface.from_ascii_file(Path(base_surface))
                logger.info("Loaded surface in ascii format")
                ConfigWarning.warn(ASCII_SURFACE_WARNING_MESSAGE % base_surface)
                file_format = "ascii"
            yflip = -1 if surf.header.yinc < 0 else 1
        except Exception as err:
            raise ConfigValidationError.with_context(
                f"Could not load surface {base_surface!r}", config_list
            ) from err
        return cls(
            ncol=surf.header.ncol,
            nrow=surf.header.nrow,
            xori=surf.header.xori,
            yori=surf.header.yori,
            xinc=surf.header.xinc,
            yinc=surf.header.yinc * yflip,
            rotation=surf.header.rot,
            yflip=yflip,
            name=name,
            forward_init=forward_init,
            forward_init_file=init_file,
            output_file=Path(out_file),
            base_surface_path=base_surface,
            update=update_parameter,
            file_format=file_format,
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
        if self.file_format == "ascii":
            surface = IrapSurface.from_ascii_file(file_path)
        elif self.file_format == "binary":
            surface = IrapSurface.from_binary_file(file_path)
        else:
            raise ValueError(f"Unsupported file format {self.file_format}")
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

        yinc = self.yinc * self.yflip
        surf = IrapSurface(
            header=IrapHeader(
                ncol=self.ncol,
                nrow=self.nrow,
                xori=self.xori,
                yori=self.yori,
                xinc=self.xinc,
                yinc=yinc,
                rot=self.rotation,
                xrot=self.xori,
                yrot=self.yori,
            ),
            values=data.values,
        )

        file_path = run_path / substitute_runpath_name(
            str(self.output_file), real_nr, ensemble.iteration
        )
        file_path.parent.mkdir(exist_ok=True, parents=True)
        if self.file_format == "ascii":
            surf.to_ascii_file(file_path)
        elif self.file_format == "binary":
            surf.to_binary_file(file_path)
        else:
            raise ValueError(f"Unsupported file format {self.file_format}")

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[int, xr.Dataset]]:
        for i, realization in enumerate(iens_active_index):
            surface_parameters = from_data[:, i]
            surface_size = surface_parameters.size
            base_surface_size = self.nrow * self.ncol
            if surface_size != base_surface_size:
                raise SurfaceMismatchError(
                    self.name,
                    self.base_surface_path,
                    surface_size,
                    realization,
                    self.ncol,
                    self.nrow,
                ) from None
            ds = xr.Dataset(
                {
                    "values": (
                        ["x", "y"],
                        surface_parameters.reshape(self.ncol, self.nrow).astype(
                            "float32"
                        ),
                    )
                }
            )
            yield int(realization), ds

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
