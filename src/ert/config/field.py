from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast, overload

import networkx as nx
import numpy as np
import xarray as xr
import xtgeo
from pydantic import field_serializer

from ert.field_utils import (
    FieldFileFormat,
    GridGeometry,
    Shape,
    calculate_grid_geometry,
    get_shape,
    read_field,
    save_field,
)
from ert.substitutions import substitute_runpath_name
from ert.utils import log_duration

from ._str_to_bool import str_to_bool
from .parameter_config import ParameterConfig
from .parsing import ConfigValidationError, ConfigWarning

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

_logger = logging.getLogger(__name__)


def create_flattened_cube_graph(px: int, py: int, pz: int) -> nx.Graph[int]:
    """graph created with nodes numbered from 0 to px*py*pz
    corresponds to the "vectorization" or flattening of
    a 3D cube with shape (px,py,pz) in the same way as
    reshaping such a cube into a one-dimensional array.
    The indexing scheme used to create the graph reflects
    this flattening process"""

    G: nx.Graph[int] = nx.Graph()
    for x, y, z in itertools.product(range(px), range(py), range(pz)):
        # Flatten the 3D index to a single index
        index = x * py * pz + y * pz + z

        # Connect to the right neighbor (y-direction)
        if y < py - 1:
            G.add_edge(index, index + pz)

        # Connect to the bottom neighbor (x-direction)
        if x < px - 1:
            G.add_edge(index, index + py * pz)

        # Connect to the neighbor in front (z-direction)
        if z < pz - 1:
            G.add_edge(index, index + 1)

    return G


class Field(ParameterConfig):
    type: Literal["field"] = "field"
    dimensionality: Literal[3] = 3
    grid_geometry: GridGeometry
    file_format: FieldFileFormat
    output_transformation: str | None = None
    input_transformation: str | None = None
    truncation_min: float | None = None
    truncation_max: float | None = None
    forward_init_file: str
    output_file: Path
    grid_file: str

    @field_serializer("output_file")
    def serialize_output_file(self, path: Path) -> str:
        return str(path)

    @property
    def parameter_keys(self) -> list[str]:
        return []

    @classmethod
    def from_config_list(
        cls,
        grid_file_path: str,
        config_list: list[str | dict[str, str]],
    ) -> Self:
        name = cast(str, config_list[0])
        out_file_name = cast(str, config_list[2])
        out_file = Path(out_file_name)
        options = cast(dict[str, str], config_list[3])
        init_transform = options.get("INIT_TRANSFORM")
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        output_transform = options.get("OUTPUT_TRANSFORM")
        input_transform = options.get("INPUT_TRANSFORM")
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        min_ = options.get("MIN")
        max_ = options.get("MAX")
        init_files = options.get("INIT_FILES")
        if input_transform:
            ConfigWarning.warn(
                f"Got INPUT_TRANSFORM for FIELD: {name}, "
                f"this has no effect and can be removed",
                config_list,
            )

        errors = []

        if init_transform and init_transform not in TRANSFORM_FUNCTIONS:
            errors.append(
                ConfigValidationError.with_context(
                    f"FIELD INIT_TRANSFORM:{init_transform} is an invalid function",
                    config_list,
                )
            )
        if output_transform and output_transform not in TRANSFORM_FUNCTIONS:
            errors.append(
                ConfigValidationError.with_context(
                    f"FIELD OUTPUT_TRANSFORM:{output_transform} is an invalid function",
                    config_list,
                )
            )
        file_extension = out_file.suffix[1:].upper()
        if not out_file.suffix:
            errors.append(
                ConfigValidationError.with_context(
                    f"Missing extension for field output file '{out_file}', "
                    f"valid formats are: {[f.name for f in FieldFileFormat]}",
                    out_file_name,
                )
            )
        file_format = None
        try:
            file_format = FieldFileFormat[file_extension]
        except KeyError:
            errors.append(
                ConfigValidationError.with_context(
                    f"Unknown file format for output file: {out_file.suffix!r},"
                    f" valid formats: {[f.name for f in FieldFileFormat]}",
                    out_file_name,
                )
            )
        if init_files is None:
            errors.append(
                ConfigValidationError.with_context(
                    f"Missing required INIT_FILES for field {name!r}", config_list
                )
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)
        assert file_format is not None

        assert init_files is not None

        grid_extension = Path(grid_file_path).suffix.lower()

        try:
            if grid_extension == ".egrid":
                grid = xtgeo.grid_from_file(grid_file_path)
                grid_geometry = calculate_grid_geometry(grid)
            else:
                dims = get_shape(grid_file_path)

                if dims is None:
                    raise ConfigValidationError.with_context(
                        f"Grid file {grid_file_path} did not contain dimensions",
                        grid_file_path,
                    )

                grid_geometry = GridGeometry(dims.nx, dims.ny, dims.nz)
        except Exception as err:
            raise ConfigValidationError.with_context(
                f"Could not read grid file {grid_file_path}: {err}",
                grid_file_path,
            ) from err

        return cls(
            name=name,
            grid_geometry=grid_geometry,
            file_format=file_format,
            output_transformation=output_transform,
            input_transformation=init_transform,
            truncation_max=float(max_) if max_ is not None else None,
            truncation_min=float(min_) if min_ is not None else None,
            forward_init=forward_init,
            forward_init_file=init_files,
            output_file=out_file,
            grid_file=os.path.abspath(grid_file_path),
            update=update_parameter,
        )

    def __len__(self) -> int:
        return self.grid_geometry.nx * self.grid_geometry.ny * self.grid_geometry.nz

    @log_duration(_logger, custom_name="load_field")
    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> xr.Dataset:
        file_name = substitute_runpath_name(self.forward_init_file, real_nr, iteration)
        ds = xr.Dataset(
            {
                "values": (
                    ["x", "y", "z"],
                    field_transform(
                        read_field(
                            run_path / file_name,
                            self.name,
                            Shape(
                                self.grid_geometry.nx,
                                self.grid_geometry.ny,
                                self.grid_geometry.nz,
                            ),
                        ),
                        self.input_transformation,
                    ),
                )
            }
        )
        return ds

    @log_duration(_logger, custom_name="save_field")
    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> None:
        file_out = run_path.joinpath(
            substitute_runpath_name(str(self.output_file), real_nr, ensemble.iteration)
        )
        if os.path.islink(file_out):
            os.unlink(file_out)

        save_field(
            self._transform_data(self._fetch_from_ensemble(real_nr, ensemble)),
            self.name,
            file_out,
            self.file_format,
        )

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[int, xr.Dataset]]:
        dim_nx, dim_ny, dim_nz = (
            self.grid_geometry.nx,
            self.grid_geometry.ny,
            self.grid_geometry.nz,
        )

        for i, realization in enumerate(iens_active_index):
            values = from_data[:, i].reshape((dim_nx, dim_ny, dim_nz))
            ds = xr.Dataset({"values": (["x", "y", "z"], values)})
            yield int(realization), ds

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        ds = ensemble.load_parameters(self.name, realizations)
        assert isinstance(ds, xr.Dataset)
        ensemble_size = len(ds.realizations)
        da = xr.DataArray(
            [
                np.ma.MaskedArray(data=d).compressed()
                for d in ds["values"].values.reshape(ensemble_size, -1)
            ]
        )
        return da.T.to_numpy()

    def _fetch_from_ensemble(self, real_nr: int, ensemble: Ensemble) -> xr.DataArray:
        da = ensemble.load_parameters(self.name, real_nr)["values"]
        assert isinstance(da, xr.DataArray)
        return da

    def _transform_data(
        self, data_array: xr.DataArray
    ) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
        return np.ma.MaskedArray(
            _field_truncate(
                field_transform(
                    data_array,
                    transform_name=self.output_transformation,
                ),
                self.truncation_min,
                self.truncation_max,
            ),
            fill_value=np.nan,
        )

    def load_parameter_graph(self) -> nx.Graph[int]:
        parameter_graph = create_flattened_cube_graph(
            px=self.grid_geometry.nx,
            py=self.grid_geometry.ny,
            pz=self.grid_geometry.nz,
        )
        new_labels = {
            old_label: new_label
            for new_label, old_label in enumerate(parameter_graph.nodes())
        }
        return nx.relabel_nodes(parameter_graph, new_labels, copy=True)

    @property
    def nx(self) -> int:
        return self.grid_geometry.nx

    @property
    def ny(self) -> int:
        return self.grid_geometry.ny

    @property
    def nz(self) -> int:
        return self.grid_geometry.nz


TRANSFORM_FUNCTIONS: Final[dict[str, Callable[[Any], Any]]] = {
    "LN": np.log,
    "LOG": np.log,
    "LN0": lambda v: np.log(v + 0.000001),
    "LOG10": np.log10,
    "EXP": np.exp,
    "EXP0": lambda v: np.exp(v) - 0.000001,
    "POW10": lambda v: np.power(10.0, v),
    "TRUNC_POW10": lambda v: np.maximum(np.power(10, v), 0.001),
}


@overload
def field_transform(
    data: xr.DataArray, transform_name: str | None
) -> npt.NDArray[np.float32] | xr.DataArray:
    pass


@overload
def field_transform(
    data: npt.NDArray[np.float32], transform_name: str | None
) -> npt.NDArray[np.float32]:
    pass


def field_transform(
    data: xr.DataArray | npt.NDArray[np.float32], transform_name: str | None
) -> npt.NDArray[np.float32] | xr.DataArray:
    if transform_name is None:
        return data
    return TRANSFORM_FUNCTIONS[transform_name](data)


def _field_truncate(data: npt.ArrayLike, min_: float | None, max_: float | None) -> Any:
    if min_ is not None and max_ is not None:
        return np.clip(data, min_, max_)
    elif min_ is not None:
        return np.maximum(data, min_)
    elif max_ is not None:
        return np.minimum(data, max_)
    return data
