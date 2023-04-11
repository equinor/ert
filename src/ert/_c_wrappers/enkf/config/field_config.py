from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cwrap
import numpy as np
import xtgeo
from ecl.eclfile import EclKW
from ecl.grid import EclGrid
from numpy import ma

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import EnsembleAccessor, EnsembleReader


@dataclass
class Field(ParameterConfig):
    nx: int
    ny: int
    nz: int
    file_format: str
    output_transformation: str
    input_transformation: str
    truncation_min: Optional[float]
    truncation_max: Optional[float]
    forward_init_file: str
    output_file: Path
    grid_file: str

    def load(self, run_path: Path, real_nr: int, ensemble: EnsembleAccessor):
        file_name = self.forward_init_file
        if "%d" in file_name:
            file_name = file_name % real_nr
        file_path = run_path / file_name

        key = self.name
        grid = ensemble.experiment.grid
        if isinstance(grid, xtgeo.Grid):
            try:
                props = xtgeo.gridproperty_from_file(
                    pfile=file_path,
                    name=key,
                    grid=grid,
                )
                data = props.get_npvalues1d(order="C", fill_value=np.nan)
            except PermissionError as err:
                msg = f"Failed to open init file for parameter {key}"
                raise RuntimeError(msg) from err
        elif isinstance(grid, EclGrid):
            with cwrap.open(str(file_path), "rb") as f:
                param = EclKW.read_grdecl(f, self.name)
            mask = [not e for e in grid.export_actnum()]
            masked_array = ma.MaskedArray(
                data=param.numpy_view(), mask=mask, fill_value=np.nan
            )
            data = masked_array.filled()

        trans = self.input_transformation
        data_transformed = field_transform(data, trans) if trans else data
        ensemble.save_field(key, real_nr, data_transformed)

    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        file_out = run_path.joinpath(self.output_file)
        if os.path.islink(file_out):
            os.unlink(file_out)
        ensemble.export_field(self.name, real_nr, file_out)


VALID_TRANSFORMATIONS = [
    "LN",
    "LOG",
    "LN0",
    "LOG10",
    "EXP",
    "EXP0",
    "POW10",
    "TRUNC_POW10",
]


def field_transform(data: npt.ArrayLike, transform_name: VALID_TRANSFORMATIONS) -> Any:
    def f(x: float) -> float:  # pylint: disable=too-many-return-statements
        if transform_name in ("LN", "LOG"):
            return math.log(x, math.e)
        if transform_name == "LN0":
            return math.log(x + 0.000001, math.e)
        if transform_name == "LOG10":
            return math.log(x, 10)
        if transform_name == "EXP":
            return math.exp(x)
        if transform_name == "EXP0":
            return math.exp(x) - 0.000001
        if transform_name == "POW10":
            return math.pow(x, 10)
        if transform_name == "TRUNC_POW10":
            return math.pow(max(x, 0.001), 10)
        return x

    vfunc = np.vectorize(f)

    return vfunc(data)
