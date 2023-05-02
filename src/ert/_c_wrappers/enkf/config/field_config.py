from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

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

_logger = logging.getLogger(__name__)


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
        t = time.perf_counter()
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
        _logger.debug(f"load() time_used {(time.perf_counter() - t):.4f}s")

    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        t = time.perf_counter()
        file_out = run_path.joinpath(self.output_file)
        if os.path.islink(file_out):
            os.unlink(file_out)
        ensemble.export_field(self.name, real_nr, file_out)
        _logger.debug(f"save() time_used {(time.perf_counter() - t):.4f}s")


# pylint: disable=unnecessary-lambda
_TRANSFORM_FUNCTIONS = {
    "LN": lambda x: math.log(x, math.e),
    "LOG": lambda x: math.log(x, math.e),
    "LN0": lambda x: math.log(x + 0.000001, math.e),
    "LOG10": lambda x: math.log(x, 10),
    "EXP": lambda x: math.exp(x),
    "EXP0": lambda x: math.exp(x) - 0.000001,
    "POW10": lambda x: math.log(x, math.e),
    "TRUNC_POW10": lambda x: math.pow(max(x, 0.001), 10),
}

TRANSFORM_FUNCTIONS = {k: np.vectorize(v) for k, v in _TRANSFORM_FUNCTIONS.items()}


def field_transform(data: npt.ArrayLike, transform_name: str) -> npt.ArrayLike:
    return TRANSFORM_FUNCTIONS[transform_name](data)
