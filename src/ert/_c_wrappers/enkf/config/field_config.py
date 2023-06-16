from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert.storage.field_utils.field_utils import Shape, get_masked_field, save_field

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

        grid_path = ensemble.experiment.grid_path
        assert grid_path is not None
        data = get_masked_field(file_path, self.name, grid_path)
        data = field_transform(data, self.input_transformation)
        ensemble.save_parameters(self.name, real_nr, data)
        _logger.debug(f"load() time_used {(time.perf_counter() - t):.4f}s")

    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        t = time.perf_counter()
        output_path = run_path.joinpath(self.output_file)
        if os.path.islink(output_path):
            os.unlink(output_path)

        fformat = self.file_format
        dataset = ensemble.load_parameters(self.name, real_nr)
        data = field_transform(dataset, self.output_transformation)
        data = self._truncate(data)

        save_field(
            data,
            self.name,
            ensemble.experiment.grid_path,
            Shape(self.nx, self.ny, self.nz),
            output_path,
            fformat,
        )

        _logger.debug(f"save() time_used {(time.perf_counter() - t):.4f}s")

    def _truncate(self, data: npt.ArrayLike) -> npt.NDArray[np.float32]:
        data = np.asarray(data)
        if self.truncation_min is not None:
            data = np.minimum(data, self.truncation_min)
        if self.truncation_max is not None:
            data = np.maximum(data, self.truncation_max)
        return data


TRANSFORM_FUNCTIONS = {
    "LN": np.log,
    "LOG": np.log,
    "LN0": lambda v: np.log(v + 0.000001),
    "LOG10": np.log10,
    "EXP": np.exp,
    "EXP0": lambda v: np.exp(v) - 0.000001,
    "POW10": lambda v: np.power(10.0, v),
    "TRUNC_POW10": lambda v: np.maximum(np.power(10, v), 0.001),
}


def field_transform(
    data: npt.ArrayLike, transform_name: Optional[str]
) -> npt.ArrayLike:
    data = np.asarray(data)
    if transform_name is None:
        return data
    return TRANSFORM_FUNCTIONS[transform_name](data)
