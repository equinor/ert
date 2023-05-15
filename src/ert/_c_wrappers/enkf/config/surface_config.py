from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig

if TYPE_CHECKING:
    from ert.storage import EnsembleAccessor, EnsembleReader

_logger = logging.getLogger(__name__)


@dataclass
class SurfaceConfig(ParameterConfig):
    ncol: int
    nrow: int
    xori: int
    yori: int
    xinc: int
    yinc: int
    rotation: int
    yflip: int
    forward_init_file: str
    output_file: Path

    def load(self, run_path: Path, real_nr: int, ensemble: EnsembleAccessor):
        t = time.perf_counter()
        file_name = self.forward_init_file
        if "%d" in file_name:
            file_name = file_name % real_nr
        file_path = run_path / file_name
        if file_path.exists():
            ensemble.save_surface_file(self.name, real_nr, str(file_path))
        else:
            raise ValueError(
                "Failed to initialize parameter "
                f"'{self.name}' in file {file_name}: "
                "File not found\n"
            )
        _logger.debug(f"load() time_used {(time.perf_counter() - t):.4f}s")

    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        t = time.perf_counter()
        file_path = run_path / self.output_file
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)
        surf = ensemble.load_surface_file(self.name, real_nr)
        surf.to_file(file_path, fformat="irap_ascii")
        _logger.debug(f"save() time_used {(time.perf_counter() - t):.4f}s")
