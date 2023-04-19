from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig

if TYPE_CHECKING:
    from ert.storage import EnsembleAccessor, EnsembleReader


@dataclass
class SurfaceConfig(ParameterConfig):
    forward_init_file: str
    output_file: Path
    base_surface_path: str

    def load(self, run_path: Path, real_nr: int, ensemble: EnsembleAccessor):
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

    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        file_path = run_path / self.output_file
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)
        surf = ensemble.load_surface_file(self.name, real_nr)
        surf.to_file(file_path, fformat="irap_ascii")
