import os
from typing import Optional

from ecl.ecl_util import EclFileEnum, get_file_type
from ecl.grid import EclGrid
from ecl.summary import EclSum

from ert._c_wrappers.enkf.config_keys import ConfigKeys


class EclConfig:
    def __init__(
        self,
        grid_file: Optional[str] = None,
        refcase_file: Optional[str] = None,
    ):
        self._grid_file = grid_file
        self._refcase_file = refcase_file
        self.grid = self._load_grid() if grid_file else None
        self.refcase = self._load_refcase() if refcase_file else None

    @classmethod
    def from_dict(cls, config_dict) -> "EclConfig":
        ecl_config_args = {}
        if ConfigKeys.GRID in config_dict:
            ecl_config_args["grid_file"] = os.path.realpath(
                config_dict[ConfigKeys.GRID]
            )
        if ConfigKeys.REFCASE in config_dict:
            ecl_config_args["refcase_file"] = os.path.realpath(
                config_dict[ConfigKeys.REFCASE]
            )
        return cls(**ecl_config_args)

    def _load_grid(self) -> Optional[EclGrid]:
        ecl_grid_file_types = [
            EclFileEnum.ECL_GRID_FILE,
            EclFileEnum.ECL_EGRID_FILE,
        ]
        if get_file_type(self._grid_file) not in ecl_grid_file_types:
            raise ValueError(f"grid file {self._grid_file} does not have expected type")
        return EclGrid.load_from_file(self._grid_file)

    def _load_refcase(self) -> EclSum:
        # defaults for loading refcase - necessary for using the function
        # exposed in python part of ecl
        refcase_load_args = {
            "load_case": self._refcase_file,
            "join_string": ":",
            "include_restart": True,
            "lazy_load": True,
            "file_options": 0,
        }
        return EclSum(**refcase_load_args)

    def __repr__(self):
        return (
            "EclConfig(\n"
            f"\tgrid_file={self._grid_file},\n"
            f"\trefcase_file={self._refcase_file},\n"
            ")"
        )

    def __eq__(self, other):
        if self._grid_file != other._grid_file:
            return False

        if self._refcase_file != other._refcase_file:
            return False

        return True
