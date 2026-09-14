from pathlib import Path

import pytest

from ert.config import Field, GenKwConfig, ParameterConfig, SurfaceConfig
from ert.field_utils import ErtboxParameters, FieldFileFormat


@pytest.fixture
def non_updatable_parameter_configs() -> list[ParameterConfig]:
    return [
        GenKwConfig(
            name="COEFFS",
            distribution={"name": "normal", "mean": 0, "std": 1},
            update_strategy=None,
        ),
        Field(
            name="PERMX",
            forward_init=False,
            update_strategy=None,
            ertbox_params=ErtboxParameters(nx=1, ny=1, nz=1),
            file_format=FieldFileFormat.ROFF,
            forward_init_file="permx_%d.roff",
            output_file=Path("permx.roff"),
            grid_file="grid.EGRID",
        ),
        SurfaceConfig(
            name="TOP",
            forward_init=False,
            update_strategy=None,
            ncol=1,
            nrow=1,
            xori=0,
            yori=0,
            xinc=1,
            yinc=1,
            rotation=0,
            yflip=1,
            forward_init_file="top_%d.irap",
            output_file=Path("top.irap"),
            base_surface_path="base_surface.irap",
        ),
    ]
