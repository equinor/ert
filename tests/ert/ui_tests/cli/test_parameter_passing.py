"""
The following test generates different kind of parameters
and places their configuration in the config file. Then an
ensemble experiment is ran and we assert that each realization
was passed the parameters correctly in the runpath.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import cwrap
import hypothesis.strategies as st
import numpy as np
import xtgeo
from hypothesis import given, note, settings
from hypothesis.extra.numpy import arrays
from pytest import MonkeyPatch, TempPathFactory, mark
from resdata import ResDataType
from resdata.grid import GridGenerator
from resdata.resfile import ResdataKW
from resfo_utilities.testing import egrids, summaries

from ert.field_utils import FieldFileFormat, Shape, read_field, save_field
from ert.field_utils.field_file_format import ROFF_FORMATS
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from tests.ert.grid_generator import xtgeo_box_grids

from .run_cli import run_cli

names = st.text(
    min_size=1,
    max_size=8,
    alphabet=st.characters(
        min_codepoint=ord("!"),
        max_codepoint=ord("~"),
        exclude_characters="\"'$,:%",  # These have specific meaning in configs
    ),
)

config_contents = """
NUM_REALIZATIONS {num_realizations}
QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING {num_realizations}

ECLBASE ECLBASE
SUMMARY FOPR
GRID {grid_name}.{grid_format}

FORWARD_MODEL COPY_FILE(<FROM>="../../../ECLBASE.UNSMRY",<TO>="ECLBASE.UNSMRY")
FORWARD_MODEL COPY_FILE(<FROM>="../../../ECLBASE.SMSPEC",<TO>="ECLBASE.SMSPEC")
"""


class IoLibrary(Enum):
    """Libraries that can be used to write the parameter values"""

    ERT = auto()
    XTGEO = auto()
    RESDATA = auto()


def extension(format_: FieldFileFormat):
    """The file extension of a given field file format"""
    if format_ in ROFF_FORMATS:
        return "roff"
    return format_.value


def xtgeo_fformat(
    format_: FieldFileFormat,
) -> Literal["roff", "roffasc", "grdecl", "bgrdecl"]:
    """Converts the FieldFileFormat to the corresponding xtgeo 'fformat' value.

    See xtgeo.GridProperty.to_file
    """
    if format_ == FieldFileFormat.ROFF_BINARY:
        return "roff"
    elif format_ == FieldFileFormat.ROFF_ASCII:
        return "roffasc"
    return format_.value


class IoProvider:
    """Provides the ability to generate grid, field and surface files."""

    def __init__(self, data: st.DataObject) -> None:
        self.data = data
        self.field_values = {}
        self.surface_values = {}

        coordinates = st.integers(min_value=1, max_value=4)
        self.dims = data.draw(
            st.tuples(coordinates, coordinates, coordinates), label="shape"
        )
        self.size = self.dims[0] * self.dims[1] * self.dims[2]
        self.actnum = data.draw(
            st.lists(
                elements=st.integers(min_value=0, max_value=1),
                min_size=self.size,
                max_size=self.size,
            ),
            label="actnum",
        )

    def _choose_lib(self) -> IoLibrary:
        return self.data.draw(st.sampled_from(IoLibrary), label="writer")

    def write_grid_file(self, grid_name: str, grid_format: Literal["grid", "egrid"]):
        """Writes a grid file with the name '{grid_name}.{grid_format}'"""
        grid_file = grid_name + "." + grid_format
        if grid_format == "grid":
            grid = GridGenerator.create_rectangular(
                self.dims, (1, 1, 1), actnum=self.actnum
            )
            grid.save_GRID(grid_file)
            return
        lib = self._choose_lib()
        if lib == IoLibrary.RESDATA:
            grid = GridGenerator.create_rectangular(
                self.dims, (1, 1, 1), actnum=self.actnum
            )
            grid.save_EGRID(grid_file)
        elif lib == IoLibrary.XTGEO:
            grid = self.data.draw(xtgeo_box_grids())
            self.dims = tuple(grid.dimensions)
            self.size = self.dims[0] * self.dims[1] * self.dims[2]
            self.actnum = (
                grid.actnum_array
                if grid.actnum_array is not None
                else np.ones(self.size)
            )
            grid.to_file(grid_file, str(grid_format))
        elif lib == IoLibrary.ERT:
            egrid = self.data.draw(egrids)
            self.dims = (
                egrid.global_grid.grid_head.num_x,
                egrid.global_grid.grid_head.num_y,
                egrid.global_grid.grid_head.num_z,
            )
            self.size = self.dims[0] * self.dims[1] * self.dims[2]
            self.actnum = (
                egrid.global_grid.actnum
                if egrid.global_grid.actnum is not None
                else np.ones(self.size)
            )
            egrid.to_file(grid_file)
        else:
            raise ValueError

    def _random_values(self, shape: tuple[int, ...], name: str):
        return self.data.draw(
            arrays(
                elements=st.floats(min_value=2.0, max_value=4.0, width=32),
                dtype=np.float32,
                shape=shape,
            ),
            label=name,
        )

    def write_field_file(
        self,
        field_name: str,
        file_name: str,
        fformat: FieldFileFormat,
    ) -> None:
        lib = self._choose_lib()
        values = self._random_values(self.dims, file_name)
        self.field_values[file_name] = values

        if lib == IoLibrary.XTGEO:
            prop = xtgeo.GridProperty(
                ncol=self.dims[0],
                nrow=self.dims[1],
                nlay=self.dims[2],
                name=field_name,
                values=values,
            )
            prop.to_file(file_name, fformat=xtgeo_fformat(fformat))
        elif lib == IoLibrary.RESDATA:
            if fformat == FieldFileFormat.GRDECL:
                kw = ResdataKW(field_name, self.size, ResDataType.RD_FLOAT)
                data = values.ravel(order="F")
                for i in range(self.size):
                    kw[i] = data[i]
                with cwrap.open(file_name, mode="w") as f:
                    kw.write_grdecl(f)
            else:
                # resdata cannot write roff
                save_field(np.ma.masked_array(values), field_name, file_name, fformat)

        elif lib == IoLibrary.ERT:
            save_field(np.ma.masked_array(values), field_name, file_name, fformat)
        else:
            raise ValueError

    def write_surface_file(self, file_name: str) -> None:
        values = self._random_values((2, 5), file_name)
        self.surface_values[file_name] = values
        xtgeo.RegularSurface(
            ncol=values.shape[0],
            nrow=values.shape[1],
            xinc=1.0,
            yinc=1.0,
            values=values,
        ).to_file(file_name, fformat="irap_ascii")


class Transform(Enum):
    LN = auto()
    EXP = auto()

    def __call__(
        self, values: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        if self == Transform.LN:
            return np.log(values)
        if self == Transform.EXP:
            return np.exp(values)
        raise ValueError


@dataclass
class Parameter:
    @abstractmethod
    def configuration(self) -> str:
        """The contents of the config file for the parameter"""

    @abstractmethod
    def create_file(self, io_source: IoProvider, num_realizations: int):
        """Writes the parameter files with the given IoProvider"""

    @abstractmethod
    def check(self, io_source: IoProvider, mask, num_realizations: int):
        """Check that the files in the runpath have the correct values"""


@dataclass
class FieldParameter(Parameter):
    name: str
    infformat: FieldFileFormat
    outfformat: FieldFileFormat
    min: float | None
    max: float | None
    input_transform: Transform | None
    output_transform: Transform | None
    forward_init: bool

    @property
    def inext(self):
        """The file extension of the input file"""
        return extension(self.infformat)

    @property
    def outext(self):
        """The file extension of the output file"""
        return extension(self.outfformat)

    @property
    def out_filename(self):
        return self.name.replace("/", "slash") + "." + self.outext

    @property
    def in_filename(self):
        return self.name.replace("/", "slash") + "." + self.inext

    def configuration(self):
        decl = f"FIELD {self.name} PARAMETER {self.out_filename} "
        if self.forward_init:
            decl += f" FORWARD_INIT:True INIT_FILES:{self.out_filename} "
        else:
            decl += f" INIT_FILES:%d{self.in_filename} "
        if self.min is not None:
            decl += f" MIN:{self.min} "
        if self.max is not None:
            decl += f" MAX:{self.max} "
        if self.input_transform is not None:
            decl += f" INIT_TRANSFORM:{self.input_transform.name} "
        if self.output_transform is not None:
            decl += f" OUTPUT_TRANSFORM:{self.output_transform.name} "

        # If forward_init, a forward model step is expected to produce the
        # init file. The following COPY_FILE is that forward model step.
        if self.forward_init:
            decl += (
                "\nFORWARD_MODEL COPY_FILE("
                f'<FROM>="../../../{self.out_filename}",<TO>=.)'
            )
        return decl

    def create_file(self, io_source: IoProvider, num_realizations: int):
        if self.forward_init:
            io_source.write_field_file(
                self.name,
                f"{self.out_filename}",
                self.outfformat,
            )
        else:
            for i in range(num_realizations):
                io_source.write_field_file(
                    self.name, str(i) + self.in_filename, self.infformat
                )

    def check(self, io_source: IoProvider, mask, num_realizations: int):
        for i in range(num_realizations):
            if self.forward_init:
                values = io_source.field_values[self.out_filename]
            else:
                values = io_source.field_values[str(i) + self.in_filename]
                if self.input_transform:
                    values = self.input_transform(values)
                if self.output_transform:
                    values = self.output_transform(values)
                if self.min is not None or self.max is not None:
                    values = np.clip(values, self.min, self.max)
            path = Path(f"simulations/realization-{i}/iter-0")
            read_values = read_field(
                path / self.out_filename,
                self.name,
                shape=Shape(*io_source.dims),
            )
            np.testing.assert_allclose(
                read_values,
                values,
                atol=5e-5,
                rtol=1e-4,
            )


@st.composite
def field_parameters(draw):
    min_value = draw(
        st.one_of(
            st.none(),
            st.floats(
                min_value=-1e6, max_value=4.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    max_value = st.one_of(
        st.none(),
        st.floats(
            min_value=max([2.0, min_value if min_value is not None else 0.0]),
            max_value=1e9,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    return draw(
        st.builds(FieldParameter, name=names, min=st.just(min_value), max=max_value)
    )


@dataclass
class SurfaceParameter(Parameter):
    name: str
    forward_init: bool

    @property
    def filename(self):
        return self.name.replace("/", "slash") + ".irap"

    def configuration(self):
        if self.forward_init:
            return (
                f"SURFACE {self.name} OUTPUT_FILE:{self.filename} "
                f"INIT_FILES:{self.filename} BASE_SURFACE:BASE{self.filename} "
                "FORWARD_INIT:True\n"
                # If forward_init, a forward model step is expected to produce the
                # init file. The following COPY_FILE is that forward model step.
                f'FORWARD_MODEL COPY_FILE(<FROM>="../../../{self.filename}",<TO>=.)'
            )

        else:
            return (
                f"SURFACE {self.name} OUTPUT_FILE:{self.filename}"
                f" INIT_FILES:%d{self.filename} BASE_SURFACE:BASE{self.filename}"
            )

    def create_file(self, io_source: IoProvider, num_realizations: int):
        io_source.write_surface_file("BASE" + self.filename)
        if self.forward_init:
            io_source.write_surface_file(self.filename)
        else:
            for i in range(num_realizations):
                io_source.write_surface_file(str(i) + self.filename)

    def check(self, io_source: IoProvider, mask, num_realizations: int):
        for i in range(num_realizations):
            values = io_source.surface_values[
                self.filename if self.forward_init else str(i) + self.filename
            ]
            path = Path(f"simulations/realization-{i}/iter-0")
            np.testing.assert_allclose(
                xtgeo.surface_from_file(
                    path / self.filename,
                    "irap_ascii",
                ).values,
                values,
                atol=5e-5,
            )


@settings(max_examples=10)
@given(
    io_source=st.builds(IoProvider, data=st.data()),
    grid_format=st.sampled_from(["grid", "egrid"]),
    summary=summaries(
        start_date=st.just(datetime(1996, 1, 1)),
        time_deltas=st.just([1.0, 2.0]),
        summary_keys=st.just(["FOPR"]),
    ),
    num_realizations=st.integers(min_value=1, max_value=10),
    parameters=st.lists(
        st.one_of(
            field_parameters(),
            st.builds(SurfaceParameter, names),
        ),
        unique_by=lambda x: x.name,
        max_size=3,
    ),
)
@mark.skip_mac_ci  # test is slow
def test_that_parameters_are_placed_in_the_runpath_as_expected(
    io_source: IoProvider,
    grid_format: Literal["grid", "egrid"],
    summary,
    tmp_path_factory: TempPathFactory,
    num_realizations: int,
    parameters: list[Parameter],
):
    tmp_path = tmp_path_factory.mktemp("parameter_example")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        GRID_NAME = "GRID"
        contents = config_contents.format(
            grid_name=GRID_NAME,
            grid_format=grid_format,
            num_realizations=num_realizations,
        ) + "\n".join(p.configuration() for p in parameters)
        note(f"config file: {contents}")
        Path("config.ert").write_text(contents, encoding="utf-8")
        io_source.write_grid_file(GRID_NAME, grid_format)

        for p in parameters:
            p.create_file(io_source, num_realizations)

        # A COPY_FILE forward model step copies in the summary files expected to be
        # created for the SUMMARY keyword
        smspec, unsmry = summary
        smspec.to_file("ECLBASE.SMSPEC")
        unsmry.to_file("ECLBASE.UNSMRY")

        run_cli(ENSEMBLE_EXPERIMENT_MODE, "--disable-monitoring", "config.ert")

        mask = np.logical_not(
            np.array(io_source.actnum).reshape(io_source.dims, order="F")
        )
        for p in parameters:
            p.check(io_source, mask, num_realizations)
