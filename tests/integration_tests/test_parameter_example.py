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
from pytest import MonkeyPatch
from resdata import ResDataType
from resdata.grid import GridGenerator
from resdata.resfile import ResdataKW

from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.field_utils import FieldFileFormat, read_field, save_field
from tests.unit_tests.config.egrid_generator import egrids
from tests.unit_tests.config.summary_generator import summaries

from .run_cli import run_cli

config_contents = """
        NUM_REALIZATIONS {num_realizations}
        QUEUE_SYSTEM LOCAL
        MAX_RUNNING {num_realizations}
        FIELD PARAM_A  PARAMETER PARAM_A.grdecl  INIT_FILES:PARAM_A%d.grdecl
        FIELD AFI      PARAMETER AFI.grdecl      INIT_FILES:AFI.grdecl FORWARD_INIT:True
        FIELD A_ROFF   PARAMETER A_ROFF.roff     INIT_FILES:A_ROFF%d.roff
        FIELD PARAM_M5 PARAMETER PARAM_M5.grdecl INIT_FILES:PARAM_M5%d.grdecl MIN:0.5
        FIELD PARAM_M8 PARAMETER PARAM_M8.grdecl INIT_FILES:PARAM_M8%d.grdecl MAX:0.8
        FIELD TR58     PARAMETER TR58.grdecl     INIT_FILES:TR58%d.grdecl MIN:0.5 MAX:0.8
        FIELD TRANS1    PARAMETER TRANS1.roff      INIT_FILES:TRANS1%d.grdecl OUTPUT_TRANSFORM:LN
        FIELD TRANS2    PARAMETER TRANS2.roff      INIT_FILES:TRANS2%d.grdecl INIT_TRANSFORM:LN

        SURFACE TOP     OUTPUT_FILE:TOP.irap   INIT_FILES:TOP%d.irap BASE_SURFACE:BASETOP.irap
        SURFACE BOTTOM  OUTPUT_FILE:BOT.irap   INIT_FILES:BOT.irap BASE_SURFACE:BASEBOT.irap FORWARD_INIT:True


        ECLBASE ECLBASE
        SUMMARY FOPR
        GRID {grid_name}.{grid_format}


        FORWARD_MODEL COPY_FILE(<FROM>="../../../AFI.grdecl",<TO>="AFI.grdecl")
        FORWARD_MODEL COPY_FILE(<FROM>="../../../BOT.irap",<TO>="BOT.irap")
        FORWARD_MODEL COPY_FILE(<FROM>="../../../ECLBASE.UNSMRY",<TO>="ECLBASE.UNSMRY")
        FORWARD_MODEL COPY_FILE(<FROM>="../../../ECLBASE.SMSPEC",<TO>="ECLBASE.SMSPEC")
"""

template = """
MY_KEYWORD <MY_KEYWORD>
SECOND_KEYWORD <SECOND_KEYWORD>
"""

prior = """
MY_KEYWORD NORMAL 0 1
SECOND_KEYWORD NORMAL 0 1
"""


class IoLibrary(Enum):
    ERT = auto()
    XTGEO = auto()
    RESDATA = auto()


class IoProvider:
    def __init__(self, data: st.DataObject):
        self.data = data
        self.field_values = {}
        self.surface_values = {}

        coordinates = st.integers(min_value=1, max_value=4)
        self.dims = data.draw(st.tuples(coordinates, coordinates, coordinates))
        self.size = self.dims[0] * self.dims[1] * self.dims[2]
        self.actnum = data.draw(
            st.lists(
                elements=st.integers(min_value=0, max_value=3),
                min_size=self.size,
                max_size=self.size,
            )
        )

    def _choose_lib(self):
        return self.data.draw(st.sampled_from(IoLibrary))

    def create_grid(self, grid_name: str, grid_format: Literal["grid", "egrid"]):
        lib = self._choose_lib()
        grid_file = grid_name + "." + grid_format
        if grid_format == "grid":
            grid = GridGenerator.create_rectangular(
                self.dims, (1, 1, 1), actnum=self.actnum
            )
            grid.save_GRID(grid_file)
        elif lib == IoLibrary.RESDATA:
            grid = GridGenerator.create_rectangular(
                self.dims, (1, 1, 1), actnum=self.actnum
            )
            grid.save_EGRID(grid_file)
        elif lib == IoLibrary.XTGEO:
            grid = xtgeo.create_box_grid(dimension=self.dims)
            grid.to_file(grid_file, str(grid_format))
        elif lib == IoLibrary.RESDATA:
            grid = GridGenerator.create_rectangular(
                self.dims, (1, 1, 1), actnum=self.actnum
            )
            if grid_format == "egrid":
                grid.save_EGRID(grid_file)
            elif grid_format == "grid":
                grid.save_GRID(grid_file)
            else:
                raise ValueError()
        elif lib == IoLibrary.ERT:
            egrid = self.data.draw(egrids)
            self.dims = egrid.shape
            self.size = self.dims[0] * self.dims[1] * self.dims[2]
            self.actnum = egrid.global_grid.actnum
            egrid.to_file(grid_file)
        else:
            raise ValueError()

    def _random_values(self, shape):
        return self.data.draw(
            arrays(
                elements=st.floats(min_value=1.0, max_value=10.0, width=32),
                dtype=np.float32,
                shape=shape,
            )
        )

    def create_field(
        self,
        name: str,
        file_name: str,
        fformat: FieldFileFormat,
    ) -> None:
        lib = self._choose_lib()
        values = self._random_values(self.dims)
        self.field_values[file_name] = values

        if lib == IoLibrary.XTGEO:
            prop = xtgeo.GridProperty(
                ncol=self.dims[0],
                nrow=self.dims[1],
                nlay=self.dims[2],
                name=name,
                values=values,
            )
            prop.to_file(
                file_name, fformat="roff" if fformat == "roff_binary" else fformat
            )
        elif lib == IoLibrary.RESDATA:
            if fformat == FieldFileFormat.GRDECL:
                kw = ResdataKW(name, self.size, ResDataType.RD_FLOAT)
                data = values.ravel()
                for i in range(self.size):
                    kw[i] = data[i]
                with cwrap.open(file_name, mode="w") as f:
                    kw.write_grdecl(f)
            else:
                # resdata cannot write roff
                save_field(np.ma.masked_array(values), name, file_name, fformat)

        elif lib == IoLibrary.ERT:
            save_field(np.ma.masked_array(values), name, file_name, fformat)
        else:
            raise ValueError()

    def create_surface(self, file_name: str) -> None:
        values = self._random_values((2, 5))
        self.surface_values[file_name] = values
        xtgeo.RegularSurface(
            ncol=values.shape[0],
            nrow=values.shape[1],
            xinc=1.0,
            yinc=1.0,
            values=values,
        ).to_file(file_name, fformat="irap_ascii")


@settings(max_examples=10)
@given(
    io_source=st.builds(IoProvider, data=st.data()),
    grid_format=st.sampled_from(["grid", "egrid"]),
    summary=summaries(
        start_date=st.just(datetime(1996, 1, 1)),
        time_deltas=st.just([1.0, 2.0]),
        summary_keys=st.just(["FOPR"]),
    ),
)
def test_parameter_example(io_source, grid_format, summary, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("parameter_example")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        NUM_REALIZATIONS = 10
        GRID_NAME = "GRID"
        Path("config.ert").write_text(
            config_contents.format(
                grid_name=GRID_NAME,
                grid_format=grid_format,
                num_realizations=NUM_REALIZATIONS,
            )
        )
        Path("prior.txt").write_text(prior)
        Path("template.txt").write_text(template)
        io_source.create_grid(GRID_NAME, grid_format)

        # create non-forward-init files that will have to exist before
        # first iteration for all iterations
        grdecl_fields = ["PARAM_A", "PARAM_M5", "PARAM_M8", "TR58"]
        for i in range(NUM_REALIZATIONS):
            for field in grdecl_fields:
                io_source.create_field(
                    field, f"{field}{i}.grdecl", FieldFileFormat.GRDECL
                )
            io_source.create_surface(f"TOP{i}.irap")
            io_source.create_field(
                "A_ROFF", f"A_ROFF{i}.roff", FieldFileFormat.ROFF_BINARY
            )
            io_source.create_field(
                "TRANS1", f"TRANS1{i}.grdecl", FieldFileFormat.GRDECL
            )
            io_source.create_field(
                "TRANS2", f"TRANS2{i}.grdecl", FieldFileFormat.GRDECL
            )

        # Creates forward init files that will be copied in by the
        # COPY_FILE forward model. This fails if ert has already created
        # the file.
        io_source.create_field("AFI", "AFI.grdecl", FieldFileFormat.GRDECL)
        io_source.create_surface("BOT.irap")

        # A COPY_FILE forward model also copies in the
        # summary files expected to be created for the
        # SUMMARY keyword
        smspec, unsmry = summary
        smspec.to_file("ECLBASE.SMSPEC")
        unsmry.to_file("ECLBASE.UNSMRY")

        io_source.create_surface("BASETOP.irap")
        io_source.create_surface("BASEBOT.irap")

        run_cli(ENSEMBLE_EXPERIMENT_MODE, "config.ert")

        for i in range(NUM_REALIZATIONS):
            path = Path(f"simulations/realization-{i}/iter-0")
            np.testing.assert_allclose(
                read_field(
                    path / "PARAM_A.grdecl",
                    "PARAM_A",
                    shape=io_source.dims,
                    mask=io_source.actnum != 0,
                ),
                io_source.field_values[f"PARAM_A{i}.grdecl"],
                atol=5e-5,
            )
            np.testing.assert_allclose(
                read_field(
                    path / "PARAM_M5.grdecl",
                    "PARAM_M5",
                    shape=io_source.dims,
                    mask=io_source.actnum != 0,
                ),
                np.clip(io_source.field_values[f"PARAM_M5{i}.grdecl"], 5.0, None),
                atol=5e-5,
            )
            np.testing.assert_allclose(
                read_field(
                    path / "TR58.grdecl",
                    "TR58",
                    shape=io_source.dims,
                    mask=io_source.actnum != 0,
                ),
                np.clip(io_source.field_values[f"PARAM_M8{i}.grdecl"], 5.0, 8.0),
                atol=5e-5,
            )
            np.testing.assert_allclose(
                read_field(
                    path / "TRANS1.roff",
                    "TRANS1",
                    shape=io_source.dims,
                    mask=io_source.actnum != 0,
                ),
                np.log(io_source.field_values[f"TRANS1{i}.grdecl"]),
                atol=5e-5,
            )
            np.testing.assert_allclose(
                read_field(
                    path / "TRANS2.roff",
                    "TRANS2",
                    shape=io_source.dims,
                    mask=io_source.actnum != 0,
                ),
                np.log(io_source.field_values[f"TRANS2{i}.grdecl"]),
                atol=5e-5,
            )

            np.testing.assert_allclose(
                xtgeo.surface_from_file(
                    path / "TOP.irap",
                    "irap_ascii",
                ).values,
                (io_source.surface_values[f"TOP{i}.irap"]),
                atol=5e-5,
            )
