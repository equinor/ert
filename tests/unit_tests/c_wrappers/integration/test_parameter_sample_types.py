# pylint: disable=too-many-lines
import logging
import math
import os
import stat
from argparse import ArgumentParser
from contextlib import ExitStack as does_not_raise
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import numpy as np
import numpy.testing
import pytest
import xtgeo
from ecl.util.geometry import Surface

from ert.__main__ import ert_parser
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.libres_facade import LibresFacade


def write_file(fname, contents):
    with open(fname, mode="w", encoding="utf-8") as fout:
        fout.writelines(contents)


def create_runpath(config, active_mask=None):
    active_mask = [True] if active_mask is None else active_mask
    res_config = ErtConfig.from_file(config)
    ert = EnKFMain(res_config)

    prior = ert.load_ensemble_context("default", active_mask, 0)

    ert.sample_prior(prior.sim_fs, prior.active_realizations)
    ert.createRunPath(prior)
    return ert


def load_from_forward_model(ert):
    facade = LibresFacade(ert)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model("default", realizations, 0)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "config_str, expected, extra_files, expectation",
    [
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt\nRANDOM_SEED 1234",
            "MY_KEYWORD -0.881423",
            [],
            does_not_raise(),
        ),
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt INIT_FILES:custom_param%d",
            "MY_KEYWORD 1.31",
            [("custom_param0", "MY_KEYWORD 1.31")],
            does_not_raise(),
        ),
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt INIT_FILES:custom_param%d",
            "MY_KEYWORD 1.31",
            [("custom_param0", "1.31")],
            does_not_raise(),
        ),
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt INIT_FILES:custom_param0",  # noqa
            "Not expecting a file",
            [],
            pytest.raises(
                ConfigValidationError, match="Loading GEN_KW from files requires %d"
            ),
        ),
    ],
)
def test_gen_kw(tmpdir, config_str, expected, extra_files, expectation):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        """
        )
        config += config_str
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        for fname, contents in extra_files:
            write_file(fname, contents)

        with expectation:
            create_runpath("config.ert")
            assert (
                Path("simulations/realization-0/iter-0/kw.txt").read_text(
                    encoding="utf-8"
                )
                == expected
            )


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "relpath",
    [
        "somepath/",
        # This test was added to show current behaviour for Ert.
        # If absolute paths should be possible to be used like this is up for debate.
        "/tmp/somepath/",  # ert removes leading '/'
    ],
)
def test_gen_kw_outfile_will_use_paths(tmpdir, relpath: str):
    with tmpdir.as_cwd():
        config = dedent(
            f"""
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt {relpath}kw.txt prior.txt
        """
        )

        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        create_runpath("config.ert")
        assert os.path.exists(f"simulations/realization-0/iter-0/{relpath}kw.txt")


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "config_str, expected, extra_files",
    [
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt INIT_FILES:custom_param%d",
            "MY_KEYWORD 1.31\nMY_SECOND_KEYWORD 1.01",
            [("custom_param0", "MY_SECOND_KEYWORD 1.01\nMY_KEYWORD 1.31")],
        ),
    ],
)
def test_that_order_of_input_in_user_input_is_abritrary_for_gen_kw_init_files(
    tmpdir, config_str, expected, extra_files
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        """
        )
        config += config_str
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines(
                "MY_KEYWORD <MY_KEYWORD>\nMY_SECOND_KEYWORD <MY_SECOND_KEYWORD>"
            )
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1\nMY_SECOND_KEYWORD NORMAL 0 1")
        for fname, contents in extra_files:
            write_file(fname, contents)

        create_runpath("config.ert")
        assert (
            Path("simulations/realization-0/iter-0/kw.txt").read_text("utf-8")
            == expected
        )


def test_field_load_two_parameters_forward_init(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:../../../param_a.grdecl FORWARD_INIT:True
        FIELD PARAM_B PARAMETER param_b.grdecl INIT_FILES:../../../param_b.grdecl FORWARD_INIT:True
        GRID MY_EGRID.EGRID
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a = np.ndarray(shape=(10, 10, 1))
        param_b = np.ndarray(shape=(10, 10, 1))
        param_a.fill(22)
        param_b.fill(77)

        gp_a = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_a,
            grid=grid,
            name="PARAM_A",
        )
        gp_a.to_file("param_a.grdecl", fformat="grdecl")

        gp_b = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_b,
            grid=grid,
            name="PARAM_B",
        )
        gp_b.to_file("param_b.grdecl", fformat="grdecl")

        ert = create_runpath("config.ert")
        fs = ert.storage_manager["default"]

        assert ert.ensembleConfig()["PARAM_A"].getUseForwardInit()
        assert ert.ensembleConfig()["PARAM_B"].getUseForwardInit()
        assert not Path("simulations/realization-0/iter-0/param_a.grdecl").exists()
        assert not Path("simulations/realization-0/iter-0/param_b.grdecl").exists()

        # should not be loaded yet
        with pytest.raises(KeyError, match="Unable to load FIELD for key: PARAM_A"):
            fs.load_field("PARAM_A", [0])

        with pytest.raises(KeyError, match="Unable to load FIELD for key: PARAM_B"):
            fs.load_field("PARAM_B", [0])

        assert load_from_forward_model(ert) == 1

        create_runpath("config.ert")

        prop_a = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_a.values.data, param_a)

        prop_b = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_b.grdecl",
            name="PARAM_B",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_b.values.data, param_b)

        # should be loaded now
        loaded_a = fs.load_field("PARAM_A", [0])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == 22

        loaded_b = fs.load_field("PARAM_B", [0])
        for e in range(0, loaded_b.shape[0]):
            assert loaded_b[e][0] == 77


def test_field_load_two_parameters(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:param_a_%d.grdecl
        FIELD PARAM_B PARAMETER param_b.grdecl INIT_FILES:param_b_%d.grdecl
        GRID MY_EGRID.EGRID
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a = np.ndarray(shape=(10, 10, 1))
        param_b = np.ndarray(shape=(10, 10, 1))
        param_a.fill(22)
        param_b.fill(77)

        gp_a = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_a,
            grid=grid,
            name="PARAM_A",
        )
        gp_a.to_file("param_a_0.grdecl", fformat="grdecl")

        gp_b = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_b,
            grid=grid,
            name="PARAM_B",
        )
        gp_b.to_file("param_b_0.grdecl", fformat="grdecl")

        ert = create_runpath("config.ert")
        assert not ert.ensembleConfig()["PARAM_A"].getUseForwardInit()
        assert not ert.ensembleConfig()["PARAM_B"].getUseForwardInit()

        fs = ert.storage_manager["default"]
        loaded_a = fs.load_field("PARAM_A", [0])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == 22

        loaded_b = fs.load_field("PARAM_B", [0])
        for e in range(0, loaded_b.shape[0]):
            assert loaded_b[e][0] == 77

        prop_a = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_a.values.data, param_a)

        prop_b = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_b.grdecl",
            name="PARAM_B",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_b.values.data, param_b)


@pytest.mark.parametrize(
    "min_, max_, field_config",
    [
        (
            0.5,
            None,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MIN:0.5",  # pylint: disable=line-too-long  # noqa: E501
        ),
        (
            None,
            0.8,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MAX:0.8",  # pylint: disable=line-too-long  # noqa: E501
        ),
        (
            0.5,
            0.8,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MIN:0.5 MAX:0.8",  # pylint: disable=line-too-long  # noqa: E501
        ),
    ],
)
def test_field_with_min_max(tmpdir, min_: int, max_: int, field_config: str):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GRID MY_EGRID.EGRID
        """
        )
        config += field_config
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        my_param = np.ndarray(shape=(10, 10, 1), buffer=np.random.random_sample(100))
        my_param[8][7] = 0.001
        my_param[8][3] = 1.001

        gp_a = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=my_param,
            grid=grid,
            name="MY_PARAM",
        )
        gp_a.to_file("my_param_0.grdecl", fformat="grdecl")

        create_runpath("config.ert", [True])

        my_prop = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/my_param.grdecl",
            name="MY_PARAM",
            grid=grid,
        )
        if min_ and max_:
            vfunc = np.vectorize(
                lambda x: ((x + 0.0001) >= min_) and ((x - 0.0001) <= max_)
            )
            assert vfunc(my_prop.values.data).all()
        elif min_:
            vfunc = np.vectorize(lambda x: (x + 0.0001) >= min_)
            assert vfunc(my_prop.values.data).all()
        elif max_:
            vfunc = np.vectorize(lambda x: (x - 0.0001) <= max_)
            assert vfunc(my_prop.values.data).all()


def test_field_with_transformation(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 2
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:param_a_%d.grdecl INIT_TRANSFORM:LN OUTPUT_TRANSFORM:EXP
        GRID MY_EGRID.EGRID
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a_1 = np.ndarray(shape=(10, 10, 1), dtype=float)
        param_a_2 = np.ndarray(shape=(10, 10, 1), dtype=float)
        param_a_1.fill(math.exp(2.5))
        param_a_2.fill(math.exp(1.5))

        gp_a_1 = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_a_1,
            grid=grid,
            name="PARAM_A",
        )
        gp_a_1.to_file("param_a_0.grdecl", fformat="grdecl")

        gp_a_2 = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=param_a_2,
            grid=grid,
            name="PARAM_A",
        )
        gp_a_2.to_file("param_a_1.grdecl", fformat="grdecl")

        ert = create_runpath("config.ert", [True, True])
        fs = ert.storage_manager["default"]

        # stored internally as 2.5, 1.5
        loaded_a = fs.load_field("PARAM_A", [0, 1])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == pytest.approx(2.5)
            assert loaded_a[e][1] == pytest.approx(1.5)

        prop_a_1 = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_almost_equal(prop_a_1.values.data, param_a_1, decimal=5)

        prop_a_2 = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-1/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_almost_equal(prop_a_2.values.data, param_a_2, decimal=5)


@pytest.mark.parametrize(
    "config_str, expect_forward_init",
    [
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl "
            "INIT_FILES:../../../my_param_0.grdecl FORWARD_INIT:True",
            True,
        ),
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl",
            False,
        ),
    ],
)
def test_field_forward_init(tmpdir, config_str, expect_forward_init):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GRID MY_EGRID.EGRID
        """
        )
        config += config_str
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(4, 4, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        expect_param = np.ndarray(
            shape=(4, 4, 1), buffer=np.arange(start=0, stop=4 * 4)
        )
        gp = xtgeo.GridProperty(
            ncol=4,
            nrow=4,
            nlay=1,
            values=expect_param,
            grid=grid,
            name="MY_PARAM",
        )
        gp.to_file("my_param_0.grdecl", fformat="grdecl")

        ert = create_runpath("config.ert")
        assert (
            ert.ensembleConfig()["MY_PARAM"].getUseForwardInit() is expect_forward_init
        )

        fs = ert.storage_manager["default"]
        # Assert that the data has been written to runpath
        if expect_forward_init:
            # FORWARD_INIT: True means that ERT waits until the end of the
            # forward model to internalise the data
            assert not Path("simulations/realization-0/iter-0/my_param.grdecl").exists()

            with pytest.raises(
                KeyError, match="Unable to load FIELD for key: MY_PARAM"
            ):
                fs.load_field("MY_PARAM", [0])

            # We try to load the parameters from the forward model, this would fail if
            # forward init was not set correctly
            assert load_from_forward_model(ert) == 1

            # Once data has been internalised, ERT will generate the
            # parameter files
            create_runpath("config.ert")

        prop = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/my_param.grdecl",
            name="MY_PARAM",
            grid=grid,
        )
        numpy.testing.assert_equal(prop.values.data, expect_param)

        if expect_forward_init:
            arr = fs.load_field("MY_PARAM", [0])
            assert len(arr) == 16


@pytest.mark.parametrize(
    "config_str, expect_forward_init, expect_num_loaded, error",
    [
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap   INIT_FILES:surf%d.irap   "
            "BASE_SURFACE:surf0.irap",
            False,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:../../../surf%d.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:../../../surf.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf%d.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            0,
            "Failed to initialize parameter 'MY_PARAM' in file surf0.irap",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            0,
            "Failed to initialize parameter 'MY_PARAM' in file surf.irap",
        ),
    ],
)
def test_surface_param(
    tmpdir,
    config_str,
    expect_forward_init,
    expect_num_loaded,
    error,
    caplog,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        """
        )
        config += config_str
        expect_surface = Surface(
            nx=2, ny=2, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        for i in range(4):
            expect_surface[i] = float(i)
        expect_surface.write("surf.irap")
        expect_surface.write("surf0.irap")

        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        ert = create_runpath("config.ert")
        assert (
            ert.ensembleConfig()["MY_PARAM"].getUseForwardInit() is expect_forward_init
        )
        # We try to load the parameters from the forward model, this would fail if
        # forward init was not set correctly
        assert load_from_forward_model(ert) == expect_num_loaded
        assert error in "".join(caplog.messages)

        # Assert that the data has been written to runpath
        if expect_num_loaded > 0:
            if expect_forward_init:
                # FORWARD_INIT: True means that ERT waits until the end of the
                # forward model to internalise the data
                assert not Path("simulations/realization-0/iter-0/surf.irap").exists()

                # Once data has been internalised, ERT will generate the
                # parameter files
                create_runpath("config.ert")

            actual_surface = Surface("simulations/realization-0/iter-0/surf.irap")
            assert actual_surface == expect_surface

        # Assert that the data has been internalised to storage
        fs = ert.storage_manager["default"]
        if expect_num_loaded > 0:
            arr = fs.load_surface_data("MY_PARAM", [0])
            assert arr.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
        else:
            with pytest.raises(KeyError, match="No parameter: MY_PARAM in storage"):
                fs.load_surface_data("MY_PARAM", [0])


@pytest.mark.parametrize(
    "config_str",
    [
        "SURFACE MY_PARAM OUTPUT_FILE:surf.irap   INIT_FILES:surf.irap   BASE_SURFACE:surf0.irap",  # noqa
    ],
)
def test_copy_case(
    tmpdir,
    config_str,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        """
        )
        config += config_str
        expect_surface = Surface(
            nx=2, ny=2, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        for i in range(4):
            expect_surface[i] = float(i)
        expect_surface.write("surf.irap")
        expect_surface.write("surf0.irap")

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        ert = create_runpath("config.ert", active_mask=[True for _ in range(10)])

        # Assert that the data has been internalised to storage
        fs = ert.storage_manager["default"]
        new_fs = ert.storage_manager.add_case("copy")
        fs.copy_from_case(
            new_fs,
            ["MY_PARAM"],
            [True if x in range(5) else False for x in range(10)],
        )
        arr_old = fs.load_surface_data("MY_PARAM", [0, 2, 3])

        arr_new = new_fs.load_surface_data("MY_PARAM", [0, 2, 3])
        assert np.array_equal(arr_old, arr_new)


@pytest.mark.integration_test
@pytest.mark.parametrize("load_forward_init", [True, False])
def test_gen_kw_forward_init(tmpdir, load_forward_init):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt """
            f"""FORWARD_INIT:{str(load_forward_init)} INIT_FILES:custom_param%d
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        if not load_forward_init:
            write_file("custom_param0", "1.31")

        if load_forward_init:
            with pytest.raises(
                ConfigValidationError,
                match=(
                    "Loading GEN_KW from files created by "
                    "the forward model is not supported."
                ),
            ):
                create_runpath("config.ert")
        else:
            ert = create_runpath("config.ert")
            assert Path("simulations/realization-0/iter-0/kw.txt").exists()
            facade = LibresFacade(ert)
            df = facade.load_all_gen_kw_data("default")
            assert df["KW_NAME:MY_KEYWORD"][0] == 1.31


@pytest.mark.parametrize(
    "check_random_seed, expectation",
    [
        pytest.param(
            True,
            does_not_raise(),
            id=(
                "The second initialization we extract the random seed from the first "
                "case and set that in the second case to make sure the sampling can "
                "be reproduced"
            ),
        ),
    ],
)
def test_initialize_random_seed(tmpdir, caplog, check_random_seed, expectation):
    """
    This test initializes a case twice, the first time without a random
    seed.
    """
    with caplog.at_level(logging.INFO), tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        create_runpath("config.ert")
        # We read the first parameter value as a reference value
        expected = Path("simulations/realization-0/iter-0/kw.txt").read_text("utf-8")

        # Make a clean directory for the second case, which is identical
        # to the first, except that it uses the random seed from the first
        os.makedirs("second")
        os.chdir("second")
        random_seed = next(
            message for message in caplog.messages if message.startswith("RANDOM_SEED")
        ).split()[1]
        if check_random_seed:
            config += f"RANDOM_SEED {random_seed}"
        with open("config_2.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        create_runpath("config_2.ert")
        with expectation:
            assert (
                Path("simulations/realization-0/iter-0/kw.txt").read_text("utf-8")
                == expected
            )


def test_that_first_three_parameters_sampled_snapshot(tmpdir):
    """
    Nothing special about the first three, but there was a regression
    in nr. 2, so added one extra.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        RANDOM_SEED 1234
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        ert = create_runpath("config.ert", [True] * 3)
        fs = ert.storage_manager.current_case
        prior = fs.load_gen_kw("KW_NAME", list(range(3)))
        expected = [-0.8814228, 1.5847818, 1.009956]
        np.testing.assert_almost_equal(prior, np.array([expected]))


@pytest.mark.parametrize(
    "num_realisations",
    [4, 5, 10],
)
@pytest.mark.parametrize(
    "template, prior",
    [
        (
            "MY_KEYWORD <MY_KEYWORD>\nMY_SECOND_KEYWORD <MY_SECOND_KEYWORD>",
            "MY_KEYWORD NORMAL 0 1\nMY_SECOND_KEYWORD NORMAL 0 1",
        ),
        (
            "MY_KEYWORD <MY_KEYWORD>",
            "MY_KEYWORD NORMAL 0 1",
        ),
        (
            "MY_FIRST_KEYWORD <MY_FIRST_KEYWORD>\nMY_KEYWORD <MY_KEYWORD>",
            "MY_FIRST_KEYWORD NORMAL 0 1\nMY_KEYWORD NORMAL 0 1",
        ),
    ],
)
def test_that_sampling_is_fixed_from_name(tmpdir, template, prior, num_realisations):
    """
    Testing that the order and number of parameters is not relevant for the values,
    only that name of the parameter and the global seed determine the values.
    """
    with tmpdir.as_cwd():
        config = dedent(
            f"""
        NUM_REALIZATIONS {num_realisations}
        RANDOM_SEED 1234
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines(template)
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines(prior)

        res_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(res_config)

        run_context = ert.create_ensemble_context(
            "prior",
            [True] * num_realisations,
            iteration=0,
        )
        ert.sample_prior(run_context.sim_fs, run_context.active_realizations)

        key_hash = sha256(b"1234" + b"KW_NAME:MY_KEYWORD")
        seed = np.frombuffer(key_hash.digest(), dtype="uint32")
        expected = np.random.default_rng(seed).standard_normal(num_realisations)
        assert LibresFacade(ert).gather_gen_kw_data(
            "prior", "KW_NAME:MY_KEYWORD"
        ).values.flatten().tolist() == list(expected)


@pytest.mark.parametrize(
    "mask, expected",
    [
        pytest.param(
            [True] * 5,
            [
                -0.8814227775506998,
                1.5847817694032422,
                1.009956004559659,
                -0.3614874716984976,
                0.12143084130052884,
            ],
            id="Sampling all values, checking that we get length of 5",
        ),
        pytest.param(
            [False, True, False, True],
            [
                1.5847817694032422,
                -0.3614874716984976,
            ],
            id=(
                "Sampling a subset of parameters (at index 1 and 4), checking"
                "that those values match the corresponding values from the full"
                "sample at the same index"
            ),
        ),
    ],
)
def test_that_sub_sample_maintains_order(tmpdir, mask, expected):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 5
        RANDOM_SEED 1234
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        res_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(res_config)

        run_context = ert.create_ensemble_context(
            "prior",
            mask,
            iteration=0,
        )
        ert.sample_prior(run_context.sim_fs, run_context.active_realizations)

        assert (
            LibresFacade(ert)
            .gather_gen_kw_data("prior", "KW_NAME:MY_KEYWORD")
            .values.flatten()
            .tolist()
            == expected
        )


@pytest.mark.integration_test
def test_field_param_update(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
NUM_REALIZATIONS 5
OBS_CONFIG observations

FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param.grdecl FORWARD_INIT:True
GRID MY_EGRID.EGRID

GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
INSTALL_JOB poly_eval POLY_EVAL
SIMULATION_JOB poly_eval
TIME_MAP time_map
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        my_param = np.ndarray(shape=(10, 10, 1), dtype=float)
        my_param.fill(math.exp(2.5))
        gp = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=my_param,
            grid=grid,
            name="MY_PARAM",
        )
        gp.to_file("my_param.grdecl", fformat="grdecl")

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    """#!/usr/bin/env python
import xtgeo
import numpy as np
import os

if __name__ == "__main__":
    if not os.path.exists("my_param.grdecl"):
        grid= xtgeo.create_box_grid(dimension=(10,10,1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        my_param = np.ndarray(shape=(10,10,1), buffer=np.random.random_sample(100))
        gp = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=my_param,
            grid=grid,
            name="MY_PARAM",
        )
        gp.to_file("my_param.grdecl", fformat="grdecl")

    a= np.random.standard_normal()
    b= np.random.standard_normal()
    c= np.random.standard_normal()
    output = [a * x**2 + b * x + c for x in range(10)]
    with open("gen_data_0.out", "w") as f:
        f.write("\\n".join(map(str, output)))
        """
                )
            )
        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        with open("POLY_EVAL", "w", encoding="utf-8") as fout:
            fout.write("EXECUTABLE forward_model")
        with open("observations", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = MY_RESPONSE;
                INDEX_LIST = 0,2,4,6,8;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
                )
            )

        with open("obs.txt", "w", encoding="utf-8") as fobs:
            fobs.write(
                dedent(
                    """
            2.1457049781272213 0.6
            8.769219841380755 1.4
            12.388014786122742 3.0
            25.600464531354252 5.4
            42.35204755970952 8.6"""
                )
            )

        with open("time_map", "w", encoding="utf-8") as fobs:
            fobs.write("2014-09-10")

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "prior",
                "--target-case",
                "smoother_update",
                "config.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)
        ert = EnKFMain(ErtConfig.from_file("config.ert"))
        prior = ert.storage_manager["prior"]
        posterior = ert.storage_manager["smoother_update"]

        prior_param = prior.load_field("MY_PARAM", list(range(5)))
        posterior_param = posterior.load_field("MY_PARAM", list(range(5)))

        assert prior_param.shape == (100, 5)
        assert posterior_param.shape == (100, 5)

        pp0 = posterior_param[:, 0]
        pp1 = posterior_param[:, 1]
        pp2 = posterior_param[:, 2]
        pp3 = posterior_param[:, 3]
        pp4 = posterior_param[:, 4]

        assert not np.equal(pp0, pp1).all()
        assert not np.equal(pp1, pp2).all()
        assert not np.equal(pp2, pp3).all()
        assert not np.equal(pp3, pp4).all()


@pytest.mark.integration_test
def test_surface_param_update(tmpdir):
    """Full update with a surface parameter, it mirrors the poly example,
    except it uses SURFACE instead of GEN_KW.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
NUM_REALIZATIONS 5
QUEUE_OPTION LOCAL MAX_RUNNING 5
OBS_CONFIG observations
SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap BASE_SURFACE:surf.irap FORWARD_INIT:True
GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
INSTALL_JOB poly_eval POLY_EVAL
SIMULATION_JOB poly_eval
TIME_MAP time_map
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        expect_surface = Surface(
            nx=1, ny=3, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        expect_surface.write("surf.irap")

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    """#!/usr/bin/env python
from ecl.util.geometry import Surface
import numpy as np
import os

if __name__ == "__main__":
    if not os.path.exists("surf.irap"):
        surf = Surface(nx=1, ny=3, xinc=1, yinc=1, xstart=1, ystart=1, angle=0)
        values = np.random.standard_normal(surf.getNX() * surf.getNY())
        for i, value in enumerate(values):
            surf[i] = value
            surf.write(f"surf.irap")
    a, b, c = list(Surface(filename="surf.irap"))
    output = [a * x**2 + b * x + c for x in range(10)]
    with open("gen_data_0.out", "w") as f:
        f.write("\\n".join(map(str, output)))
        """
                )
            )
        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        with open("POLY_EVAL", "w", encoding="utf-8") as fout:
            fout.write("EXECUTABLE forward_model")
        with open("observations", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = MY_RESPONSE;
                INDEX_LIST = 0,2,4,6,8;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
                )
            )

        with open("obs.txt", "w", encoding="utf-8") as fobs:
            fobs.write(
                dedent(
                    """
            2.1457049781272213 0.6
            8.769219841380755 1.4
            12.388014786122742 3.0
            25.600464531354252 5.4
            42.35204755970952 8.6"""
                )
            )

        with open("time_map", "w", encoding="utf-8") as fobs:
            fobs.write("2014-09-10")

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "prior",
                "--target-case",
                "smoother_update",
                "config.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)
        ert = EnKFMain(ErtConfig.from_file("config.ert"))
        prior = ert.storage_manager["prior"]
        posterior = ert.storage_manager["smoother_update"]
        prior_param = prior.load_surface_data("MY_PARAM", list(range(5)))
        posterior_param = posterior.load_surface_data("MY_PARAM", list(range(5)))
        assert np.linalg.det(np.cov(prior_param)) > np.linalg.det(
            np.cov(posterior_param)
        )
