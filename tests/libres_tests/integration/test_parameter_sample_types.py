import logging
import os
import sys
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent

import cwrap
import numpy as np
import pytest
from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGrid
from ecl.util.geometry import Surface

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert._clib import update
from ert._clib.update import Parameter
from ert.libres_facade import LibresFacade


def write_file(fname, contents):
    with open(fname, "w") as fout:
        fout.writelines(contents)


def create_runpath(config, active_mask=None):
    active_mask = [True] if active_mask is None else active_mask
    res_config = ResConfig(config)
    ert = EnKFMain(res_config)

    run_context = ert.create_ensemble_experiment_run_context(
        active_mask=active_mask,
        iteration=0,
    )
    ert.createRunPath(run_context)
    return ert


def load_from_forward_model(ert):
    facade = LibresFacade(ert)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model("default_0", realizations, 0)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "config_str, expected, extra_files, expectation",
    [
        (
            "GEN_KW KW_NAME template.txt kw.txt prior.txt\nRANDOM_SEED 1234",
            "MY_KEYWORD -0.996621"
            if sys.platform == "darwin"
            else "MY_KEYWORD 1.85656",
            [],
            does_not_raise(),
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
        with open("config.ert", "w") as fh:
            fh.writelines(config)
        with open("template.txt", "w") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        for fname, contents in extra_files:
            write_file(fname, contents)
        create_runpath("config.ert")
        with expectation:
            assert (
                Path("simulations/realization0/kw.txt").read_text("utf-8") == expected
            )


@pytest.mark.parametrize(
    "config_str, expected",
    [
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:../../my_param_0.grdecl FORWARD_INIT:True",  # noqa
            True,
        ),
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl",
            False,
        ),
    ],
)
def test_field_param(tmpdir, config_str, expected):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GRID MY_GRID.GRID
        """
        )
        config += config_str
        grid = EclGrid.create_rectangular(
            (4, 4, 1), (1, 1, 1)  # This is minimum size, any smaller will util_abort
        )
        grid.save_GRID("MY_GRID.GRID")

        poro = EclKW("MY_PARAM", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        for i in range(grid.getGlobalSize()):
            poro[i] = i

        with cwrap.open("my_param_0.grdecl", mode="w") as f:
            grid.write_grdecl(poro, f)
        with open("config.ert", "w") as fh:
            fh.writelines(config)
        ert = create_runpath("config.ert")
        assert ert.ensembleConfig()["MY_PARAM"].getUseForwardInit() is expected
        # We try to load the parameters from the forward model, this would fail if
        # forward init was not set correctly
        assert load_from_forward_model(ert) == 1

        fs = ert.getEnkfFsManager().getFileSystem("default_0", read_only=True)
        if expected:
            arr = fs.load_parameter(
                ert.ensembleConfig(), [0], update.Parameter("MY_PARAM")
            )
            assert len(arr) == 16
        else:
            # load_parameter should probably handle errors better than
            # to throw an exception.
            with pytest.raises(IndexError):
                fs.load_parameter(
                    ert.ensembleConfig(), [0], update.Parameter("MY_PARAM")
                )


@pytest.mark.parametrize(
    "config_str, expected, expect_loaded, error",
    [
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap   INIT_FILES:surf%d.irap   BASE_SURFACE:surf0.irap",  # noqa
            False,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:../../surf%d.irap BASE_SURFACE:surf0.irap FORWARD_INIT:True",  # noqa
            True,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf%d.irap BASE_SURFACE:surf0.irap FORWARD_INIT:True",  # noqa
            True,
            0,
            "surf0.irap - failed to initialize node: MY_PARAM",
        ),
    ],
)
def test_surface_param(
    tmpdir,
    config_str,
    expected,
    expect_loaded,
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
        s0 = Surface(nx=2, ny=2, xinc=1, yinc=1, xstart=1, ystart=1, angle=0)
        s0.write("surf.irap")
        s0.write("surf0.irap")

        with open("config.ert", "w") as fh:
            fh.writelines(config)
        ert = create_runpath("config.ert")
        assert ert.ensembleConfig()["MY_PARAM"].getUseForwardInit() is expected
        # We try to load the parameters from the forward model, this would fail if
        # forward init was not set correctly
        assert load_from_forward_model(ert) == expect_loaded
        assert error in "".join(caplog.messages)

        fs = ert.getEnkfFsManager().getFileSystem("default_0", read_only=True)
        if expected and expect_loaded:
            arr = fs.load_parameter(
                ert.ensembleConfig(), [0], update.Parameter("MY_PARAM")
            )
            assert len(arr) == 4
        else:
            # load_parameter should probably handle errors better than
            # to throw an exception.
            with pytest.raises(IndexError):
                fs.load_parameter(
                    ert.ensembleConfig(), [0], update.Parameter("MY_PARAM")
                )


@pytest.mark.integration_test
@pytest.mark.parametrize("load_forward_init", [True, False])
def test_gen_kw_forward_init(tmpdir, load_forward_init):
    with tmpdir.as_cwd():
        config = dedent(
            f"""
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt FORWARD_INIT:{str(load_forward_init)} INIT_FILES:custom_param0
        """  # noqa
        )
        with open("config.ert", "w") as fh:
            fh.writelines(config)

        with open("template.txt", "w") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        if not load_forward_init:
            write_file("custom_param0", "MY_KEYWORD 1.31")

        with pytest.raises(
            KeyError,
            match=(
                "Loading GEN_KW from files created by "
                "the forward model is not supported."
            ),
        ):
            create_runpath("config.ert")


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
    with caplog.at_level(logging.INFO):
        with tmpdir.as_cwd():
            config = dedent(
                """
            JOBNAME my_name%d
            NUM_REALIZATIONS 1
            GEN_KW KW_NAME template.txt kw.txt prior.txt
            """
            )
            with open("config.ert", "w") as fh:
                fh.writelines(config)
            with open("template.txt", "w") as fh:
                fh.writelines("MY_KEYWORD <MY_KEYWORD>")
            with open("prior.txt", "w") as fh:
                fh.writelines("MY_KEYWORD NORMAL 0 1")
            create_runpath("config.ert")
            # We read the first parameter value as a reference value
            expected = Path("simulations/realization0/kw.txt").read_text("utf-8")

            # Make a clean directory for the second case, which is identical
            # to the first, except that it uses the random seed from the first
            os.makedirs("second")
            os.chdir("second")
            random_seed = next(
                message
                for message in caplog.messages
                if message.startswith("RANDOM_SEED")
            ).split()[1]
            if check_random_seed:
                config += f"RANDOM_SEED {random_seed}"
            with open("config_2.ert", "w") as fh:
                fh.writelines(config)
            with open("template.txt", "w") as fh:
                fh.writelines("MY_KEYWORD <MY_KEYWORD>")
            with open("prior.txt", "w") as fh:
                fh.writelines("MY_KEYWORD NORMAL 0 1")

            create_runpath("config_2.ert")
            with expectation:
                assert (
                    Path("simulations/realization0/kw.txt").read_text("utf-8")
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
        with open("config.ert", "w") as fh:
            fh.writelines(config)
        with open("template.txt", "w") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        ert = create_runpath("config.ert", [True] * 3)
        fs = ert.getCurrentFileSystem()
        prior = fs.load_parameter(
            ert.ensembleConfig(), list(range(3)), Parameter("KW_NAME")
        )
        expected = (
            [-0.9966211, 0.01418702, -0.44262875]
            if sys.platform.startswith("darwin")
            else [1.8565558, -1.0516311, 1.6020288]
        )
        np.testing.assert_almost_equal(prior, np.array([expected]))
