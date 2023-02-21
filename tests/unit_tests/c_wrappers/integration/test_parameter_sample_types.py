import logging
import os
import stat
from argparse import ArgumentParser
from contextlib import ExitStack as does_not_raise
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import cwrap
import numpy as np
import pytest
from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGrid
from ecl.util.geometry import Surface

from ert.__main__ import ert_parser
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert._clib import update
from ert._clib.update import Parameter
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.libres_facade import LibresFacade


def write_file(fname, contents):
    with open(fname, mode="w", encoding="utf-8") as fout:
        fout.writelines(contents)


def create_runpath(config, active_mask=None):
    active_mask = [True] if active_mask is None else active_mask
    ert_config = ErtConfig.from_file(config)
    ert = EnKFMain(ert_config)

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
def test_field_param(tmpdir, config_str, expect_forward_init):
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

        expect_param = EclKW("MY_PARAM", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        for i in range(grid.getGlobalSize()):
            expect_param[i] = i

        with cwrap.open("my_param_0.grdecl", mode="w") as f:
            grid.write_grdecl(expect_param, f)
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
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

            parameter = update.Parameter("MY_PARAM")
            config_node = ert.ensembleConfig().getNode(parameter.name)
            with pytest.raises(KeyError, match="No parameter: MY_PARAM in storage"):
                fs.load_parameter(config_node, [0], parameter)

            # We try to load the parameters from the forward model, this would fail if
            # forward init was not set correctly
            assert load_from_forward_model(ert) == 1

            # Once data has been internalised, ERT will generate the
            # parameter files
            create_runpath("config.ert")

        with cwrap.open("simulations/realization-0/iter-0/my_param.grdecl", "rb") as f:
            actual_param = EclKW.read_grdecl(f, "MY_PARAM")
        assert actual_param == expect_param

        if expect_forward_init:
            parameter = update.Parameter("MY_PARAM")
            config_node = ert.ensembleConfig().getNode(parameter.name)
            arr = fs.load_parameter(config_node, [0], parameter)

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
            "surf0.irap - failed to initialize node: MY_PARAM",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            0,
            "surf.irap - failed to initialize node: MY_PARAM",
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
            parameter = update.Parameter("MY_PARAM")
            config_node = ert.ensembleConfig().getNode(parameter.name)
            arr = fs.load_parameter(config_node, [0], parameter)
            assert arr.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
        else:
            parameter = update.Parameter("MY_PARAM")
            config_node = ert.ensembleConfig().getNode(parameter.name)
            with pytest.raises(KeyError, match="No parameter: MY_PARAM in storage"):
                fs.load_parameter(config_node, [0], parameter)


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
        parameter = Parameter("KW_NAME")
        config_node = ert.ensembleConfig().getNode(parameter.name)
        prior = fs.load_parameter(config_node, list(range(3)), parameter)
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

        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)

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

        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)

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
def test_surface_param_update(tmpdir):
    """Full update with a surface parameter, it mirrors the poly example,
    except it uses SURFACE instead of GEN_KW.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
NUM_REALIZATIONS 5
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
        parameter_name = "MY_PARAM"
        parameter_config_node = ert.ensembleConfig().getNode(parameter_name)
        prior_param = prior.load_parameter(
            parameter_config_node, list(range(5)), update.Parameter(parameter_name)
        )
        posterior_param = posterior.load_parameter(
            parameter_config_node, list(range(5)), update.Parameter(parameter_name)
        )
        assert np.linalg.det(np.cov(prior_param)) > np.linalg.det(
            np.cov(posterior_param)
        )
