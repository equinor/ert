# pylint: disable=too-many-lines

import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest
import xtgeo

from ert import ensemble_evaluator
from ert.__main__ import ert_parser
from ert.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.cli.main import ErtCliError, run_cli
from ert.config import ConfigValidationError, ConfigWarning, ErtConfig


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
def test_bad_config_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            str(tmp_path / "test.ert"),
        ],
    )
    with pytest.raises(ConfigValidationError, match="NUM_REALIZATIONS must be set."):
        run_cli(parsed)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ITERATIVE_ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_cli_raises_exceptions_when_parameters_are_missing(mode):
    with open("poly.ert", "r", encoding="utf-8") as fin, open(
        "poly-no-gen-kw.ert", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if "GEN_KW" not in line:
                fout.write(line)

    args = Mock()
    args.config = "poly-no-gen-kw.ert"
    parser = ArgumentParser(prog="test_main")

    ert_args = [mode, "poly-no-gen-kw.ert", "--target-case"]

    testcase = "testcase" if mode is ENSEMBLE_SMOOTHER_MODE else "testcase-%d"
    ert_args.append(testcase)

    parsed = ert_parser(
        parser,
        ert_args,
    )

    with pytest.raises(
        ErtCliError,
        match=f"To run {mode}, GEN_KW, FIELD or SURFACE parameters are needed.",
    ):
        run_cli(parsed)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_cli_raises_exceptions_when_no_weight_provided_for_es_mda():
    args = Mock()
    args.config = "poly.ert"
    parser = ArgumentParser(prog="test_main")

    ert_args = ["es_mda", "poly.ert", "--target-case", "testcase-%d", "--weights", "0"]

    parsed = ert_parser(
        parser,
        ert_args,
    )

    with pytest.raises(
        ErtCliError,
        match=(
            "Operation halted: ES-MDA requires weights to proceed. "
            "Please provide appropriate weights and try again."
        ),
    ):
        run_cli(parsed)


def test_ert_config_parser_fails_gracefully_on_unreadable_config_file(
    copy_case, caplog
):
    copy_case("snake_oil_field")
    config_file_name = "snake_oil_surface.ert"
    os.chmod(config_file_name, 0x0)
    caplog.set_level(logging.WARNING)

    with pytest.raises(OSError, match="[Pp]ermission"):
        ErtConfig.from_file(config_file_name)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_field_init_file_not_readable(copy_case, monkeypatch):
    monkeypatch.setattr(
        ensemble_evaluator._wait_for_evaluator, "WAIT_FOR_EVALUATOR_TIMEOUT", 5
    )
    copy_case("snake_oil_field")
    config_file_name = "snake_oil_field.ert"
    field_file_rel_path = "fields/permx0.grdecl"
    os.chmod(field_file_rel_path, 0x0)

    try:
        run_ert_test_run(config_file_name)
    except ErtCliError as err:
        assert "Permission denied:" in str(err)


@pytest.mark.scheduler
def test_surface_init_fails_during_forward_model_callback(
    copy_case, monkeypatch, try_queue_and_scheduler
):
    copy_case("snake_oil_field")

    rng = np.random.default_rng()

    Path("./surface").mkdir()
    nx = 5
    ny = 10
    surf = xtgeo.RegularSurface(
        ncol=nx, nrow=ny, xinc=1.0, yinc=1.0, values=rng.standard_normal(size=(nx, ny))
    )
    surf.to_file("surface/surf_init_0.irap", fformat="irap_ascii")

    config_file_name = "snake_oil_surface.ert"
    parameter_name = "TOP"
    with open(config_file_name, mode="r+", encoding="utf-8") as config_file_handler:
        content_lines = config_file_handler.read().splitlines()
        index_line_with_surface_top = [
            index
            for index, line in enumerate(content_lines)
            if line.startswith(f"SURFACE {parameter_name}")
        ][0]
        line_with_surface_top = content_lines[index_line_with_surface_top]
        breaking_line_with_surface_top = line_with_surface_top
        content_lines[index_line_with_surface_top] = breaking_line_with_surface_top
        config_file_handler.seek(0)
        config_file_handler.write("\n".join(content_lines))

    try:
        run_ert_test_run(config_file_name)
    except ErtCliError as err:
        assert f"Failed to initialize parameter {parameter_name!r}" in str(err)


def test_unopenable_observation_config_fails_gracefully(copy_case):
    copy_case("snake_oil_field")
    config_file_name = "snake_oil_field.ert"
    with open(config_file_name, mode="r", encoding="utf-8") as config_file_handler:
        content_lines = config_file_handler.read().splitlines()
    index_line_with_observation_config = [
        index
        for index, line in enumerate(content_lines)
        if line.startswith("OBS_CONFIG")
    ][0]
    line_with_observation_config = content_lines[index_line_with_observation_config]
    observation_config_rel_path = line_with_observation_config.split(" ")[1]
    observation_config_abs_path = os.path.join(os.getcwd(), observation_config_rel_path)
    os.chmod(observation_config_abs_path, 0x0)

    with pytest.raises(
        ValueError,
        match="Do not have permission to open observation config file "
        f"{observation_config_abs_path!r}",
    ):
        run_ert_test_run(config_file_name)


def run_ert_test_run(config_file: str) -> None:
    parser = ArgumentParser(prog="test_run")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            config_file,
        ],
    )
    run_cli(parsed)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ITERATIVE_ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_model_raises_exception_if_active_less_than_minimum_realizations(mode):
    """
    Verify that the run model checks that active realizations 20 is less than 100
    Omit testing of SingleTestRun because that executes with 1 active realization
    regardless of configuration.
    """
    with open("poly.ert", "r", encoding="utf-8") as fin, open(
        "poly_high_min_reals.ert", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if "MIN_REALIZATIONS" in line:
                fout.write("MIN_REALIZATIONS 100")
            else:
                fout.write(line)

    args = Mock()
    args.config = "poly_high_min_reals.ert"
    parser = ArgumentParser(prog="test_main")

    ert_args = [
        mode,
        "poly_high_min_reals.ert",
        "--realizations",
        "0-19",
        "--target-case",
    ]
    ert_args.append("testcase" if mode is ENSEMBLE_SMOOTHER_MODE else "testcase-%d")

    parsed = ert_parser(
        parser,
        ert_args,
    )

    with pytest.raises(
        ErtCliError,
        match="Number of active realizations",
    ):
        run_cli(parsed)


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.scheduler
def test_that_the_model_warns_when_active_realizations_less_min_realizations(
    monkeypatch, try_queue_and_scheduler
):
    """
    Verify that the run model checks that active realizations is equal or higher than
    NUM_REALIZATIONS when running ensemble_experiment.
    A warning is issued when NUM_REALIZATIONS is higher than active_realizations.
    """
    with open("poly.ert", "r", encoding="utf-8") as fin, open(
        "poly_lower_active_reals.ert", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if "MIN_REALIZATIONS" in line:
                fout.write("MIN_REALIZATIONS 100")
            else:
                fout.write(line)

    args = Mock()
    args.config = "poly_lower_active_reals.ert"
    parser = ArgumentParser(prog="test_main")

    ert_args = [
        "ensemble_experiment",
        "poly_lower_active_reals.ert",
        "--realizations",
        "0-4",
    ]

    parsed = ert_parser(
        parser,
        ert_args,
    )

    with pytest.warns(
        ConfigWarning,
        match="Due to active_realizations 5 is lower than MIN_REALIZATIONS",
    ):
        run_cli(parsed)


@pytest.fixture
def setenv_config(tmp_path):
    config = tmp_path / "test.ert"

    # Given that environment variables are set in the config
    config.write_text(
        """
        NUM_REALIZATIONS 1
        SETENV FIRST first:$PATH
        SETENV SECOND $MYVAR
        SETENV MYVAR foo
        SETENV THIRD  TheThirdValue
        SETENV FOURTH fourth:$MYVAR
        INSTALL_JOB ECHO ECHO.txt
        FORWARD_MODEL ECHO
        """,
        encoding="utf-8",
    )
    run_script = tmp_path / "run.py"
    run_script.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        'print(os.environ["FIRST"])\n'
        'print(os.environ["SECOND"])\n'
        'print(os.environ["THIRD"])\n'
        'print(os.environ["FOURTH"])\n',
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    (tmp_path / "ECHO.txt").write_text(
        dedent(
            """
        EXECUTABLE run.py
        """
        )
    )
    return config


expected_vars = {
    "FIRST": "first:$PATH",
    "SECOND": "$MYVAR",
    "MYVAR": "foo",
    "THIRD": "TheThirdValue",
    "FOURTH": "fourth:$MYVAR",
}


@pytest.mark.scheduler
def test_that_setenv_config_is_parsed_correctly(
    setenv_config, monkeypatch, try_queue_and_scheduler
):
    config = ErtConfig.from_file(str(setenv_config))
    # then res config should read the SETENV as is
    assert config.env_vars == expected_vars


@pytest.mark.scheduler
def test_that_setenv_sets_environment_variables_in_jobs(
    setenv_config, monkeypatch, try_queue_and_scheduler
):
    # When running the jobs
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            str(setenv_config),
        ],
    )

    run_cli(parsed)

    # Then the environment variables are put into jobs.json
    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        data = json.load(f)
        global_env = data.get("global_environment")
        assert global_env == expected_vars

    path = os.environ["PATH"]

    # and then job_dispatch should expand the variables on the compute side
    with open("simulations/realization-0/iter-0/ECHO.stdout.0", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 4
        # the compute-nodes path is the same since it's running locally,
        # so we can test that we can prepend to it
        assert lines[0].strip() == f"first:{path}"
        # MYVAR is not set in the compyte node yet, so it should not be expanded
        assert lines[1].strip() == "$MYVAR"
        # THIRD is just a simple value
        assert lines[2].strip() == "TheThirdValue"
        # now MYVAR now set, so should be expanded inside the value of FOURTH
        assert lines[3].strip() == "fourth:foo"


@pytest.mark.usefixtures("use_tmpdir", "copy_poly_case")
@pytest.mark.parametrize(
    ("job_src", "script_name", "script_src", "expect_stopped"),
    [
        (
            dedent(
                """
                    STOP_ON_FAIL True
                    INTERNAL False
                    EXECUTABLE failing_script.sh
                """
            ),
            "failing_script.sh",
            dedent(
                """
                    #!/bin/bash
                    ekho helo wordl
                """
            ),
            True,
        ),
        (
            dedent(
                """
                    STOP_ON_FAIL False
                    INTERNAL False
                    EXECUTABLE failing_script.sh
                """
            ),
            "failing_script.sh",
            dedent(
                """
                    #!/bin/bash
                    ekho helo wordl
                """
            ),
            False,
        ),
        (
            dedent(
                """
                    INTERNAL False
                    EXECUTABLE failing_script.sh
                """
            ),
            "failing_script.sh",
            dedent(
                """
                    #!/bin/bash
                    STOP_ON_FAIL=False
                    ekho helo wordl
                """
            ),
            False,
        ),
        (
            dedent(
                """
                    STOP_ON_FAIL True
                    INTERNAL False
                    EXECUTABLE failing_script.sh
                """
            ),
            "failing_script.sh",
            dedent(
                """
                    #!/bin/bash
                    ekho helo wordl
                    STOP_ON_FAIL=False
                """
            ),
            True,
        ),
        (
            dedent(
                """
                   STOP_ON_FAIL False
                   INTERNAL False
                   EXECUTABLE failing_script.sh
                """
            ),
            "failing_script.sh",
            dedent(
                """
                   #!/bin/bash
                   ekho helo wordl
                   STOP_ON_FAIL=TRUE
               """
            ),
            False,
        ),
        (
            dedent(
                """
                    INTERNAL False
                    EXECUTABLE failing_script_w_stop.sh
                """
            ),
            "failing_script_w_stop.sh",
            dedent(
                """
                    #!/bin/bash
                    ekho helo wordl
                    STOP_ON_FAIL=True
                """
            ),
            True,
        ),
        (
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_ert_script.py
                """
            ),
            "failing_ert_script.py",
            """
from ert import ErtScript
class AScript(ErtScript):
    stop_on_fail = True

    def run(self):
        assert False, "failure"
""",
            True,
        ),
        (
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_ert_script.py
                    STOP_ON_FAIL False
                """
            ),
            "failing_ert_script.py",
            """
from ert import ErtScript
class AScript(ErtScript):
    stop_on_fail = True

    def run(self):
        assert False, "failure"
""",
            False,
        ),
    ],
)
@pytest.mark.scheduler
def test_that_stop_on_fail_workflow_jobs_stop_ert(
    job_src,
    script_name,
    script_src,
    expect_stopped,
    monkeypatch,
    try_queue_and_scheduler,
):
    with open("failing_job", "w", encoding="utf-8") as f:
        f.write(job_src)

    with open(script_name, "w", encoding="utf-8") as s:
        s.write(script_src)

    os.chmod(script_name, os.stat(script_name).st_mode | 0o111)

    with open("dump_failing_workflow", "w", encoding="utf-8") as f:
        f.write("failjob")

    with open("poly.ert", mode="a", encoding="utf-8") as fh:
        fh.write(
            dedent(
                """
                   LOAD_WORKFLOW_JOB failing_job failjob
                   LOAD_WORKFLOW dump_failing_workflow wffail
                   HOOK_WORKFLOW wffail POST_SIMULATION
                """
            )
        )

    parsed = ert_parser(None, args=[TEST_RUN_MODE, "poly.ert"])

    if expect_stopped:
        with pytest.raises(Exception, match="Workflow job .* failed with error"):
            run_cli(parsed)
    else:
        run_cli(parsed)
