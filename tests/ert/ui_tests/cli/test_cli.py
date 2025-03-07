import asyncio
import fileinput
import json
import logging
import os
import stat
import threading
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import xtgeo
import zmq
from psutil import NoSuchProcess, Popen, Process, ZombieProcess
from resdata.summary import Summary

import _ert.threading
import ert.shared
from _ert.forward_model_runner.client import Client
from ert.cli.main import ErtCliError
from ert.config import ConfigValidationError, ErtConfig
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EnsembleEvaluator
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
)
from ert.scheduler.job import Job
from ert.storage import open_storage

from .run_cli import run_cli


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
def test_bad_config_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")
    with pytest.raises(ConfigValidationError, match="NUM_REALIZATIONS must be set\\."):
        run_cli(TEST_RUN_MODE, "--disable-monitoring", str(tmp_path / "test.ert"))


def test_test_run_on_lsf_configuration_works_with_no_errors(tmp_path):
    (tmp_path / "test.ert").write_text(
        "NUM_REALIZATIONS 1\nQUEUE_SYSTEM LSF", encoding="utf-8"
    )
    run_cli(TEST_RUN_MODE, "--disable-monitoring", str(tmp_path / "test.ert"))


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_cli_raises_exceptions_when_parameters_are_missing(mode):
    with (
        open("poly.ert", encoding="utf-8") as fin,
        open("poly-no-gen-kw.ert", "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if "GEN_KW" not in line:
                fout.write(line)

    with pytest.raises(
        ErtCliError,
        match=f"To run {mode}, GEN_KW, FIELD or SURFACE parameters are needed.",
    ):
        run_cli(
            mode,
            "--disable-monitoring",
            "poly-no-gen-kw.ert",
            "--target-ensemble",
            "testensemble-%d",
        )


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_cli_raises_exceptions_when_no_weight_provided_for_es_mda():
    with pytest.raises(
        ErtCliError,
        match="Invalid weights: 0",
    ):
        run_cli(
            ES_MDA_MODE,
            "--disable-monitoring",
            "poly.ert",
            "--target-ensemble",
            "testensemble-%d",
            "--weights",
            "0",
        )


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_field_init_file_not_readable(monkeypatch):
    config_file_name = "snake_oil_field.ert"
    field_file_rel_path = "fields/permx0.grdecl"
    os.chmod(field_file_rel_path, 0x0)

    with pytest.raises(ErtCliError, match="Permission denied:"):
        run_cli(TEST_RUN_MODE, "--disable-monitoring", config_file_name)


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_surface_init_fails_during_forward_model_callback():
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
        index_line_with_surface_top = next(
            index
            for index, line in enumerate(content_lines)
            if line.startswith(f"SURFACE {parameter_name}")
        )
        line_with_surface_top = content_lines[index_line_with_surface_top]
        breaking_line_with_surface_top = line_with_surface_top
        content_lines[index_line_with_surface_top] = (
            breaking_line_with_surface_top.replace(
                "FORWARD_INIT:False", "FORWARD_INIT:True"
            )
        )
        config_file_handler.seek(0)
        config_file_handler.write("\n".join(content_lines))

    with pytest.raises(
        ErtCliError, match=f"Failed to initialize parameter {parameter_name!r}"
    ):
        run_cli(TEST_RUN_MODE, "--disable-monitoring", config_file_name)


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_unopenable_observation_config_fails_gracefully():
    config_file_name = "snake_oil_field.ert"
    with open(config_file_name, encoding="utf-8") as config_file_handler:
        content_lines = config_file_handler.read().splitlines()
    index_line_with_observation_config = next(
        index
        for index, line in enumerate(content_lines)
        if line.startswith("OBS_CONFIG")
    )
    line_with_observation_config = content_lines[index_line_with_observation_config]
    observation_config_rel_path = line_with_observation_config.split(" ")[1]
    observation_config_abs_path = os.path.join(os.getcwd(), observation_config_rel_path)
    os.chmod(observation_config_abs_path, 0x0)

    with pytest.raises(
        ValueError,
        match="Do not have permission to open observation config file "
        f"{observation_config_abs_path!r}",
    ):
        run_cli(TEST_RUN_MODE, config_file_name)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_model_raises_exception_if_successful_realizations_less_than_minimum_realizations(
    mode,
):
    with (
        open("poly.ert", encoding="utf-8") as fin,
        open("failing_realizations.ert", "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if "MIN_REALIZATIONS" in line:
                fout.write("MIN_REALIZATIONS 2\n")
            elif "NUM_REALIZATIONS" in line:
                fout.write("NUM_REALIZATIONS 2\n")
            else:
                fout.write(line)
        fout.write(
            dedent(
                """
            INSTALL_JOB failing_fm FAILING_FM
            FORWARD_MODEL failing_fm
            """
            )
        )
    Path("FAILING_FM").write_text("EXECUTABLE failing_fm.py", encoding="utf-8")
    Path("failing_fm.py").write_text(
        "#!/usr/bin/env python3\nraise RuntimeError('fm failed')", encoding="utf-8"
    )
    os.chmod("failing_fm.py", 0o755)

    with pytest.raises(
        ErtCliError,
        match="Number of successful realizations",
    ):
        run_cli(mode, "--disable-monitoring", "failing_realizations.ert")


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


@pytest.mark.usefixtures("set_site_config")
def test_that_setenv_config_is_parsed_correctly(setenv_config):
    config = ErtConfig.from_file(str(setenv_config))
    # then res config should read the SETENV as is
    assert config.env_vars == expected_vars


@pytest.mark.usefixtures("set_site_config")
def test_that_setenv_sets_environment_variables_in_steps(setenv_config):
    # When running the jobs
    run_cli(
        TEST_RUN_MODE,
        "--disable-monitoring",
        str(setenv_config),
    )

    # Then the environment variables are put into jobs.json
    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        data = json.load(f)
        global_env = data.get("global_environment")
        for key in ["_ERT_ENSEMBLE_ID", "_ERT_EXPERIMENT_ID"]:
            assert key in global_env
            global_env.pop(key)
        assert global_env["_ERT_SIMULATION_MODE"] == TEST_RUN_MODE
        global_env.pop("_ERT_SIMULATION_MODE")

        assert global_env == expected_vars

    path = os.environ["PATH"]

    # and then fm_dispatch should expand the variables on the compute side
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


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    (
        "workflow_job_config_content",
        "file_extension",
        "script_content",
        "expect_stopped",
    ),
    [
        pytest.param(
            dedent(
                """
                    STOP_ON_FAIL True

                    EXECUTABLE failing_script.sh
                """
            ),
            "sh",
            dedent(
                """\
                    #!/bin/bash
                    set -e
                    ekho helo wordl
                """
            ),
            True,
            id="external_bash_script__stop_on_fail_enabled",
        ),
        pytest.param(
            dedent(
                """
                   STOP_ON_FAIL False

                   EXECUTABLE failing_script.sh
                """
            ),
            "sh",
            dedent(
                """\
                   #!/bin/bash
                   set -e
                   ekho helo wordl
               """
            ),
            False,
            id="external_bash_script__stop_on_fail_disabled",
        ),
        pytest.param(
            dedent(
                """

                    EXECUTABLE failing_script.sh
                """
            ),
            "sh",
            dedent(
                """\
                    #!/bin/bash
                    set -e
                    ekho helo wordl
                """
            ),
            False,
            id="external_bash_script__stop_on_fail_disabled_by_default",
        ),
        pytest.param(
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_script.py
                """
            ),
            "py",
            dedent(
                """
                    from ert import ErtScript
                    class AScript(ErtScript):

                        def run(self):
                            assert False, "failure"
                """
            ),
            False,
            id="internal_python_script__stop_on_fail_disabled_by_default",
        ),
        pytest.param(
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_script.py
                    STOP_ON_FAIL False
                """
            ),
            "py",
            dedent(
                """
                    from ert import ErtScript
                    class AScript(ErtScript):

                        def run(self):
                            assert False, "failure"
                """
            ),
            False,
            id="internal_python_script__stop_on_failed_disabled",
        ),
        pytest.param(
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_script.py
                    STOP_ON_FAIL True
                """
            ),
            "py",
            dedent(
                """
                    from ert import ErtScript
                    class AScript(ErtScript):

                        def run(self):
                            assert False, "failure"

                """
            ),
            True,
            id="internal_python_script__stop_on_failed_enabled_in_config",
        ),
        pytest.param(
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_script.py
                """
            ),
            "py",
            dedent(
                """
                    from ert import ErtScript
                    class AScript(ErtScript):
                        stop_on_fail = True
                        def run(self):
                            assert False, "failure"

                """
            ),
            True,
            id="internal_python_script__stop_on_fail_enabled_in_script",
        ),
        pytest.param(
            dedent(
                """
                    INTERNAL True
                    SCRIPT failing_script.py
                """
            ),
            "py",
            dedent(
                """
                    from ert import ErtScript
                    class AScript(ErtScript):
                        stop_on_fail = False
                        def run(self):
                            assert False, "failure"

                """
            ),
            False,
            id="internal_python_script__stop_on_fail_disabled_in_script",
        ),
    ],
)
def test_that_stop_on_fail_workflow_jobs_stop_ert(
    workflow_job_config_content,
    file_extension,
    script_content,
    expect_stopped,
    monkeypatch,
):
    script_name = f"failing_script.{file_extension}"
    monkeypatch.setattr(_ert.threading, "_can_raise", False)

    with open("failing_job", "w", encoding="utf-8") as f:
        f.write(workflow_job_config_content)

    with open(script_name, "w", encoding="utf-8") as s:
        s.write(script_content)

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

    if expect_stopped:
        with pytest.raises(Exception, match=r"Workflow job .* failed with error"):
            run_cli(TEST_RUN_MODE, "--disable-monitoring", "poly.ert")
    else:
        run_cli(TEST_RUN_MODE, "--disable-monitoring", "poly.ert")


@pytest.fixture(name="mock_cli_run")
def fixture_mock_cli_run(monkeypatch):
    end_event = Mock()
    end_event.failed = False
    mocked_monitor = Mock(return_value=end_event)
    mocked_thread_start = Mock()
    mocked_thread_join = Mock()
    monkeypatch.setattr(threading.Thread, "start", mocked_thread_start)
    monkeypatch.setattr(threading.Thread, "join", mocked_thread_join)
    monkeypatch.setattr(ert.cli.monitor.Monitor, "monitor", mocked_monitor)
    yield mocked_monitor, mocked_thread_join, mocked_thread_start


@pytest.mark.usefixtures("copy_poly_case")
def test_es_mda(snapshot):
    with fileinput.input("poly.ert", inplace=True) as fin:
        for line_nr, line in enumerate(fin):
            if line_nr == 1:
                print("RANDOM_SEED 1234")
            print(line, end="")

    run_cli(
        ES_MDA_MODE,
        "--disable-monitoring",
        "--target-ensemble",
        "iter-%d",
        "--realizations",
        "1,2,4,8,16",
        "poly.ert",
    )

    with open_storage("storage", "r") as storage:
        data = []
        experiment = storage.get_experiment_by_name("es-mda")
        for iter_nr in range(4):
            ensemble = experiment.get_ensemble_by_name(f"iter-{iter_nr}")
            data.append(ensemble.load_all_gen_kw_data())
    result = pd.concat(
        data,
        keys=[f"iter-{iter}" for iter in range(len(data))],
        names=("Iteration", "Realization"),
    )
    snapshot.assert_match(
        result.to_csv(float_format="%.12g"), "es_mda_integration_snapshot"
    )


@pytest.mark.parametrize(
    "mode, target",
    [
        pytest.param(
            ENSEMBLE_SMOOTHER_MODE, "target_%d", id=f"{ENSEMBLE_SMOOTHER_MODE}"
        ),
        pytest.param(ES_MDA_MODE, "iter-%d", id=f"{ES_MDA_MODE}"),
    ],
)
@pytest.mark.usefixtures("copy_poly_case")
def test_cli_does_not_run_without_observations(mode, target):
    def remove_linestartswith(file_name: str, startswith: str):
        lines = Path(file_name).read_text(encoding="utf-8").split("\n")
        lines = [line for line in lines if not line.startswith(startswith)]
        Path(file_name).write_text("\n".join(lines), encoding="utf-8")

    # Remove observations from config file
    remove_linestartswith("poly.ert", "OBS_CONFIG")

    with pytest.raises(ErtCliError, match=f"To run {mode}, observations are needed."):
        run_cli(mode, "--disable-monitoring", "--target-ensemble", target, "poly.ert")


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_smoother():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "--realizations",
        "1,2,4,8,16,32,64",
        "poly.ert",
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_cli_test_run(mock_cli_run):
    run_cli(TEST_RUN_MODE, "--disable-monitoring", "poly.ert")

    monitor_mock, thread_join_mock, thread_start_mock = mock_cli_run
    monitor_mock.assert_called_once()
    thread_join_mock.assert_called_once()
    thread_start_mock.assert_has_calls([[call(), call()]])


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    "prior_mask,reals_rerun_option",
    [
        pytest.param(range(5), "0-4", id="All realisations first, subset second run"),
        pytest.param(
            [1, 2, 3, 4],
            "2-3",
            id="Subset of realisation first run, subs-subset second run",
        ),
    ],
)
def test_that_prior_is_not_overwritten_in_ensemble_experiment(
    prior_mask,
    reals_rerun_option,
    caplog,
):
    ert_config = ErtConfig.from_file("poly.ert")
    num_realizations = ert_config.model_config.num_realizations
    with open_storage(ert_config.ens_path, mode="w") as storage:
        experiment_id = storage.create_experiment(
            ert_config.ensemble_config.parameter_configuration, name="test-experiment"
        )
        ensemble = storage.create_ensemble(
            experiment_id, name="iter-0", ensemble_size=num_realizations
        )
        sample_prior(ensemble, prior_mask)
        experiment = storage.get_experiment_by_name("test-experiment")
        prior_values = experiment.get_ensemble_by_name(ensemble.name).load_parameters(
            "COEFFS"
        )["values"]
    with caplog.at_level(logging.INFO):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "--disable-monitoring",
            "poly.ert",
            "--current-case=iter-0",
            "--realizations",
            reals_rerun_option,
        )

    with open_storage(ert_config.ens_path, mode="w") as storage:
        parameter_values = storage.get_ensemble(ensemble.id).load_parameters("COEFFS")[
            "values"
        ]
        np.testing.assert_array_equal(parameter_values, prior_values)
    assert len([msg for msg in caplog.messages if "RANDOM_SEED" in msg]) == 1


@pytest.mark.usefixtures("copy_poly_case")
def test_failing_step_cli_error_message():
    # modify poly_eval.py
    with open("poly_eval.py", mode="a", encoding="utf-8") as poly_script:
        poly_script.writelines(["    raise RuntimeError('Argh')"])

    expected_substrings = [
        "Realization: 0 failed after reaching max submit (1)",
        "Step poly_eval failed",
        "Process exited with status code 1",
        "Traceback",
        "raise RuntimeError('Argh')",
        "RuntimeError: Argh",
    ]
    try:
        run_cli(TEST_RUN_MODE, "--disable-monitoring", "poly.ert")
    except ErtCliError as error:
        for substring in expected_substrings:
            assert substring in f"{error}"
    else:
        pytest.fail(msg="Expected run cli to raise ErtCliError!")


@pytest.mark.usefixtures("copy_poly_case")
def test_exclude_parameter_from_update():
    with fileinput.input("poly.ert", inplace=True) as fin:
        for line in fin:
            if "GEN_KW" in line:
                print("GEN_KW ANOTHER_KW distribution.txt UPDATE:FALSE")
            print(line, end="")
    with open("distribution.txt", mode="w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")

    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "--target-ensemble",
        "iter-%d",
        "--realizations",
        "0-5",
        "poly.ert",
    )
    with open_storage("storage", "r") as storage:
        experiment = storage.get_experiment_by_name("es")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")
        assert prior.load_parameters(
            "ANOTHER_KW", tuple(range(5))
        ) == posterior.load_parameters("ANOTHER_KW", tuple(range(5)))

    log_paths = list(Path("update_log").iterdir())
    assert log_paths
    assert (log_paths[0] / "Report.report").exists()
    assert (log_paths[0] / "Report.csv").exists()


from ert.scheduler.driver import Driver


@pytest.mark.timeout(15)
@pytest.mark.usefixtures("copy_poly_case")
def test_that_driver__init__exceptions_are_propagated(monkeypatch, capsys):
    def mocked__init__(*args, **kwargs) -> None:
        raise ValueError("Foobar error")

    monkeypatch.setattr(Driver, "__init__", mocked__init__)
    with pytest.raises(
        ErtCliError,
    ):
        run_cli(
            TEST_RUN_MODE,
            "poly.ert",
        )
    captured = capsys.readouterr()
    assert "Foobar error" in captured.err


@pytest.mark.usefixtures("copy_poly_case")
def test_that_log_is_cleaned_up_from_repeated_forward_model_steps(caplog):
    """Verify that the run model now gereneates a cleanup log when
    there are repeated forward models
    """
    with (
        open("poly.ert", encoding="utf-8") as fin,
        open("poly_repeated_forward_model_steps.ert", "w", encoding="utf-8") as fout,
    ):
        forward_model_steps = ["FORWARD_MODEL poly_eval\n"] * 5
        lines = fin.readlines() + forward_model_steps
        fout.writelines(lines)

    expected_msg = "Config contains forward model step poly_eval 6 time(s)"

    with caplog.at_level(logging.INFO):
        run_cli(
            "ensemble_experiment",
            "--disable-monitoring",
            "poly_repeated_forward_model_steps.ert",
            "--realizations",
            "0-4",
        )
    assert len([msg for msg in caplog.messages if expected_msg in msg]) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_a_custom_eclrun_can_be_activated_through_setenv():
    """Mock an eclrun binary that will output the list of valid versions and also
    mock Eclipse100 output"""
    Path("bin").mkdir()
    eclrun = Path("bin") / "eclrun"
    eclrun.write_text(
        dedent(
            """
            #!/bin/sh
            if [ "$2" = "--report-versions" ]; then
                echo "I_AM_VERSION_2044"
            else
                # Mock Eclipse100 output
                touch FOO.UNSMRY
                echo "Errors 0" > FOO.PRT
                echo "Bugs 0" >> FOO.PRT
            fi
            """
        ).strip(),
        encoding="utf-8",
    )
    os.chmod(eclrun, os.stat(eclrun).st_mode | stat.S_IEXEC)

    Path("FOO.DATA").touch()
    config_file = Path("config.ert")
    config_file.write_text(
        dedent(
            f"""
            NUM_REALIZATIONS 1
            DATA_FILE FOO.DATA
            ECLBASE FOO
            SETENV ECLRUN_PATH {eclrun.parent.absolute()}
            FORWARD_MODEL ECLIPSE100(<VERSION>=I_AM_VERSION_2044)
            """
        ).strip(),
        encoding="utf-8",
    )
    run_cli(
        TEST_RUN_MODE,
        str(config_file),
        "--disable-monitoring",
    )


def run_sim(start_date):
    """
    Create a summary file, the contents of which are not important
    """
    summary = Summary.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    summary.add_variable("FOPR", unit="SM3/DAY")
    t_step = summary.add_t_step(1, sim_days=1)
    t_step["FOPR"] = 1
    summary.fwrite()


def test_tracking_missing_ecl(monkeypatch, tmp_path, caplog):
    config_file = tmp_path / "config.ert"
    monkeypatch.chdir(tmp_path)
    config_file.write_text(
        dedent(
            """
            NUM_REALIZATIONS 2

            ECLBASE ECLIPSE_CASE
            SUMMARY *
            MAX_SUBMIT 1 -- will fail first and every time
            REFCASE ECLIPSE_CASE

            """
        )
    )
    # We create a reference case, but there will be no response
    run_sim(datetime(2014, 9, 10))
    with pytest.raises(ErtCliError):
        run_cli(
            TEST_RUN_MODE,
            str(config_file),
        )
    assert (
        f"Realization: 0 failed after reaching max submit (1):\n\t\n"
        "status from done callback: "
        "Could not find any unified "
        f"summary file matching case path "
        f"{Path().absolute()}/simulations/realization-0/"
        "iter-0/ECLIPSE_CASE"
    ) in caplog.messages

    case = f"{Path().absolute()}/simulations/realization-0/iter-0/ECLIPSE_CASE"
    assert (
        f"Expected file {case}.UNSMRY not created by forward model!\nExpected "
        f"file {case}.SMSPEC not created by forward model!"
    ) in caplog.messages


@pytest.mark.usefixtures("copy_poly_case")
def test_that_connection_errors_do_not_effect_final_result(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Client, "DEFAULT_MAX_RETRIES", 1)
    monkeypatch.setattr(Client, "DEFAULT_ACK_TIMEOUT", 1)
    monkeypatch.setattr(EnsembleEvaluator, "CLOSE_SERVER_TIMEOUT", 0.01)
    monkeypatch.setattr(Job, "DEFAULT_CHECKSUM_TIMEOUT", 0)

    def raise_connection_error(*args, **kwargs):
        raise zmq.error.ZMQError(None, None)

    with patch(
        "ert.ensemble_evaluator.evaluator.dispatch_event_from_json",
        raise_connection_error,
    ):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "poly.ert",
        )


@pytest.mark.usefixtures("copy_poly_case")
async def test_that_killed_ert_does_not_leave_storage_server_process():
    ert_subprocess = Popen(["ert", "gui", "poly.ert"])
    assert ert_subprocess.is_running()

    async def _find_storage_process_pid() -> int:
        while True:
            for ert_child_process in ert_subprocess.children():
                try:
                    if "storage" in "".join(ert_child_process.cmdline()):
                        return ert_child_process.pid
                except (ZombieProcess, NoSuchProcess):
                    pass
            await asyncio.sleep(0.05)

    storage_process_pid = await asyncio.wait_for(
        _find_storage_process_pid(), timeout=120
    )
    # wait for storage server to have connected to ert
    await asyncio.sleep(5)
    storage_process = Process(storage_process_pid)

    assert ert_subprocess.is_running()
    assert storage_process.is_running()
    kill_ert_subprocess = await asyncio.create_subprocess_exec(
        "kill", "-9", f"{ert_subprocess.pid}"
    )
    await kill_ert_subprocess.wait()

    async def _wait_for_storage_process_to_shut_down():
        storage_server_has_shutdown = asyncio.Event()
        while not storage_server_has_shutdown.is_set():
            if not storage_process.is_running():
                storage_server_has_shutdown.set()
            await asyncio.sleep(0.1)
        print(
            f"Waiting for killed ert:{ert_subprocess.pid} to stop storage:{storage_process.pid}"
        )

    await asyncio.wait_for(_wait_for_storage_process_to_shut_down(), timeout=90)
    assert not storage_process.is_running()
