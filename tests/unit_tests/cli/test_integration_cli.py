import asyncio
import fileinput
import os
import shutil
import threading
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import Mock, call

import pandas as pd
import pytest

import ert.shared
from ert import LibresFacade
from ert.__main__ import ert_parser
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.cli.main import ErtCliError, run_cli
from ert.shared.feature_toggling import FeatureToggling


@pytest.fixture(name="mock_cli_run")
def fixture_mock_cli_run(monkeypatch):
    mocked_monitor = Mock()
    mocked_thread_start = Mock()
    mocked_thread_join = Mock()
    monkeypatch.setattr(threading.Thread, "start", mocked_thread_start)
    monkeypatch.setattr(threading.Thread, "join", mocked_thread_join)
    monkeypatch.setattr(ert.cli.monitor.Monitor, "monitor", mocked_monitor)
    yield mocked_monitor, mocked_thread_join, mocked_thread_start


@pytest.mark.integration_test
def test_target_case_equal_current_case(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )
    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "test_case",
                "--target-case",
                "test_case",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        with pytest.raises(ErtCliError, match="They were both: test_case"):
            run_cli(parsed)


@pytest.mark.integration_test
def test_runpath_file(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        with open("poly_example/poly.ert", "a", encoding="utf-8") as fh:
            config_lines = [
                "LOAD_WORKFLOW_JOB ASSERT_RUNPATH_FILE\n"
                "LOAD_WORKFLOW TEST_RUNPATH_FILE\n",
                "HOOK_WORKFLOW TEST_RUNPATH_FILE PRE_SIMULATION\n",
            ]

            fh.writelines(config_lines)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)

        assert os.path.isfile("RUNPATH_WORKFLOW_0.OK")
        assert os.path.isfile("RUNPATH_WORKFLOW_1.OK")


@pytest.mark.integration_test
def test_ensemble_evaluator(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
def test_es_mda(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ES_MDA_MODE,
                "--target-case",
                "iter-%d",
                "--realizations",
                "1,2,4,8,16",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
                "--weights",
                "1",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
def test_ensemble_evaluator_disable_monitoring(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--disable-monitoring",
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
def test_cli_test_run(tmpdir, source_root, mock_cli_run):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                TEST_RUN_MODE,
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)

    monitor_mock, thread_join_mock, thread_start_mock = mock_cli_run
    monitor_mock.assert_called_once()
    thread_join_mock.assert_called_once()
    thread_start_mock.assert_has_calls([[call(), call()]])


@pytest.mark.integration_test
def test_that_logged_errors_are_propagated_from_callback(copy_case):
    """We create a case where the forward model is out of sync with the refcase
    and check that the error message is shown to the user. To do that we remove
    the other parts of the forward model and the GEN_DATA from the case
    """
    copy_case("snake_oil")
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if (
                line.startswith("GEN_DATA")
                or "SNAKE_OIL_NPV" in line
                or "SNAKE_OIL_DIFF" in line
            ):
                continue
            print(line, end="")
    with fileinput.input(
        Path() / "jobs" / "snake_oil_simulator.py", inplace=True
    ) as fin:
        for line in fin:
            if "datetime(2010, 1, 1)" in line:
                line = line.replace("datetime(2010, 1, 1)", "datetime(2000, 1, 1)")
            print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            "snake_oil.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    FeatureToggling.update_from_args(parsed)
    with pytest.raises(
        ErtCliError, match="Realization: 0 failed with: 201 inconsistencies in time_map"
    ):
        run_cli(parsed)
    FeatureToggling.reset()


@pytest.mark.integration_test
def test_ies(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "iter-%d",
                "--realizations",
                "1,2,4,8,16",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
@pytest.mark.timeout(40)
def test_experiment_server_ensemble_experiment(tmpdir, source_root, capsys):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
                "--enable-experiment-server",
                "--realizations",
                "0-4",
            ],
        )

        FeatureToggling.update_from_args(parsed)
        run_cli(parsed)
        captured = capsys.readouterr()
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        assert captured.out == "Successful realizations: 5\n"

    FeatureToggling.reset()


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
    with pytest.raises(ConfigValidationError, match=r"Parsing config file.*errors"):
        run_cli(parsed)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "prior_mask,reals_rerun_option,should_resample",
    [
        pytest.param(
            None, "0-4", False, id="All realisations first, subset second run"
        ),
        pytest.param(
            [False, True, True, True, True],
            "2-3",
            False,
            id="Subset of realisation first run, subs-subset second run",
        ),
        pytest.param(
            [True] * 3,
            "0-5",
            True,
            id="Subset of realisation first, superset in second run - must resample",
        ),
    ],
)
def test_that_prior_is_not_overwritten_in_ensemble_experiment(
    prior_mask, reals_rerun_option, should_resample, tmpdir, source_root, capsys
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        ert = EnKFMain(ResConfig("poly_example/poly.ert"))
        prior_mask = prior_mask or [True] * ert.getEnsembleSize()
        prior_context = ert.load_ensemble_context("default", prior_mask, 0)
        ert.sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        facade = LibresFacade(ert)
        prior_values = facade.load_all_gen_kw_data("default")

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
                "--realizations",
                reals_rerun_option,
            ],
        )

        FeatureToggling.update_from_args(parsed)
        run_cli(parsed)
        post_facade = LibresFacade.from_config_file("poly.ert")
        parameter_values = post_facade.load_all_gen_kw_data("default")

        if should_resample:
            with pytest.raises(AssertionError):
                pd.testing.assert_frame_equal(parameter_values, prior_values)
        else:
            pd.testing.assert_frame_equal(parameter_values, prior_values)
