import asyncio
import logging
import os
import shutil
import threading
from argparse import ArgumentParser
from unittest.mock import Mock, call

import pandas as pd
import pytest

import ert.shared
from ert import LibresFacade, ensemble_evaluator
from ert.__main__ import ert_parser
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.cli.main import ErtCliError, run_cli
from ert.shared.feature_toggling import FeatureToggling
from ert.storage import open_storage


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


@pytest.mark.parametrize(
    "mode, target",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE, "target", id=f"{ENSEMBLE_SMOOTHER_MODE}"),
        pytest.param(
            ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
            "iter-%d",
            id=f"{ITERATIVE_ENSEMBLE_SMOOTHER_MODE}",
        ),
        pytest.param(ES_MDA_MODE, "iter-%d", id=f"{ES_MDA_MODE}"),
    ],
)
@pytest.mark.integration_test
def test_cli_does_not_run_without_observations(tmpdir, source_root, mode, target):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    def remove_line(file_name, line_num):
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(file_name, "w", encoding="utf-8") as f:
            f.writelines(lines[: line_num - 1] + lines[line_num:])

    with tmpdir.as_cwd():
        # Remove observations from config file
        remove_line("poly_example/poly.ert", 8)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                mode,
                "--target-case",
                target,
                "poly_example/poly.ert",
            ],
        )
        with pytest.raises(
            RuntimeError, match=f"To run {mode}, observations are needed."
        ):
            run_cli(parsed)


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
    prior_mask,
    reals_rerun_option,
    should_resample,
    tmpdir,
    source_root,
    capsys,
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        ert = EnKFMain(ErtConfig.from_file("poly_example/poly.ert"))
        prior_mask = prior_mask or [True] * ert.getEnsembleSize()
        storage = open_storage(ert.ert_config.ens_path, mode="w")
        experiment_id = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment_id, name="iter-0", ensemble_size=ert.getEnsembleSize()
        )
        prior_ensemble_id = ensemble.id
        prior_context = ert.ensemble_context(ensemble, prior_mask, 0)
        ert.sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        facade = LibresFacade(ert)
        prior_values = facade.load_all_gen_kw_data(
            storage.get_ensemble_by_name("iter-0")
        )
        storage.close()

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly_example/poly.ert",
                "--current-case=iter-0",
                "--port-range",
                "1024-65535",
                "--realizations",
                reals_rerun_option,
            ],
        )

        FeatureToggling.update_from_args(parsed)
        run_cli(parsed)
        post_facade = LibresFacade.from_config_file("poly.ert")
        storage = open_storage(ert.ert_config.ens_path, mode="w")
        parameter_values = post_facade.load_all_gen_kw_data(
            storage.get_ensemble(prior_ensemble_id)
        )

        if should_resample:
            with pytest.raises(AssertionError):
                pd.testing.assert_frame_equal(parameter_values, prior_values)
        else:
            pd.testing.assert_frame_equal(parameter_values, prior_values)
        storage.close()


def test_config_parser_fails_gracefully_on_unreadable_config_file(copy_case, caplog):
    """we cannot test on the config file directly, as the argument parser already check
    if the file is readable. so we use the GEN_KW parameter file which is also parsed
    using our config parser."""

    copy_case("snake_oil_field")
    config_file_name = "snake_oil_surface.ert"

    with open(config_file_name, mode="r", encoding="utf-8") as config_file_handler:
        content_lines = config_file_handler.read().splitlines()

    index_line_with_gen_kw = [
        index for index, line in enumerate(content_lines) if line.startswith("GEN_KW")
    ][0]
    gen_kw_parameter_file = content_lines[index_line_with_gen_kw].split(" ")[4]
    os.chmod(gen_kw_parameter_file, 0x0)
    gen_kw_parameter_file_abs_path = os.path.join(os.getcwd(), gen_kw_parameter_file)
    caplog.set_level(logging.WARNING)

    ErtConfig.from_file(config_file_name)

    assert (
        f"could not open file `{gen_kw_parameter_file_abs_path}` for parsing"
        in caplog.text
    )


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
        assert "Failed to open init file for parameter 'PERMX'" in str(err)


def test_surface_init_fails_during_forward_model_callback(copy_case):
    copy_case("snake_oil_field")
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
        breaking_line_with_surface_top = line_with_surface_top + " FORWARD_INIT:True"
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

    try:
        run_ert_test_run(config_file_name)
    except RuntimeError as err:
        assert (
            "Do not have permission to open observation config file "
            f"{observation_config_abs_path!r}" in str(err)
        )


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
