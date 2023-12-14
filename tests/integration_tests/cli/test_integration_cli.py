# pylint: disable=too-many-lines

import fileinput
import os
import shutil
import threading
from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

import ert.shared
from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.cli.main import ErtCliError, run_cli
from ert.config import ErtConfig
from ert.enkf_main import sample_prior
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
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
def test_es_mda(tmpdir, source_root, snapshot):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        with fileinput.input("poly_example/poly.ert", inplace=True) as fin:
            for line_nr, line in enumerate(fin):
                if line_nr == 1:
                    print("RANDOM_SEED 1234", end="")
                print(line, end="")
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
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()
        facade = LibresFacade.from_config_file("poly.ert")
        with open_storage("storage", "r") as storage:
            data = []
            for iter_nr in range(4):
                data.append(
                    facade.load_all_gen_kw_data(
                        storage.get_ensemble_by_name(f"iter-{iter_nr}")
                    )
                )
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

    def remove_linestartswith(file_name: str, startswith: str):
        lines = Path(file_name).read_text(encoding="utf-8").split("\n")
        lines = [line for line in lines if not line.startswith(startswith)]
        Path(file_name).write_text("\n".join(lines), encoding="utf-8")

    with tmpdir.as_cwd():
        # Remove observations from config file
        remove_linestartswith("poly_example/poly.ert", "OBS_CONFIG")

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
            ErtCliError, match=f"To run {mode}, observations are needed."
        ):
            run_cli(parsed)


@pytest.mark.integration_test
def test_ensemble_evaluator_disable_monitoring(
    tmpdir, source_root
):
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
        parsed = ert_parser(parser, [TEST_RUN_MODE, "poly_example/poly.ert"])
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
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.mark.integration_test
def test_that_running_ies_with_different_steplength_produces_different_result(
    tmpdir, source_root
):
    """This is a regression test to make sure that different step-lengths
    give different results when running SIES.
    """
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    def _run(target):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                f"{target}-%d",
                "--realizations",
                "1,2,4,8",
                "poly_example/poly.ert",
                "--num-iterations",
                "1",
            ],
        )
        run_cli(parsed)
        facade = LibresFacade.from_config_file("poly.ert")

        with open_storage(facade.enspath) as storage:
            iter_0_fs = storage.get_ensemble_by_name(f"{target}-0")
            df_iter_0 = facade.load_all_gen_kw_data(iter_0_fs)
            iter_1_fs = storage.get_ensemble_by_name(f"{target}-1")
            df_iter_1 = facade.load_all_gen_kw_data(iter_1_fs)

            result = pd.concat(
                [df_iter_0, df_iter_1],
                keys=["iter-0", "iter-1"],
            )
            return result

    # Run SIES with step-lengths defined
    with tmpdir.as_cwd():
        with open("poly_example/poly.ert", mode="a", encoding="utf-8") as fh:
            fh.write(
                dedent(
                    """
                RANDOM_SEED 123456
                ANALYSIS_SET_VAR IES_ENKF IES_MAX_STEPLENGTH 0.5
                ANALYSIS_SET_VAR IES_ENKF IES_MIN_STEPLENGTH 0.2
                ANALYSIS_SET_VAR IES_ENKF IES_DEC_STEPLENGTH 2.5
                """
                )
            )

        result_1 = _run("target_result_1")

    # Run SIES with different step-lengths defined
    with tmpdir.as_cwd():
        with open("poly_example/poly.ert", mode="a", encoding="utf-8") as fh:
            fh.write(
                dedent(
                    """
                ANALYSIS_SET_VAR IES_ENKF IES_MAX_STEPLENGTH 0.6
                ANALYSIS_SET_VAR IES_ENKF IES_MIN_STEPLENGTH 0.3
                ANALYSIS_SET_VAR IES_ENKF IES_DEC_STEPLENGTH 2.0
                """
                )
            )

        result_2 = _run("target_result_2")

        # Prior should be the same
        assert result_1.loc["iter-0"].equals(result_2.loc["iter-0"])

        # Posterior should be different
        assert not np.isclose(result_1.loc["iter-1"], result_2.loc["iter-1"]).all()


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "prior_mask,reals_rerun_option,should_resample",
    [
        pytest.param(
            range(5), "0-4", False, id="All realisations first, subset second run"
        ),
        pytest.param(
            [1, 2, 3, 4],
            "2-3",
            False,
            id="Subset of realisation first run, subs-subset second run",
        ),
        pytest.param(
            [0, 1, 2],
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
    source_root
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        ert_config = ErtConfig.from_file("poly_example/poly.ert")
        num_realizations = ert_config.model_config.num_realizations
        storage = open_storage(ert_config.ens_path, mode="w")
        experiment_id = storage.create_experiment(
            ert_config.ensemble_config.parameter_configuration
        )
        ensemble = storage.create_ensemble(
            experiment_id, name="iter-0", ensemble_size=num_realizations
        )
        sample_prior(ensemble, prior_mask)
        prior_values = storage.get_ensemble(ensemble.id).load_parameters("COEFFS")["values"]
        storage.close()

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly_example/poly.ert",
                "--current-case=iter-0",
                "--realizations",
                reals_rerun_option,
            ],
        )

        FeatureToggling.update_from_args(parsed)
        run_cli(parsed)
        storage = open_storage(ert_config.ens_path, mode="w")
        parameter_values = storage.get_ensemble(ensemble.id).load_parameters("COEFFS")["values"]

        if should_resample:
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(parameter_values, prior_values)
        else:
            np.testing.assert_array_equal(parameter_values, prior_values)
        storage.close()


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_failing_job_cli_error_message():
    # modify poly_eval.py
    with open("poly_eval.py", mode="a", encoding="utf-8") as poly_script:
        poly_script.writelines(["    raise RuntimeError('Argh')"])

    args = Mock()
    args.config = "poly_high_min_reals.ert"
    parser = ArgumentParser(prog="test_main")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [TEST_RUN_MODE, "poly.ert"],
    )
    expected_substrings = [
        "Realization: 0 failed after reaching max submit (2)",
        "job poly_eval failed",
        "Process exited with status code 1",
        "Traceback",
        "raise RuntimeError('Argh')",
        "RuntimeError: Argh",
    ]
    try:
        run_cli(parsed)
    except ErtCliError as error:
        for substring in expected_substrings:
            assert substring in f"{error}"
    else:
        pytest.fail(msg="Expected run cli to raise ErtCliError!")
