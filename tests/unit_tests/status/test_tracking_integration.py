import fileinput
import logging
import os
import re
import shutil
import threading
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from jsonpath_ng import parse
from resdata.summary import Summary

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE, TEST_RUN_MODE
from ert.cli.model_factory import create_model
from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorTracker
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.state import (
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_START,
    REALIZATION_STATE_FINISHED,
)
from ert.shared.feature_toggling import FeatureToggling
from ert.storage.realization_storage_state import RealizationStorageState


def check_expression(original, path_expression, expected, msg_start):
    assert isinstance(original, dict), f"{msg_start}data is not a dict"
    jsonpath_expr = parse(path_expression)
    match_found = False
    for match in jsonpath_expr.find(original):
        match_found = True
        assert match.value == expected, (
            f"{msg_start}{str(match.full_path)} value "
            f"({match.value}) is not equal to ({expected})"
        )
    assert match_found, f"{msg_start} Nothing matched {path_expression}"


@pytest.mark.integration_test
@pytest.mark.parametrize(
    (
        "extra_config, extra_poly_eval, cmd_line_arguments,"
        "num_successful,num_iters,progress,assert_present_in_snapshot, expected_state"
    ),
    [
        pytest.param(
            "MAX_RUNTIME 5",
            "    import time; time.sleep(1000)",
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            0,
            1,
            1.0,
            [
                (".*", "reals.*.jobs.*.status", JOB_STATE_FAILURE),
                (
                    ".*",
                    "reals.*.jobs.*.error",
                    "The run is cancelled due to reaching MAX_RUNTIME",
                ),
            ],
            [RealizationStorageState.LOAD_FAILURE] * 2,
            id="ee_poly_experiment_cancelled_by_max_runtime",
        ),
        pytest.param(
            "",
            "",
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            2,
            1,
            1.0,
            [(".*", "reals.*.jobs.*.status", JOB_STATE_FINISHED)],
            [RealizationStorageState.HAS_DATA] * 2,
            id="ee_poly_experiment",
        ),
        pytest.param(
            "",
            "",
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            2,
            2,
            1.0,
            [(".*", "reals.*.jobs.*.status", JOB_STATE_FINISHED)],
            [RealizationStorageState.HAS_DATA] * 2,
            id="ee_poly_smoother",
        ),
        pytest.param(
            "",
            '    import os\n    if os.getcwd().split("/")[-2].split("-")[1] == "0": sys.exit(1)',  # noqa 501
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            1,
            1,
            # Fails halfway, due to unable to run update
            0.5,
            [
                ("0", "reals.'0'.jobs.'0'.status", JOB_STATE_FAILURE),
                ("0", "reals.'0'.jobs.'1'.status", JOB_STATE_START),
                (".*", "reals.'1'.jobs.*.status", JOB_STATE_FINISHED),
            ],
            [
                RealizationStorageState.LOAD_FAILURE,
                RealizationStorageState.HAS_DATA,
            ],
            id="ee_failing_poly_smoother",
        ),
    ],
)
def test_tracking(
    extra_config,
    extra_poly_eval,
    cmd_line_arguments,
    num_successful,
    num_iters,
    progress,
    assert_present_in_snapshot,
    expected_state,
    tmpdir,
    source_root,
    storage,
):
    experiment_folder = "poly_example"
    shutil.copytree(
        os.path.join(source_root, "test-data", f"{experiment_folder}"),
        os.path.join(str(tmpdir), f"{experiment_folder}"),
    )

    config_lines = [
        "INSTALL_JOB poly_eval2 POLY_EVAL\nSIMULATION_JOB poly_eval2\n",
        extra_config,
    ]

    with tmpdir.as_cwd():
        with open(f"{experiment_folder}/poly.ert", "a", encoding="utf-8") as fh:
            fh.writelines(config_lines)

        with fileinput.input(f"{experiment_folder}/poly_eval.py", inplace=True) as fin:
            for line in fin:
                if line.strip().startswith("coeffs"):
                    print(extra_poly_eval)
                print(line, end="")

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            cmd_line_arguments,
        )
        FeatureToggling.update_from_args(parsed)

        ert_config = ErtConfig.from_file(parsed.config)
        os.chdir(ert_config.config_path)

        model = create_model(
            ert_config,
            storage,
            parsed,
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(1024, 65535),
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        thread = threading.Thread(
            name="ert_cli_simulation_thread",
            target=model.start_simulations_thread,
            args=(evaluator_server_config,),
        )
        thread.start()

        tracker = EvaluatorTracker(
            model,
            ee_con_info=evaluator_server_config.get_connection_info(),
        )

        snapshots = {}

        for event in tracker.track():
            if isinstance(event, FullSnapshotEvent):
                snapshots[event.iteration] = event.snapshot
            if (
                isinstance(event, SnapshotUpdateEvent)
                and event.partial_snapshot is not None
            ):
                snapshots[event.iteration].merge(event.partial_snapshot.data())
            if isinstance(event, EndEvent):
                pass

        assert tracker._progress() == progress

        assert len(snapshots) == num_iters
        for snapshot in snapshots.values():
            successful_reals = list(
                filter(
                    lambda item: item[1].status == REALIZATION_STATE_FINISHED,
                    snapshot.reals.items(),
                )
            )
            assert len(successful_reals) == num_successful

        for (
            iter_expression,
            snapshot_expression,
            expected,
        ) in assert_present_in_snapshot:
            for i, snapshot in snapshots.items():
                if re.match(iter_expression, str(i)):
                    check_expression(
                        snapshot.to_dict(),
                        snapshot_expression,
                        expected,
                        f"Snapshot {i} did not match:\n",
                    )
        thread.join()
        state_map = storage.get_ensemble_by_name("default").state_map
        assert state_map[:2] == expected_state
    FeatureToggling.reset()


@pytest.mark.integration_test
@pytest.mark.parametrize(
    ("mode, cmd_line_arguments"),
    [
        pytest.param(
            TEST_RUN_MODE,
            [
                "poly_example/poly.ert",
            ],
            id="test_run",
        ),
        pytest.param(
            ENSEMBLE_EXPERIMENT_MODE,
            [
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            id="ensemble_experiment",
        ),
    ],
)
def test_setting_env_context_during_run(
    mode,
    cmd_line_arguments,
    tmpdir,
    source_root,
    storage,
):
    experiment_folder = "poly_example"
    shutil.copytree(
        os.path.join(source_root, "test-data", f"{experiment_folder}"),
        os.path.join(str(tmpdir), f"{experiment_folder}"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        cmd_line_arguments = [mode] + cmd_line_arguments
        parsed = ert_parser(
            parser,
            cmd_line_arguments,
        )
        FeatureToggling.update_from_args(parsed)

        ert_config = ErtConfig.from_file(parsed.config)
        os.chdir(ert_config.config_path)

        model = create_model(
            ert_config,
            storage,
            parsed,
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(1024, 65535),
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        thread = threading.Thread(
            name="ert_cli_simulation_thread",
            target=model.start_simulations_thread,
            args=(evaluator_server_config,),
        )
        thread.start()

        tracker = EvaluatorTracker(
            model,
            ee_con_info=evaluator_server_config.get_connection_info(),
        )

        expected = ["_ERT_SIMULATION_MODE", "_ERT_EXPERIMENT_ID", "_ERT_ENSEMBLE_ID"]
        for event in tracker.track():
            if isinstance(event, (FullSnapshotEvent, SnapshotUpdateEvent)):
                assert model._context_env_keys == expected
                for key in expected:
                    assert key in os.environ
                assert os.environ.get("_ERT_SIMULATION_MODE") == mode
            if isinstance(event, EndEvent):
                pass
        thread.join()

        # Check environment is clean after the model run ends.
        assert not model._context_env_keys
        for key in expected:
            assert key not in os.environ
    FeatureToggling.reset()


def run_sim(start_date):
    """
    Create a summary file, the contents of which are not important
    """
    summary = Summary.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    summary.add_variable("FOPR", unit="SM3/DAY")
    t_step = summary.add_t_step(1, sim_days=1)
    t_step["FOPR"] = 1
    summary.fwrite()


@pytest.mark.integration_test
def test_tracking_missing_ecl(
    tmpdir,
    source_root,
    caplog,
    storage,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        SUMMARY *
        MAX_SUBMIT 1 -- will fail first and every time
        REFCASE ECLIPSE_CASE

        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        # We create a reference case, but there will be no response
        run_sim(datetime(2014, 9, 10))
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                TEST_RUN_MODE,
                "config.ert",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        ert_config = ErtConfig.from_file(parsed.config)
        os.chdir(ert_config.config_path)

        model = create_model(
            ert_config,
            storage,
            parsed,
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(1024, 65535),
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        thread = threading.Thread(
            name="ert_cli_simulation_thread",
            target=model.start_simulations_thread,
            args=(evaluator_server_config,),
        )
        with caplog.at_level(logging.ERROR):
            thread.start()

            tracker = EvaluatorTracker(
                model,
                ee_con_info=evaluator_server_config.get_connection_info(),
            )

            failures = []

            for event in tracker.track():
                if isinstance(event, EndEvent):
                    failures.append(event)
        assert (
            f"Realization: 0 failed after reaching max submit (1):\n\t\n"
            "status from done callback: "
            "Could not find "
            f"SUMMARY file or using non unified SUMMARY file from: "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE.UNSMRY"
        ) in caplog.messages

        # Just also check that it failed for the expected reason
        assert len(failures) == 1
        assert (
            f"Realization: 0 failed after reaching max submit (1):\n\t\n"
            "status from done callback: "
            "Could not find "
            f"SUMMARY file or using non unified SUMMARY file from: "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE.UNSMRY"
        ) in failures[0].failed_msg

        thread.join()
    FeatureToggling.reset()
