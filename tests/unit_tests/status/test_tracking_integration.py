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
from ecl.summary import EclSum
from jsonpath_ng import parse

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.ert_config import ErtConfig
from ert.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE, TEST_RUN_MODE
from ert.cli.model_factory import create_model
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
from ert.libres_facade import LibresFacade
from ert.shared.feature_toggling import FeatureToggling


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
        "num_successful,num_iters,progress,assert_present_in_snapshot"
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
                (".*", "reals.*.steps.*.jobs.*.status", JOB_STATE_FAILURE),
                (
                    ".*",
                    "reals.*.steps.*.jobs.*.error",
                    "The run is cancelled due to reaching MAX_RUNTIME",
                ),
            ],
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
            [(".*", "reals.*.steps.*.jobs.*.status", JOB_STATE_FINISHED)],
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
                "12,13",
                "poly_example/poly.ert",
            ],
            2,
            2,
            1.0,
            [(".*", "reals.*.steps.*.jobs.*.status", JOB_STATE_FINISHED)],
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
                ("0", "reals.'0'.steps.*.jobs.'0'.status", JOB_STATE_FAILURE),
                ("0", "reals.'0'.steps.*.jobs.'1'.status", JOB_STATE_START),
                (".*", "reals.'1'.steps.*.jobs.*.status", JOB_STATE_FINISHED),
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
    tmpdir,
    source_root,
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
        ert = EnKFMain(ert_config)
        facade = LibresFacade(ert)

        model = create_model(
            ert,
            facade.get_ensemble_size(),
            facade.get_current_case_name(),
            parsed,
            "experiment_id",
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
            if isinstance(event, SnapshotUpdateEvent):
                if event.partial_snapshot is not None:
                    snapshots[event.iteration].merge(event.partial_snapshot.data())
            if isinstance(event, EndEvent):
                pass

        assert tracker._progress() == progress

        assert len(snapshots) == num_iters
        for iter_, snapshot in snapshots.items():
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
    FeatureToggling.reset()


def run_sim(start_date):
    """
    Create a summary file, the contents of which are not important
    """
    ecl_sum = EclSum.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    t_step = ecl_sum.addTStep(1, sim_days=1)
    t_step["FOPR"] = 1
    ecl_sum.fwrite()


@pytest.mark.integration_test
def test_tracking_time_map(
    tmpdir,
    source_root,
    caplog,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        SUMMARY *
        MAX_SUBMIT 5 -- strictly not needed, but to make sure it is > 1
        REFCASE ECLIPSE_CASE

        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        # We create a reference case
        run_sim(datetime(2014, 9, 10))
        cwd = Path().absolute()
        sim_path = Path("simulations") / "realization-0" / "iter-0"
        sim_path.mkdir(parents=True, exist_ok=True)
        os.chdir(sim_path)
        # We are a bit sneaky here, there is no forward model creating any responses
        # but ert will happily accept the results of files that are already present in
        # the run_path (feature?). So we just create a response that does not match the
        # reference case for time.
        run_sim(datetime(2017, 5, 2))
        os.chdir(cwd)
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
        ert = EnKFMain(ert_config)
        facade = LibresFacade(ert)

        model = create_model(
            ert,
            facade.get_ensemble_size(),
            facade.get_current_case_name(),
            parsed,
            "experiment_id",
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
        with caplog.at_level(logging.INFO):
            thread.start()

            tracker = EvaluatorTracker(
                model,
                ee_con_info=evaluator_server_config.get_connection_info(),
            )

            failures = []

            for event in tracker.track():
                if isinstance(event, EndEvent):
                    failures.append(event)
        # Check that max submit > 1
        assert ert_config.queue_config.max_submit == 5
        # We check that the job was submitted first time
        assert "Submitted job ECLIPSE_CASE (attempt 0)" in caplog.messages
        # We check that the job was not submitted after the first failed
        assert "Submitted job ECLIPSE_CASE (attempt 1)" not in caplog.messages

        # Just also check that it failed for the expected reason
        assert len(failures) == 1
        assert (
            "Realization: 0 failed with: 2 inconsistencies in time_map"
            in failures[0].failed_msg
        )
        thread.join()
    FeatureToggling.reset()


@pytest.mark.integration_test
def test_tracking_missing_ecl(
    tmpdir,
    source_root,
    caplog,
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
        ert = EnKFMain(ert_config)
        facade = LibresFacade(ert)

        model = create_model(
            ert,
            facade.get_ensemble_size(),
            facade.get_current_case_name(),
            parsed,
            "experiment_id",
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
            f"Realization: 0 failed after reaching max submit with: Could not find "
            f"SUMMARY file or using non unified SUMMARY file from: "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE.UNSMRY"
        ) in caplog.messages

        # Just also check that it failed for the expected reason
        assert len(failures) == 1
        assert (
            f"Realization: 0 failed after reaching max submit with: Could not find "
            f"SUMMARY file or using non unified SUMMARY file from: "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE.UNSMRY"
        ) in failures[0].failed_msg

        thread.join()
    FeatureToggling.reset()
