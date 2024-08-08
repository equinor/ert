import fileinput
import logging
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Dict

import pytest
from jsonpath_ng import parse
from resdata.summary import Summary

from _ert.threading import ErtThread
from ert.__main__ import ert_parser
from ert.config import ErtConfig
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import Snapshot
from ert.ensemble_evaluator.state import (
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_FINISHED,
)
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.run_models.model_factory import create_model


class Events:
    def __init__(self):
        self.events = []
        self.environment = []

    def __iter__(self):
        yield from self.events

    def put(self, event):
        self.events.append(event)
        self.environment.append(os.environ.copy())


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
@pytest.mark.usefixtures("copy_poly_case")
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
                "poly.ert",
            ],
            0,
            1,
            1.0,
            [
                (".*", "reals.*.forward_models.*.status", FORWARD_MODEL_STATE_FAILURE),
                (
                    ".*",
                    "reals.*.forward_models.*.error",
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
                "poly.ert",
            ],
            2,
            1,
            1.0,
            [(".*", "reals.*.forward_models.*.status", FORWARD_MODEL_STATE_FINISHED)],
            id="ee_poly_experiment",
        ),
        pytest.param(
            "",
            "",
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--realizations",
                "0,1",
                "poly.ert",
            ],
            2,
            2,
            1.0,
            [(".*", "reals.*.forward_models.*.status", FORWARD_MODEL_STATE_FINISHED)],
            id="ee_poly_smoother",
        ),
        pytest.param(
            "",
            '    import os\n    if os.getcwd().split("/")[-2].split("-")[1] == "0": sys.exit(1)',  # noqa 501
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--realizations",
                "0,1",
                "poly.ert",
            ],
            1,
            1,
            # Fails halfway, due to unable to run update
            0.5,
            [
                (
                    "0",
                    "reals.'0'.forward_models.'0'.status",
                    FORWARD_MODEL_STATE_FAILURE,
                ),
                ("0", "reals.'0'.forward_models.'1'.status", FORWARD_MODEL_STATE_START),
                (
                    ".*",
                    "reals.'1'.forward_models.*.status",
                    FORWARD_MODEL_STATE_FINISHED,
                ),
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
    storage,
):
    config_lines = [
        "INSTALL_JOB poly_eval2 POLY_EVAL\nSIMULATION_JOB poly_eval2\n",
        extra_config,
    ]

    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines(config_lines)

    with fileinput.input("poly_eval.py", inplace=True) as fin:
        for line in fin:
            if line.strip().startswith("coeffs"):
                print(extra_poly_eval)
            print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        cmd_line_arguments,
    )

    ert_config = ErtConfig.from_file(parsed.config)
    os.chdir(ert_config.config_path)

    queue = Events()
    model = create_model(
        ert_config,
        storage,
        parsed,
        queue,
    )

    evaluator_server_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )

    thread = ErtThread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )
    thread.start()

    snapshots: Dict[str, Snapshot] = {}

    thread.join()
    for event in queue:
        if isinstance(event, FullSnapshotEvent):
            snapshots[event.iteration] = event.snapshot
        if isinstance(event, SnapshotUpdateEvent) and event.snapshot is not None:
            snapshots[event.iteration].merge_snapshot(event.snapshot)
        if isinstance(event, EndEvent):
            pass

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


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    ("mode, cmd_line_arguments"),
    [
        pytest.param(
            TEST_RUN_MODE,
            [
                "poly.ert",
            ],
            id="test_run",
        ),
        pytest.param(
            ENSEMBLE_EXPERIMENT_MODE,
            [
                "--realizations",
                "0,1",
                "poly.ert",
            ],
            id="ensemble_experiment",
        ),
    ],
)
def test_setting_env_context_during_run(
    mode,
    cmd_line_arguments,
    storage,
):
    parser = ArgumentParser(prog="test_main")
    cmd_line_arguments = [mode] + cmd_line_arguments
    parsed = ert_parser(
        parser,
        cmd_line_arguments,
    )

    ert_config = ErtConfig.from_file(parsed.config)
    os.chdir(ert_config.config_path)

    evaluator_server_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )
    queue = Events()
    model = create_model(
        ert_config,
        storage,
        parsed,
        queue,
    )

    thread = ErtThread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )
    thread.start()
    thread.join()

    expected = ["_ERT_SIMULATION_MODE", "_ERT_EXPERIMENT_ID", "_ERT_ENSEMBLE_ID"]
    for event, environment in zip(queue.events, queue.environment):
        if isinstance(event, (FullSnapshotEvent, SnapshotUpdateEvent)):
            for key in expected:
                assert key in environment
            assert environment.get("_ERT_SIMULATION_MODE") == mode
        if isinstance(event, EndEvent):
            pass

    # Check environment is clean after the model run ends.
    assert not model._context_env_keys
    for key in expected:
        assert key not in os.environ


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
def test_tracking_missing_ecl(tmpdir, caplog, storage):
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

        ert_config = ErtConfig.from_file(parsed.config)
        os.chdir(ert_config.config_path)
        events = Events()
        model = create_model(
            ert_config,
            storage,
            parsed,
            events,
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(1024, 65535),
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        thread = ErtThread(
            name="ert_cli_simulation_thread",
            target=model.start_simulations_thread,
            args=(evaluator_server_config,),
        )
        with caplog.at_level(logging.ERROR):
            thread.start()
            thread.join()
            failures = []

            for event in events:
                if isinstance(event, EndEvent):
                    failures.append(event)
        assert (
            f"Realization: 0 failed after reaching max submit (1):\n\t\n"
            "status from done callback: "
            "Could not find any unified "
            f"summary file matching case path "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE"
        ) in caplog.messages

        # Just also check that it failed for the expected reason
        assert len(failures) == 1
        assert (
            f"Realization: 0 failed after reaching max submit (1):\n\t\n"
            "status from done callback: "
            "Could not find any unified "
            f"summary file matching case path "
            f"{Path().absolute()}/simulations/realization-0/"
            "iter-0/ECLIPSE_CASE"
        ) in failures[0].msg
        case = f"{Path().absolute()}/simulations/realization-0/iter-0/ECLIPSE_CASE"
        assert (
            f"Expected file {case}.UNSMRY not created by forward model!\nExpected "
            f"file {case}.SMSPEC not created by forward model!"
        ) in caplog.messages
        assert (
            f"Expected file {case}.UNSMRY not created by forward model!\nExpected "
            f"file {case}.SMSPEC not created by forward model!"
        ) in failures[0].msg
