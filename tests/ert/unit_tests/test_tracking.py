import fileinput
import json
import os
import re
from argparse import ArgumentParser
from collections.abc import Generator
from pathlib import Path

import pytest
from jsonpath_ng import parse
from pydantic import BaseModel

from _ert.threading import ErtThread
from ert.__main__ import ert_parser
from ert.config import ErtConfig
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.ensemble_evaluator.state import (
    FORWARD_MODEL_STATE_CANCELLED,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    REALIZATION_STATE_FINISHED,
)
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.run_models.model_factory import create_model


class Events:
    def __init__(self) -> None:
        self.events: list[BaseModel] = []
        self.environment = []

    def __iter__(self) -> Generator[BaseModel]:
        yield from self.events

    def put(self, event):
        self.events.append(event)
        self.environment.append(os.environ.copy())


def check_expression(original, path_expression, expected: list[str], msg_start):
    assert isinstance(original, dict), f"{msg_start}data is not a dict"
    jsonpath_expr = parse(path_expression)
    match_found = False
    for match in jsonpath_expr.find(original):
        match_found = True
        assert match.value in expected, (
            f"{msg_start}{match.full_path!s} value "
            f"({match.value}) is not equal to ({expected})"
        )
    assert match_found, f"{msg_start} Nothing matched {path_expression}"


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    (
        "extra_config",
        "extra_poly_eval",
        "cmd_line_arguments",
        "num_successful",
        "num_iters",
        "assert_present_in_snapshot",
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
            [
                (".*", "reals.*.fm_steps.*.status", FORWARD_MODEL_STATE_FAILURE),
                (
                    ".*",
                    "reals.*.fm_steps.*.error",
                    [
                        "The run is cancelled due to reaching MAX_RUNTIME",
                        "Forward model was terminated",
                    ],
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
            [(".*", "reals.*.fm_steps.*.status", FORWARD_MODEL_STATE_FINISHED)],
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
            [(".*", "reals.*.fm_steps.*.status", FORWARD_MODEL_STATE_FINISHED)],
            id="ee_poly_smoother",
        ),
        pytest.param(
            "",
            (
                "    import os\n"
                "    import sys\n"
                '    if os.getcwd().split("/")[-2].split("-")[1] == "0": sys.exit(1)'
            ),
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--realizations",
                "0,1",
                "poly.ert",
            ],
            1,
            1,
            # Fails halfway, due to unable to run update
            [
                (
                    "0",
                    "reals.'0'.fm_steps.'0'.status",
                    FORWARD_MODEL_STATE_FAILURE,
                ),
                ("0", "reals.'0'.fm_steps.'1'.status", FORWARD_MODEL_STATE_CANCELLED),
                (
                    ".*",
                    "reals.'1'.fm_steps.*.status",
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
    assert_present_in_snapshot,
    storage,
    monkeypatch,
):
    config_lines = [
        "INSTALL_JOB poly_eval2 POLY_EVAL\nFORWARD_MODEL poly_eval2\n",
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
    monkeypatch.chdir(ert_config.config_path)

    queue = Events()
    model = create_model(
        ert_config,
        parsed,
        queue,
    )

    evaluator_server_config = EvaluatorServerConfig(use_token=False)
    model.start_simulations_thread(evaluator_server_config)

    snapshots: dict[str, EnsembleSnapshot] = {}

    assert isinstance(queue.events[-1], EndEvent)

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
                lambda item: item[1]["status"] == REALIZATION_STATE_FINISHED,
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
    ("mode", "cmd_line_arguments"),
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
    monkeypatch,
):
    parser = ArgumentParser(prog="test_main")
    cmd_line_arguments = [mode, *cmd_line_arguments]
    parsed = ert_parser(
        parser,
        cmd_line_arguments,
    )

    ert_config = ErtConfig.from_file(parsed.config)
    monkeypatch.chdir(ert_config.config_path)

    evaluator_server_config = EvaluatorServerConfig(use_token=False)
    queue = Events()
    model = create_model(
        ert_config,
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
    for event, environment in zip(queue.events, queue.environment, strict=False):
        if isinstance(event, FullSnapshotEvent | SnapshotUpdateEvent):
            for key in expected:
                assert key in environment
            assert environment.get("_ERT_SIMULATION_MODE") == mode
        if isinstance(event, EndEvent):
            pass

    # Check environment is clean after the model run ends.
    assert not model._context_env
    for key in expected:
        assert key not in os.environ


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    ("mode", "cmd_line_arguments"),
    [
        pytest.param(
            ENSEMBLE_SMOOTHER_MODE,
            ["--realizations", "0,1", "poly.ert"],
            id=ENSEMBLE_SMOOTHER_MODE,
        ),
        pytest.param(
            ENSEMBLE_EXPERIMENT_MODE,
            ["--realizations", "0,1", "poly.ert"],
            id=ENSEMBLE_EXPERIMENT_MODE,
        ),
    ],
)
def test_run_information_present_as_env_var_in_fm_context(
    mode,
    cmd_line_arguments,
    storage,
    monkeypatch,
):
    expected = ["_ERT_SIMULATION_MODE", "_ERT_EXPERIMENT_ID", "_ERT_ENSEMBLE_ID"]

    extra_poly_eval = """    import os\n"""
    for key in expected:
        extra_poly_eval += f"""    assert "{key}" in os.environ\n"""

    with fileinput.input("poly_eval.py", inplace=True) as fin:
        for line in fin:
            if line.strip().startswith("coeffs"):
                print(extra_poly_eval)
            print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, [mode, *cmd_line_arguments])

    ert_config = ErtConfig.from_file(parsed.config)
    monkeypatch.chdir(ert_config.config_path)

    evaluator_server_config = EvaluatorServerConfig(use_token=False)
    queue = Events()
    model = create_model(ert_config, parsed, queue)

    thread = ErtThread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )
    thread.start()
    thread.join()
    for event in queue.events:
        if isinstance(event, EndEvent):
            assert not event.failed, event.msg

    # Check environment is clean after the model run ends.
    for key in expected:
        assert key not in os.environ

    # Check run information in job environment
    for path in model.paths:
        with open(Path(path) / "jobs.json", encoding="utf-8") as f:
            jobs_data = json.load(f)
        for key in expected:
            assert key in jobs_data["global_environment"]
            if key == "_ERT_SIMULATION_MODE":
                assert jobs_data["global_environment"][key] == mode
