import os
import re
import shutil
import threading
import fileinput
from argparse import ArgumentParser

import pytest
from jsonpath_ng import parse

from ert.ensemble_evaluator.state import (
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_START,
    REALIZATION_STATE_FINISHED,
)
from ert.ensemble_evaluator import EvaluatorTracker
from ert_shared.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.model_factory import create_model
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.libres_facade import LibresFacade
from ert_shared.main import ert_parser
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from res.enkf.enkf_main import EnKFMain
from res.enkf.res_config import ResConfig


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
        os.path.join(source_root, "test-data", "local", f"{experiment_folder}"),
        os.path.join(str(tmpdir), f"{experiment_folder}"),
    )

    config_lines = [
        "INSTALL_JOB poly_eval2 POLY_EVAL\n" "SIMULATION_JOB poly_eval2\n",
        extra_config,
    ]

    with tmpdir.as_cwd():
        with open(f"{experiment_folder}/poly.ert", "a") as fh:
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

        res_config = ResConfig(parsed.config)
        os.chdir(res_config.config_path)
        ert = EnKFMain(res_config, strict=True)
        facade = LibresFacade(ert)

        model = create_model(
            ert,
            facade.get_ensemble_size(),
            facade.get_current_case_name(),
            parsed,
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(1024, 65535), custom_host="127.0.0.1"
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
