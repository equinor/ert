import os
import re
import shutil
import threading
from argparse import ArgumentParser

import pytest
from jsonpath_ng import parse

import ert_shared.status.entity.state as state
from ert_shared.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.model_factory import create_model
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.libres_facade import LibresFacade
from ert_shared.main import ert_parser
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_shared.status.tracker.factory import create_tracker
from res.enkf.enkf_main import EnKFMain
from res.enkf.res_config import ResConfig


def check_expression(original, path_expression, expected, msg_start):
    assert isinstance(original, dict), f"{msg_start}data is not a dict"
    jsonpath_expr = parse(path_expression)
    match_found = False
    for match in jsonpath_expr.find(original):
        match_found = True
        assert (
            match.value == expected
        ), f"{msg_start}{str(match.full_path)} value ({match.value}) is not equal to ({expected})"
    assert match_found, f"{msg_start} Nothing matched {path_expression}"


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "experiment_folder,cmd_line_arguments,num_successful,num_iters,assert_present_in_snapshot",
    [
        pytest.param(
            "max_runtime_poly_example",
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "0,1",
                "max_runtime_poly_example/poly.ert",
            ],
            0,
            1,
            [
                (".*", "reals.*.steps.*.jobs.*.status", state.JOB_STATE_FAILURE),
                (
                    ".*",
                    "reals.*.steps.*.jobs.*.error",
                    "The run is cancelled due to reaching MAX_RUNTIME",
                ),
            ],
            id="ee_poly_experiment_cancelled_by_max_runtime",
        ),
        pytest.param(
            "poly_example",
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "0,1",
                "poly_example/poly.ert",
            ],
            2,
            1,
            [(".*", "reals.*.steps.*.jobs.*.status", state.JOB_STATE_FINISHED)],
            id="ee_poly_experiment",
        ),
        pytest.param(
            "poly_example",
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
            [(".*", "reals.*.steps.*.jobs.*.status", state.JOB_STATE_FINISHED)],
            id="ee_poly_smoother",
        ),
        pytest.param(
            "failing_poly_example",
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "0,1",
                "failing_poly_example/poly.ert",
            ],
            1,
            2,
            [
                ("0", "reals.'0'.steps.*.jobs.'0'.status", state.JOB_STATE_FAILURE),
                ("0", "reals.'0'.steps.*.jobs.'1'.status", state.JOB_STATE_START),
                (".*", "reals.'1'.steps.*.jobs.*.status", state.JOB_STATE_FINISHED),
            ],
            id="ee_failing_poly_smoother",
        ),
    ],
)
def test_tracking(
    experiment_folder,
    cmd_line_arguments,
    num_successful,
    num_iters,
    assert_present_in_snapshot,
    tmpdir,
    source_root,
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", f"{experiment_folder}"),
        os.path.join(str(tmpdir), f"{experiment_folder}"),
    )

    config_lines = ["INSTALL_JOB poly_eval2 POLY_EVAL\n" "SIMULATION_JOB poly_eval2\n"]

    with tmpdir.as_cwd():
        with open(f"{experiment_folder}/poly.ert", "a") as fh:
            fh.writelines(config_lines)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            cmd_line_arguments,
        )
        FeatureToggling.update_from_args(parsed)

        res_config = ResConfig(parsed.config)
        os.chdir(res_config.config_path)
        ert = EnKFMain(res_config, strict=True, verbose=parsed.verbose)
        facade = LibresFacade(ert)

        model = create_model(
            ert,
            facade.get_analysis_module_names,
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

        tracker = create_tracker(
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

        assert tracker._progress() == 1.0

        assert len(snapshots) == num_iters
        for iter_, snapshot in snapshots.items():
            successful_reals = list(
                filter(
                    lambda item: item[1].status == state.REALIZATION_STATE_FINISHED,
                    snapshot.get_reals().items(),
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
